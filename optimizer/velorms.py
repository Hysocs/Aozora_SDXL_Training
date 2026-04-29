import math

import torch
from torch.optim import Optimizer


class VeloRMS(Optimizer):
    """
    VeloRMS: CPU-offloaded RMSProp.

    This is intentionally plain: keep RMSProp-style gradient normalization,
    store optimizer state on CPU, and apply the normalized gradient directly.
    It is a clean non-AdamW baseline for rectified-flow optimizer experiments.
    """
    RMS_BETA = 0.99

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        verbose: bool = False,
        log_every: int = 100,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid lr: {lr}")

        defaults = dict(
            lr=lr,
            eps=eps,
            weight_decay=weight_decay,
        )
        super(VeloRMS, self).__init__(params, defaults)

        self.param_device = None
        self.max_numel = 0
        self.step_counter = 0
        self.verbose = verbose
        self.log_every = max(1, int(log_every))

        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                if self.param_device is None:
                    self.param_device = p.device
                self.max_numel = max(self.max_numel, p.numel())

        if self.max_numel > 0:
            self._scratch_buffer = torch.zeros(
                2 * self.max_numel,
                device=self.param_device,
                dtype=torch.float32,
            )
        else:
            self._scratch_buffer = None

        print("\n" + "=" * 70)
        print("VeloRMS OPTIMIZER - CPU OFFLOADED RMSPROP")
        print(f"  lr={lr}, rms_beta={self.RMS_BETA}, eps={eps}")
        print("  State stored on CPU, computation on GPU")
        print(f"  Verbose logging: {'ENABLED' if verbose else 'DISABLED'}")
        print("=" * 70)

    def _get_scratch_buffers(self, p):
        numel = p.numel()
        return (
            self._scratch_buffer[:numel].view_as(p),
            self._scratch_buffer[numel:2 * numel].view_as(p),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.step_counter += 1

        if self.verbose:
            total_params = 0
            avg_grad_mag = 0.0
            avg_rms = 0.0
            avg_update = 0.0

        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]
            rms_beta = self.RMS_BETA
            wd_factor = 1.0 - lr * wd if wd > 0 else 1.0

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.float()
                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0
                    state["rms_cpu"] = torch.zeros_like(p, device="cpu", dtype=torch.float32)

                state["step"] += 1

                buf_rms, buf_fp32 = self._get_scratch_buffers(p)
                buf_rms.copy_(state["rms_cpu"], non_blocking=True)
                buf_fp32.copy_(p, non_blocking=True)

                buf_rms.mul_(rms_beta).addcmul_(grad, grad, value=1.0 - rms_beta)

                rms_correction = 1.0 - rms_beta ** state["step"]
                denom = buf_rms.sqrt().div_(math.sqrt(max(rms_correction, 1e-12))).add_(eps)
                update = grad / denom

                if wd > 0:
                    buf_fp32.mul_(wd_factor)
                buf_fp32.add_(update, alpha=-lr)

                p.copy_(buf_fp32, non_blocking=True)
                state["rms_cpu"].copy_(buf_rms, non_blocking=True)

                if self.verbose:
                    numel = p.numel()
                    total_params += numel
                    avg_grad_mag += grad.abs().mean().item() * numel
                    avg_rms += denom.mean().item() * numel
                    avg_update += update.abs().mean().item() * numel

        if self.param_device is not None and self.param_device.type == "cuda":
            torch.cuda.synchronize()

        if self.verbose and self.step_counter % self.log_every == 0 and total_params > 0:
            avg_grad_mag /= total_params
            avg_rms /= total_params
            avg_update /= total_params
            print(f"\n[VeloRMS STEP {self.step_counter:6d}]")
            print(f"  Avg Grad Mag   : {avg_grad_mag:.6f}")
            print(f"  Avg RMS Denom  : {avg_rms:.6f}")
            print(f"  Avg Update     : {avg_update:.6f}")
            print(f"  Effective LR   : ~{lr * avg_update:.6f}")

        return loss

    def save_cpu_state(self):
        cpu_state = {}
        for i, group in enumerate(self.param_groups):
            for j, p in enumerate(group["params"]):
                if p.requires_grad and p in self.state:
                    state = self.state[p]
                    cpu_state[(i, j)] = {
                        "step": state.get("step", 0),
                        "rms_cpu": state.get("rms_cpu"),
                    }
        return cpu_state

    def load_cpu_state(self, cpu_state):
        for i, group in enumerate(self.param_groups):
            for j, p in enumerate(group["params"]):
                if (i, j) not in cpu_state:
                    continue
                data = cpu_state[(i, j)]
                self.state[p] = {
                    "step": data.get("step", 0),
                    "rms_cpu": data["rms_cpu"],
                }
