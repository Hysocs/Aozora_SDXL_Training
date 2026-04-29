import math

import torch
from torch.optim import Optimizer


class TitanAdamW(Optimizer):
    """
    Raven-style AdamW core with Titan's post-accumulation CPU gradient offload.

    Titan keeps gradients on CPU between backward and optimizer step, then uses
    one reusable FP32 scratch buffer on the training device for AdamW math. CPU
    momentum state can be stored in bfloat16/float16/float32 while updates stay
    in FP32.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.01,
        eps: float = 1e-8,
        debias_strength: float = 1.0,
        use_grad_centralization: bool = False,
        gc_alpha: float = 1.0,
        momentum_dtype: torch.dtype = torch.bfloat16,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")

        valid_dtypes = [torch.float32, torch.float16, torch.bfloat16]
        if momentum_dtype not in valid_dtypes:
            raise ValueError(
                f"momentum_dtype must be one of {valid_dtypes}, got {momentum_dtype}"
            )

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
            debias_strength=debias_strength,
            use_grad_centralization=use_grad_centralization,
            gc_alpha=gc_alpha,
            momentum_dtype=momentum_dtype,
        )
        super(TitanAdamW, self).__init__(params, defaults)

        self.max_numel = 0
        self.param_device = None
        self._momentum_dtype = momentum_dtype

        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                if self.param_device is None:
                    self.param_device = p.device
                self.max_numel = max(self.max_numel, p.numel())

                if not hasattr(p, "_titan_hook_registered"):
                    if hasattr(p, "register_post_accumulate_grad_hook"):
                        p.register_post_accumulate_grad_hook(self._cpu_offload_hook)
                        p._titan_hook_registered = True

        if self.max_numel > 0:
            self._scratch_buffer = torch.zeros(
                3 * self.max_numel,
                device=self.param_device,
                dtype=torch.float32,
            )
        else:
            self._scratch_buffer = None

    def _get_scratch_buffers(self, p):
        numel = p.numel()
        return (
            self._scratch_buffer[:numel].view_as(p),
            self._scratch_buffer[numel:2 * numel].view_as(p),
            self._scratch_buffer[2 * numel:3 * numel].view_as(p),
        )

    @staticmethod
    def _cpu_offload_hook(param):
        if param.grad is None:
            return

        grad_cpu = param.grad.detach().to(device="cpu", dtype=torch.float32)
        if hasattr(param, "_cpu_grad") and param._cpu_grad is not None:
            param._cpu_grad.add_(grad_cpu)
        else:
            param._cpu_grad = grad_cpu
        param.grad = None

    def zero_grad(self, set_to_none: bool = True):
        for group in self.param_groups:
            for p in group["params"]:
                if hasattr(p, "_cpu_grad"):
                    if set_to_none:
                        p._cpu_grad = None
                    elif p._cpu_grad is not None:
                        p._cpu_grad.zero_()
        super().zero_grad(set_to_none)

    def clip_grad_norm(self, max_norm, norm_type=2.0):
        params = [
            p for group in self.param_groups for p in group["params"]
            if hasattr(p, "_cpu_grad") and p._cpu_grad is not None
        ]
        if not params:
            return torch.tensor(0.0)

        norm_type = float(norm_type)
        if norm_type == float("inf"):
            norms = [p._cpu_grad.abs().max() for p in params]
            total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
        else:
            norms = [torch.norm(p._cpu_grad, norm_type) for p in params]
            total_norm = torch.norm(torch.stack(norms), norm_type)

        if max_norm > 0:
            clip_coef = max_norm / (total_norm + 1e-6)
            if clip_coef < 1:
                for p in params:
                    p._cpu_grad.mul_(clip_coef)

        return total_norm

    def save_cpu_state(self):
        params_with_grad = [
            p for group in self.param_groups for p in group["params"] if p.requires_grad
        ]
        cpu_state = {"_momentum_dtype": self._momentum_dtype}
        for i, p in enumerate(params_with_grad):
            if p in self.state:
                state = self.state[p]
                cpu_state[i] = {
                    "step": state.get("step", 0),
                    "exp_avg_cpu": state.get("exp_avg"),
                    "exp_avg_sq_cpu": state.get("exp_avg_sq"),
                }
        return cpu_state

    def load_cpu_state(self, cpu_state):
        saved_dtype = cpu_state.get("_momentum_dtype", self._momentum_dtype)
        params_with_grad = [
            p for group in self.param_groups for p in group["params"] if p.requires_grad
        ]
        for i, p in enumerate(params_with_grad):
            if i not in cpu_state:
                continue
            saved = cpu_state[i]
            exp_avg = saved.get("exp_avg", saved.get("exp_avg_cpu"))
            exp_avg_sq = saved.get("exp_avg_sq", saved.get("exp_avg_sq_cpu"))
            self.state[p] = {
                "step": saved.get("step", 0),
                "exp_avg": (
                    exp_avg.to(device="cpu", dtype=self._momentum_dtype)
                    if exp_avg is not None else None
                ),
                "exp_avg_sq": (
                    exp_avg_sq.to(device="cpu", dtype=self._momentum_dtype)
                    if exp_avg_sq is not None else None
                ),
            }

        if saved_dtype != self._momentum_dtype:
            print(
                f"[TitanAdamW] Loaded state saved in {saved_dtype}, "
                f"converted to {self._momentum_dtype}."
            )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            debias_strength = group["debias_strength"]
            use_gc = group["use_grad_centralization"]
            gc_alpha = group["gc_alpha"]
            momentum_dtype = group.get("momentum_dtype", self._momentum_dtype)
            wd_factor = 1.0 - lr * weight_decay if weight_decay != 0 else 1.0

            for p in group["params"]:
                grad = None
                if hasattr(p, "_cpu_grad") and p._cpu_grad is not None:
                    grad = p._cpu_grad.to(self.param_device, non_blocking=True).float()
                elif p.grad is not None:
                    grad = p.grad.float()
                if grad is None:
                    continue

                buf_exp_avg, buf_exp_avg_sq, buf_p_fp32 = self._get_scratch_buffers(p)

                if use_gc and grad.dim() > 1:
                    grad_mean = grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True)
                    grad.sub_(grad_mean, alpha=gc_alpha)

                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p, device="cpu", dtype=momentum_dtype
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, device="cpu", dtype=momentum_dtype
                    )

                state["step"] += 1
                step = state["step"]

                buf_exp_avg.copy_(state["exp_avg"], non_blocking=True)
                buf_exp_avg_sq.copy_(state["exp_avg_sq"], non_blocking=True)
                buf_p_fp32.copy_(p, non_blocking=True)

                buf_exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                buf_exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bc1 = 1.0 - beta1 ** step
                bc2 = 1.0 - beta2 ** step
                if debias_strength < 1.0:
                    bc1 = 1.0 - (1.0 - bc1) * debias_strength
                    bc2 = 1.0 - (1.0 - bc2) * debias_strength

                if weight_decay != 0:
                    buf_p_fp32.mul_(wd_factor)

                denom = buf_exp_avg_sq.sqrt().div_(math.sqrt(bc2)).add_(eps)
                buf_p_fp32.addcdiv_(buf_exp_avg, denom, value=-(lr / bc1))
                p.copy_(buf_p_fp32, non_blocking=True)

                state["exp_avg"].copy_(buf_exp_avg, non_blocking=True)
                state["exp_avg_sq"].copy_(buf_exp_avg_sq, non_blocking=True)

                if self.param_device is not None and self.param_device.type == "cuda":
                    torch.cuda.synchronize()

        return loss
