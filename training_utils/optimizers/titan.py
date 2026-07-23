import math
import weakref

import torch
from torch.optim import Optimizer


class TitanAdamW(Optimizer):
    """
    Raven-style AdamW core with Titan's post-accumulation CPU gradient offload.

    Titan keeps gradients on CPU between backward and optimizer step, then uses
    one reusable FP32 scratch buffer on the training device for AdamW math. CPU
    momentum state can be stored in bfloat16/float16/float32 while updates stay
    in FP32.

    Because this optimizer is special:
    **TitanAdamW — You can train it, but you sure won’t be happy about it.**  
    Need to train on 6 GB when nothing else fits?
    Titan makes the impossible technically possible. 
    Train in hours what everyone else trains in minutes! perfect for the patient, desperate, or dangerously curious.

    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.01,
        eps: float = 1e-8,
        debias_strength: float = 1.0,
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
            momentum_dtype=momentum_dtype,
        )
        super(TitanAdamW, self).__init__(params, defaults)

        self.max_numel = 0
        self.param_device = None
        self._momentum_dtype = momentum_dtype
        self._cpu_grads = {}
        self._cpu_grad_ready = set()
        self._hook_handles = []
        self._closed = False

        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                if self.param_device is None:
                    self.param_device = p.device
                self.max_numel = max(self.max_numel, p.numel())
                self._cpu_grads[p] = torch.empty_like(
                    p,
                    device="cpu",
                    dtype=torch.float32,
                )

                if not hasattr(p, "register_post_accumulate_grad_hook"):
                    raise RuntimeError(
                        "TitanAdamW requires Tensor.register_post_accumulate_grad_hook "
                        "(PyTorch 2.0 or newer)."
                    )

                owner_ref = getattr(p, "_titan_optimizer_owner", None)
                owner = owner_ref() if callable(owner_ref) else None
                if owner is not None and owner is not self:
                    self.close()
                    raise RuntimeError(
                        "A parameter is already owned by another live TitanAdamW. "
                        "Close the old optimizer before creating a replacement."
                    )

                p._titan_optimizer_owner = weakref.ref(self)
                optimizer_ref = weakref.ref(self)

                def offload_hook(param, optimizer_ref=optimizer_ref):
                    optimizer = optimizer_ref()
                    if optimizer is not None:
                        optimizer._offload_gradient(param)

                self._hook_handles.append(
                    p.register_post_accumulate_grad_hook(offload_hook)
                )

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

    def _offload_gradient(self, param):
        if param.grad is None:
            return

        cpu_grad = self._cpu_grads[param]
        if param in self._cpu_grad_ready:
            # Accumulation requires a temporary transfer because the destination
            # already contains gradients from earlier microsteps.
            cpu_grad.add_(param.grad.detach().to(device="cpu", dtype=torch.float32))
        else:
            cpu_grad.copy_(param.grad.detach(), non_blocking=False)
            self._cpu_grad_ready.add(param)
        param.grad = None

    def close(self):
        """Remove autograd hooks and release optimizer-owned gradient buffers."""
        if getattr(self, "_closed", True):
            return
        self._closed = True
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        for p in self._cpu_grads:
            owner_ref = getattr(p, "_titan_optimizer_owner", None)
            if callable(owner_ref) and owner_ref() is self:
                delattr(p, "_titan_optimizer_owner")
        self._cpu_grad_ready.clear()
        self._cpu_grads.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def zero_grad(self, set_to_none: bool = True):
        if set_to_none:
            self._cpu_grad_ready.clear()
        else:
            for p in self._cpu_grad_ready:
                self._cpu_grads[p].zero_()
        super().zero_grad(set_to_none)

    def clip_grad_norm(self, max_norm, norm_type=2.0):
        params = [
            p for group in self.param_groups for p in group["params"]
            if p in self._cpu_grad_ready
        ]
        if not params:
            return torch.tensor(0.0)

        norm_type = float(norm_type)
        if norm_type == float("inf"):
            norms = [self._cpu_grads[p].abs().max() for p in params]
            total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
        else:
            norms = [torch.norm(self._cpu_grads[p], norm_type) for p in params]
            total_norm = torch.norm(torch.stack(norms), norm_type)

        if max_norm > 0:
            clip_coef = max_norm / (total_norm + 1e-6)
            if clip_coef < 1:
                for p in params:
                    self._cpu_grads[p].mul_(clip_coef)

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
            momentum_dtype = group.get("momentum_dtype", self._momentum_dtype)
            wd_factor = 1.0 - lr * weight_decay if weight_decay != 0 else 1.0

            for p in group["params"]:
                grad = None
                if p in self._cpu_grad_ready:
                    grad = self._cpu_grads[p].to(self.param_device, non_blocking=True).float()
                elif p.grad is not None:
                    grad = p.grad.float()
                if grad is None:
                    continue

                buf_exp_avg, buf_exp_avg_sq, buf_p_fp32 = self._get_scratch_buffers(p)

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
