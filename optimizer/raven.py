import torch
from torch.optim import Optimizer
import math

class Raven(Optimizer):
    """
    An ultra-efficient, VRAM-safe AdamW Optimizer. It removes all
    non-essential features for maximum stability.
    """
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.01,
        eps: float = 1e-8,
        use_grad_centralization: bool = False,
        gc_alpha: float = 1.0,
    ):
        if not 0.0 <= lr: raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0: raise ValueError(f"Betas must be in [0, 1), got {betas}")
        if not 0.0 <= weight_decay: raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if use_grad_centralization and not 0.0 <= gc_alpha <= 1.0: raise ValueError(f"gc_alpha must be in [0, 1], got {gc_alpha}")

        defaults = dict(
            lr=lr, betas=betas, weight_decay=weight_decay, eps=eps,
            use_grad_centralization=use_grad_centralization, gc_alpha=gc_alpha,
        )
        super(Raven, self).__init__(params, defaults)

        max_param_size = 0
        self.param_device = None
        for group in self.param_groups:
            for p in group['params']:
                if self.param_device is None: self.param_device = p.device
                max_param_size = max(max_param_size, p.numel())

        if max_param_size > 0:
            self.reusable_exp_avg_gpu = torch.zeros(max_param_size, device=self.param_device, dtype=torch.float32)
            self.reusable_exp_avg_sq_gpu = torch.zeros(max_param_size, device=self.param_device, dtype=torch.float32)
        else:
            self.reusable_exp_avg_gpu, self.reusable_exp_avg_sq_gpu = None, None

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                grad = p.grad.float()
                if grad.is_sparse: raise RuntimeError("Raven does not support sparse gradients.")
                if group['use_grad_centralization'] and grad.dim() > 1:
                    grad.sub_(grad.mean(dim=tuple(range(grad.dim() - 1)), keepdim=True), alpha=group['gc_alpha'])

                state = self.state[p]
                num_param_elements = p.numel()

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg_cpu"] = torch.zeros_like(p, memory_format=torch.preserve_format, device='cpu', dtype=torch.bfloat16)
                    state["exp_avg_sq_cpu"] = torch.zeros_like(p, memory_format=torch.preserve_format, device='cpu', dtype=torch.float32)

                state["step"] += 1
                
                exp_avg_cpu = state["exp_avg_cpu"]
                exp_avg_sq_cpu = state["exp_avg_sq_cpu"]
                beta1, beta2 = group["betas"]
                
                exp_avg_gpu_view = self.reusable_exp_avg_gpu[:num_param_elements].view_as(p)
                exp_avg_sq_gpu_view = self.reusable_exp_avg_sq_gpu[:num_param_elements].view_as(p)

                exp_avg_gpu_view.copy_(exp_avg_cpu, non_blocking=True)
                exp_avg_sq_gpu_view.copy_(exp_avg_sq_cpu, non_blocking=True)
                
                exp_avg_gpu_view.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq_gpu_view.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                
                denom = (exp_avg_sq_gpu_view.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])
                
                if group["weight_decay"] != 0:
                    p.mul_(1.0 - group["lr"] * group["weight_decay"])
                
                step_size = group["lr"] / bias_correction1
                p.addcdiv_(exp_avg_gpu_view, denom, value=-step_size)
                
                exp_avg_cpu.copy_(exp_avg_gpu_view, non_blocking=True)
                exp_avg_sq_cpu.copy_(exp_avg_sq_gpu_view, non_blocking=True)

        if torch.cuda.is_available(): torch.cuda.synchronize()
        return loss