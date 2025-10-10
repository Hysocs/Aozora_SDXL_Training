import torch
from torch.optim import Optimizer
import math

class RavenAdamW(Optimizer):
    """
    A VRAM-efficient AdamW optimizer that offloads momentum states to the CPU.

    This version has been corrected to fix critical bugs in the original implementation
    and adds a 'debias_strength' parameter for partial bias correction.
    """
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.01,
        eps: float = 1e-8,
        debias_strength: float = 1.0,  # ## NEW ##: Add debias_strength parameter
    ):
        if not 0.0 <= lr: raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0: raise ValueError(f"Betas must be in [0, 1), got {betas}")
        if not 0.0 <= weight_decay: raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        # ## NEW ##: Add validation for the new parameter
        if not 0.0 <= debias_strength <= 1.0: raise ValueError(f"debias_strength must be between 0.0 and 1.0, got {debias_strength}")

        # ## MODIFIED ##: Add debias_strength to the defaults dictionary
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, eps=eps, debias_strength=debias_strength)
        super(RavenAdamW, self).__init__(params, defaults)

        # Find the largest parameter tensor to determine the required buffer size
        max_param_size = 0
        self.param_device = None
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    if self.param_device is None: self.param_device = p.device
                    max_param_size = max(max_param_size, p.numel())

        # Create reusable GPU buffers for the optimizer states
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
                
                grad = p.grad
                if grad.is_sparse: raise RuntimeError("RavenAdamW does not support sparse gradients.")
                
                state = self.state[p]
                num_param_elements = p.numel()

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg_cpu"] = torch.zeros_like(p, memory_format=torch.preserve_format, device='cpu', dtype=torch.float32)
                    state["exp_avg_sq_cpu"] = torch.zeros_like(p, memory_format=torch.preserve_format, device='cpu', dtype=torch.float32)

                state["step"] += 1
                
                exp_avg_cpu = state["exp_avg_cpu"]
                exp_avg_sq_cpu = state["exp_avg_sq_cpu"]
                beta1, beta2 = group["betas"]
                
                exp_avg_gpu_view = self.reusable_exp_avg_gpu[:num_param_elements].view_as(p)
                exp_avg_sq_gpu_view = self.reusable_exp_avg_sq_gpu[:num_param_elements].view_as(p)

                exp_avg_gpu_view.copy_(exp_avg_cpu, non_blocking=True)
                exp_avg_sq_gpu_view.copy_(exp_avg_sq_cpu, non_blocking=True)
                
                # --- AdamW Core Logic ---
                if group["weight_decay"] != 0:
                    p.mul_(1.0 - group["lr"] * group["weight_decay"])

                exp_avg_gpu_view.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq_gpu_view.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # ## MODIFIED ##: Implement partial debiasing
                # This interpolates between no correction (1.0) and full correction.
                debias_strength = group["debias_strength"]
                bias_correction1 = 1.0 - (beta1 ** state["step"]) * debias_strength
                bias_correction2 = 1.0 - (beta2 ** state["step"]) * debias_strength
                
                # Denominator calculation remains the same, but uses the modified bias_correction2
                denom = (exp_avg_sq_gpu_view.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])
                
                # Step size calculation remains the same, but uses the modified bias_correction1
                step_size = group["lr"] / bias_correction1
                p.addcdiv_(exp_avg_gpu_view, denom, value=-step_size)
                
                exp_avg_cpu.copy_(exp_avg_gpu_view, non_blocking=True)
                exp_avg_sq_cpu.copy_(exp_avg_sq_gpu_view, non_blocking=True)

                if torch.cuda.is_available(): torch.cuda.synchronize()

        return loss