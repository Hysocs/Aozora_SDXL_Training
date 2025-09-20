import torch
from torch.optim import Optimizer
import math

class Raven(Optimizer):
    """
    An ultra-efficient, VRAM-safe AdamW Optimizer with an optional, low-VRAM
    adaptive LR mode. It removes all non-essential features for maximum stability.
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
        use_adaptive_lr: bool = False,
        adaptive_beta: float = 0.99,
        adaptive_increase_factor: float = 1.01,
        adaptive_decrease_factor: float = 0.97,
    ):
        if not 0.0 <= lr: raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0: raise ValueError(f"Betas must be in [0, 1), got {betas}")
        if not 0.0 <= weight_decay: raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if use_grad_centralization and not 0.0 <= gc_alpha <= 1.0: raise ValueError(f"gc_alpha must be in [0, 1], got {gc_alpha}")

        defaults = dict(
            lr=lr, betas=betas, weight_decay=weight_decay, eps=eps,
            use_grad_centralization=use_grad_centralization, gc_alpha=gc_alpha,
            use_adaptive_lr=use_adaptive_lr, adaptive_beta=adaptive_beta,
            adaptive_increase_factor=adaptive_increase_factor, adaptive_decrease_factor=adaptive_decrease_factor,
        )
        super(Raven, self).__init__(params, defaults)

        for group in self.param_groups:
            if group['use_adaptive_lr']:
                group['adaptive_lr'] = group['lr']
                group['smoothed_global_dot_prod'] = 0.0

        max_param_size = 0
        self.param_device = None
        for group in self.param_groups:
            for p in group['params']:
                if self.param_device is None: self.param_device = p.device
                max_param_size = max(max_param_size, p.numel())

        if max_param_size > 0:
            self.reusable_exp_avg_gpu = torch.zeros(max_param_size, device=self.param_device, dtype=torch.float32)
            self.reusable_exp_avg_sq_gpu = torch.zeros(max_param_size, device=self.param_device, dtype=torch.float32)
            # --- FIX: Re-introduce the third buffer. This is necessary for the feature to work. ---
            self.reusable_prev_update_gpu = torch.zeros(max_param_size, device=self.param_device, dtype=torch.float32) if any(g['use_adaptive_lr'] for g in self.param_groups) else None
        else:
            self.reusable_exp_avg_gpu, self.reusable_exp_avg_sq_gpu, self.reusable_prev_update_gpu = None, None, None

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            effective_lr = group.get('adaptive_lr', group['lr']) if group['use_adaptive_lr'] else group['lr']
            global_dot_product_sum = 0.0

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
                    if group['use_adaptive_lr']:
                        state['prev_update_cpu'] = torch.zeros_like(p, memory_format=torch.preserve_format, device='cpu', dtype=torch.bfloat16)

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
                current_update_direction = exp_avg_gpu_view / denom

                if group['use_adaptive_lr'] and state['step'] > 1:
                    # --- FIX: Use the pre-allocated buffer to move prev_update to the GPU ---
                    prev_update_gpu_view = self.reusable_prev_update_gpu[:num_param_elements].view_as(p)
                    prev_update_gpu_view.copy_(state['prev_update_cpu'], non_blocking=True)
                    
                    # Now the dot product is a fast, same-device operation
                    dot_product = torch.dot(current_update_direction.view(-1), prev_update_gpu_view.view(-1))
                    global_dot_product_sum += dot_product

                # Save the current update direction for the next step's calculation
                if group['use_adaptive_lr']:
                    state['prev_update_cpu'].copy_(current_update_direction, non_blocking=True)
                
                if group["weight_decay"] != 0:
                    p.mul_(1.0 - effective_lr * group["weight_decay"])
                
                step_size = effective_lr / bias_correction1
                p.addcdiv_(exp_avg_gpu_view, denom, value=-step_size)
                
                exp_avg_cpu.copy_(exp_avg_gpu_view, non_blocking=True)
                exp_avg_sq_cpu.copy_(exp_avg_sq_gpu_view, non_blocking=True)

            if group['use_adaptive_lr']:
                scheduler_lr_ceiling = group['lr']
                group['smoothed_global_dot_prod'] = group['adaptive_beta'] * group['smoothed_global_dot_prod'] + (1 - group['adaptive_beta']) * global_dot_product_sum
                
                if group['smoothed_global_dot_prod'] > 0:
                    group['adaptive_lr'] *= group['adaptive_increase_factor']
                else:
                    group['adaptive_lr'] *= group['adaptive_decrease_factor']
                
                group['adaptive_lr'] = min(group['adaptive_lr'], scheduler_lr_ceiling)

        if torch.cuda.is_available(): torch.cuda.synchronize()
        return loss