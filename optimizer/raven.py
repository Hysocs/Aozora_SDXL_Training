import torch
from torch.optim import Optimizer
import math

class RavenAdamW(Optimizer):
    """
    RavenAdamW: High-Performance Optimizer with CPU State Offloading
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
    ):
        if not 0.0 <= lr: 
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0: 
            raise ValueError(f"Betas must be in [0, 1), got {betas}")
        if not 0.0 <= weight_decay: 
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= debias_strength <= 1.0: 
            raise ValueError(f"debias_strength must be between 0.0 and 1.0, got {debias_strength}")
        if use_grad_centralization and not 0.0 <= gc_alpha <= 1.0: 
            raise ValueError(f"gc_alpha must be in [0, 1], got {gc_alpha}")

        defaults = dict(
            lr=lr, 
            betas=betas, 
            weight_decay=weight_decay, 
            eps=eps, 
            debias_strength=debias_strength,
            use_grad_centralization=use_grad_centralization,
            gc_alpha=gc_alpha,
        )
        super(RavenAdamW, self).__init__(params, defaults)

        # Store device for synchronization purposes
        self.param_device = None
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    if self.param_device is None: 
                        self.param_device = p.device
                    break
            if self.param_device is not None:
                break

    def save_cpu_state(self):
        """Returns a state dictionary containing only the essential, CPU-bound tensors."""
        params_with_grad = [p for group in self.param_groups for p in group['params'] if p.requires_grad]
        cpu_state = {}
        for i, p in enumerate(params_with_grad):
            if p in self.state:
                state = self.state[p]
                cpu_state[i] = {
                    'step': state['step'],
                    'exp_avg_cpu': state['exp_avg_cpu'],
                    'exp_avg_sq_cpu': state['exp_avg_sq_cpu']
                }
        return cpu_state

    def load_cpu_state(self, cpu_state):
        """Loads a state dictionary created by `save_cpu_state`, ensuring all tensors remain on the CPU."""
        params_with_grad = [p for group in self.param_groups for p in group['params'] if p.requires_grad]
        for i, p in enumerate(params_with_grad):
            if i in cpu_state:
                saved_param_state = cpu_state[i]
                self.state[p] = {
                    'step': saved_param_state['step'],
                    'exp_avg_cpu': saved_param_state['exp_avg_cpu'].to('cpu', non_blocking=True),
                    'exp_avg_sq_cpu': saved_param_state['exp_avg_sq_cpu'].to('cpu', non_blocking=True)
                }

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, beta1, beta2, weight_decay, eps, debias_strength, use_gc, gc_alpha = (
                group['lr'], group['betas'][0], group['betas'][1], group['weight_decay'], 
                group['eps'], group['debias_strength'], group['use_grad_centralization'], group['gc_alpha']
            )

            for p in group["params"]:
                if p.grad is None: 
                    continue
                
                grad = p.grad.float()
                
                if grad.is_sparse: 
                    raise RuntimeError("RavenAdamW does not support sparse gradients.")
                
                # --- Gradient Centralization ---
                if use_gc and grad.dim() > 1:
                    if grad.dim() >= 3:
                        grad_mean = grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True)
                    else:
                        grad_mean = grad.mean(dim=1, keepdim=True)
                    grad.sub_(grad_mean, alpha=gc_alpha)
                
                state = self.state[p]

                # Initialize state if needed
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg_cpu"] = torch.zeros_like(p, memory_format=torch.preserve_format, device='cpu', dtype=torch.float32)
                    state["exp_avg_sq_cpu"] = torch.zeros_like(p, memory_format=torch.preserve_format, device='cpu', dtype=torch.float32)

                state["step"] += 1
                step, exp_avg_cpu, exp_avg_sq_cpu = state["step"], state["exp_avg_cpu"], state["exp_avg_sq_cpu"]
                
                # Create temporary GPU buffers for this parameter
                exp_avg_gpu = torch.zeros_like(p, dtype=torch.float32)
                exp_avg_sq_gpu = torch.zeros_like(p, dtype=torch.float32)
                
                # Copy momentum states from CPU to GPU
                exp_avg_gpu.copy_(exp_avg_cpu, non_blocking=True)
                exp_avg_sq_gpu.copy_(exp_avg_sq_cpu, non_blocking=True)
                
                # Synchronize to ensure copies complete before computation
                if self.param_device.type == 'cuda': 
                    torch.cuda.synchronize()
                
                # Convert parameter to FP32 for computation
                p_fp32 = p.to(torch.float32)
                
                # Update momentum states
                exp_avg_gpu.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq_gpu.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Compute bias corrections
                bias_correction1 = 1.0 - math.pow(beta1, step)
                bias_correction2 = 1.0 - math.pow(beta2, step)
                
                # Apply debias strength (scales the bias correction)
                if debias_strength < 1.0:
                    bias_correction1 = 1.0 - (1.0 - bias_correction1) * debias_strength
                    bias_correction2 = 1.0 - (1.0 - bias_correction2) * debias_strength
                
                step_size = lr / bias_correction1 if bias_correction1 > 0 else lr
                
                # Compute denominator with bias correction
                denom = (exp_avg_sq_gpu.sqrt() / math.sqrt(bias_correction2) if bias_correction2 > 0 else exp_avg_sq_gpu.sqrt()).add_(eps)
                
                # Apply AdamW update (gradient step)
                p_fp32.addcdiv_(exp_avg_gpu, denom, value=-step_size)
                
                # Apply weight decay (decoupled, after gradient update)
                if weight_decay != 0:
                    p_fp32.mul_(1.0 - lr * weight_decay)
                
                # Copy updated parameter back
                p.copy_(p_fp32)
                
                # Copy updated momentum states back to CPU
                exp_avg_cpu.copy_(exp_avg_gpu, non_blocking=True)
                exp_avg_sq_cpu.copy_(exp_avg_sq_gpu, non_blocking=True)

        # Final synchronization to ensure all copies complete
        if self.param_device is not None and self.param_device.type == 'cuda': 
            torch.cuda.synchronize()
            
        return loss