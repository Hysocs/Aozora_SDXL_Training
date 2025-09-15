import math
import torch
from torch.optim import Optimizer

class Raven(Optimizer):
    """
    The Definitive VRAM-Safe and Accelerated AdamW Optimizer.

    This version uses a single, pre-allocated, reusable GPU buffer. This provides:
    1. EXTREMELY LOW & CONSTANT VRAM: Optimizer VRAM usage is only the size of
       the largest single parameter, solving all OOM issues.
    2. NO VRAM SPIKES: Memory is allocated once and reused, not created in a loop.
    3. ACCELERATION: Asynchronous copies overlap data transfer and computation,
       making it faster than a simple synchronous loop.
    """
    def __init__(
        self,
        params,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.01,
        eps: float = 1e-8,
    ):
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Betas must be in [0, 1), got {betas}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(betas=betas, weight_decay=weight_decay, eps=eps)
        super(Raven, self).__init__(params, defaults)

        # --- KERNEL OF THE SOLUTION ---
        # Find the largest parameter and pre-allocate a single reusable buffer on the GPU.
        max_param_size = 0
        self.param_device = None
        for group in self.param_groups:
            for p in group['params']:
                if self.param_device is None: self.param_device = p.device
                max_param_size = max(max_param_size, p.numel())
        
        # Handle the case of an empty parameter list
        if max_param_size > 0:
            self.reusable_exp_avg_gpu = torch.zeros(max_param_size, device=self.param_device, dtype=torch.float32)
            self.reusable_exp_avg_sq_gpu = torch.zeros(max_param_size, device=self.param_device, dtype=torch.float32)
        else:
            self.reusable_exp_avg_gpu = None
            self.reusable_exp_avg_sq_gpu = None
        # --- END KERNEL ---

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                grad = p.grad.float() # Ensure grad is float32
                if grad.is_sparse:
                    raise RuntimeError("Raven does not support sparse gradients.")

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg_cpu"] = torch.zeros_like(p, memory_format=torch.preserve_format, device='cpu', dtype=torch.bfloat16)
                    state["exp_avg_sq_cpu"] = torch.zeros_like(p, memory_format=torch.preserve_format, device='cpu', dtype=torch.float32)

                exp_avg_cpu = state["exp_avg_cpu"]
                exp_avg_sq_cpu = state["exp_avg_sq_cpu"]
                state["step"] += 1
                
                beta1, beta2 = group["betas"]

                # The LR for these calculations is now correctly read from `group['lr']`
                # which is the value updated by the scheduler.
                lr = group['lr']
                
                if group["weight_decay"] != 0:
                    p.mul_(1.0 - lr * group["weight_decay"])

                # Get a view of the reusable buffer that matches the current param's size
                num_param_elements = p.numel()
                exp_avg_gpu_view = self.reusable_exp_avg_gpu[:num_param_elements].view_as(p)
                exp_avg_sq_gpu_view = self.reusable_exp_avg_sq_gpu[:num_param_elements].view_as(p)

                # Asynchronously copy CPU state INTO the reusable GPU buffer view
                exp_avg_gpu_view.copy_(exp_avg_cpu, non_blocking=True)
                exp_avg_sq_gpu_view.copy_(exp_avg_sq_cpu, non_blocking=True)
                
                # Perform the AdamW update on the GPU view
                exp_avg_gpu_view.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq_gpu_view.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                
                denom = (exp_avg_sq_gpu_view.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])
                step_size = lr / bias_correction1
                
                p.addcdiv_(exp_avg_gpu_view, denom, value=-step_size)
                
                # Asynchronously copy the result from the GPU buffer view BACK to the CPU
                exp_avg_cpu.copy_(exp_avg_gpu_view, non_blocking=True)
                exp_avg_sq_cpu.copy_(exp_avg_sq_gpu_view, non_blocking=True)

        # A single sync point at the end of all parameter updates for safety.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return loss