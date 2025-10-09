import torch
from torch.optim import Optimizer
import math

class RavenAdamW(Optimizer):
    """
    A VRAM-efficient AdamW optimizer that offloads momentum states to the CPU.

    This optimizer implements the AdamW algorithm exactly but stores the first and
    second momentum terms (exp_avg and exp_avg_sq) on the CPU. During the `step()`
    call, these states are moved to a reusable buffer on the GPU for the update
    calculation and then moved back, minimizing peak VRAM usage.
    """
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.01,
        eps: float = 1e-8,
    ):
        if not 0.0 <= lr: raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0: raise ValueError(f"Betas must be in [0, 1), got {betas}")
        if not 0.0 <= weight_decay: raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, eps=eps)
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
                    # Store momentum on CPU. bfloat16 is efficient for exp_avg.
                    state["exp_avg_cpu"] = torch.zeros_like(p, memory_format=torch.preserve_format, device='cpu', dtype=torch.bfloat16)
                    # Store variance on CPU. float32 provides needed precision.
                    state["exp_avg_sq_cpu"] = torch.zeros_like(p, memory_format=torch.preserve_format, device='cpu', dtype=torch.float32)

                state["step"] += 1
                
                # Retrieve states from CPU
                exp_avg_cpu = state["exp_avg_cpu"]
                exp_avg_sq_cpu = state["exp_avg_sq_cpu"]
                beta1, beta2 = group["betas"]
                
                # Get views into the reusable GPU buffers
                exp_avg_gpu_view = self.reusable_exp_avg_gpu[:num_param_elements].view_as(p)
                exp_avg_sq_gpu_view = self.reusable_exp_avg_sq_gpu[:num_param_elements].view_as(p)

                # Move states from CPU to GPU for calculation
                exp_avg_gpu_view.copy_(exp_avg_cpu, non_blocking=True)
                exp_avg_sq_gpu_view.copy_(exp_avg_sq_cpu, non_blocking=True)
                
                # --- AdamW Core Logic ---
                # 1. Decoupled Weight Decay
                if group["weight_decay"] != 0:
                    p.mul_(1.0 - group["lr"] * group["weight_decay"])

                # 2. Update biased first moment estimate
                exp_avg_gpu_view.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                
                # 3. Update biased second raw moment estimate
                exp_avg_sq_gpu_view.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # 4. Bias correction
                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                
                # 5. Calculate denominator
                denom = (exp_avg_sq_gpu_view.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])
                
                # 6. Final parameter update
                step_size = group["lr"] / bias_correction1
                p.addcdiv_(exp_avg_gpu_view, denom, value=-step_size)
                
                # Move updated states from GPU back to CPU for storage
                exp_avg_cpu.copy_(exp_avg_gpu_view, non_blocking=True)
                exp_avg_sq_cpu.copy_(exp_avg_sq_gpu_view, non_blocking=True)

        # Ensure all async copies are complete before proceeding
        if torch.cuda.is_available(): torch.cuda.synchronize()
        return loss