import torch
from torch.optim import Optimizer
import math

class RavenAdamW(Optimizer):
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

        max_param_size = 0
        self.param_device = None
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    if self.param_device is None: 
                        self.param_device = p.device
                    max_param_size = max(max_param_size, p.numel())

        if max_param_size > 0:
            self.reusable_exp_avg_gpu = torch.zeros(
                max_param_size, device=self.param_device, dtype=torch.float32
            )
            self.reusable_exp_avg_sq_gpu = torch.zeros(
                max_param_size, device=self.param_device, dtype=torch.float32
            )
        else:
            self.reusable_exp_avg_gpu = None
            self.reusable_exp_avg_sq_gpu = None

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            weight_decay = group['weight_decay']
            eps = group['eps']
            debias_strength = group['debias_strength']
            use_gc = group['use_grad_centralization']
            gc_alpha = group['gc_alpha']

            for p in group["params"]:
                if p.grad is None: 
                    continue
                
                grad = p.grad.float()
                
                if grad.is_sparse: 
                    raise RuntimeError("RavenAdamW does not support sparse gradients.")
                
                if use_gc and grad.dim() > 1:
                    grad_mean = grad.mean(dim=tuple(range(grad.dim() - 1)), keepdim=True)
                    grad.sub_(grad_mean, alpha=gc_alpha)
                
                state = self.state[p]
                num_param_elements = p.numel()

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg_cpu"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format, 
                        device='cpu', dtype=torch.bfloat16
                    )
                    state["exp_avg_sq_cpu"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format, 
                        device='cpu', dtype=torch.float32
                    )

                state["step"] += 1
                step = state["step"]

                exp_avg_cpu = state["exp_avg_cpu"]
                exp_avg_sq_cpu = state["exp_avg_sq_cpu"]
                
                exp_avg_gpu_view = self.reusable_exp_avg_gpu[:num_param_elements].view_as(p)
                exp_avg_sq_gpu_view = self.reusable_exp_avg_sq_gpu[:num_param_elements].view_as(p)

                exp_avg_gpu_view.copy_(exp_avg_cpu, non_blocking=True)
                exp_avg_sq_gpu_view.copy_(exp_avg_sq_cpu, non_blocking=True)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                p_fp32 = p.to(torch.float32)
                
                if weight_decay != 0:
                    p_fp32.mul_(1.0 - lr * weight_decay)
                
                exp_avg_gpu_view.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq_gpu_view.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0
                bias_correction2 = 1.0
                if debias_strength > 0:
                    bias_correction1 -= math.pow(beta1, step) * debias_strength
                    bias_correction2 -= math.pow(beta2, step) * debias_strength
                
                step_size = lr / bias_correction1 if bias_correction1 != 0 else lr
                
                bias_correction2_sqrt = math.sqrt(bias_correction2) if bias_correction2 > 0 else 1.0
                denom = (exp_avg_sq_gpu_view.sqrt() / bias_correction2_sqrt).add_(eps)
                
                p_fp32.addcdiv_(exp_avg_gpu_view, denom, value=-step_size)
                
                p.copy_(p_fp32)
                
                exp_avg_cpu.copy_(exp_avg_gpu_view, non_blocking=True)
                exp_avg_sq_cpu.copy_(exp_avg_sq_gpu_view, non_blocking=True)

        if torch.cuda.is_available(): 
            torch.cuda.synchronize()

        return loss

    def get_state_for_save(self):
        state_dict = self.state_dict()
        if self.reusable_exp_avg_gpu is not None:
            state_dict['reusable_exp_avg_gpu'] = self.reusable_exp_avg_gpu.clone().cpu()
        else:
            state_dict['reusable_exp_avg_gpu'] = None
            
        if self.reusable_exp_avg_sq_gpu is not None:
            state_dict['reusable_exp_avg_sq_gpu'] = self.reusable_exp_avg_sq_gpu.clone().cpu()
        else:
            state_dict['reusable_exp_avg_sq_gpu'] = None
            
        state_dict['param_device'] = str(self.param_device)
        return state_dict