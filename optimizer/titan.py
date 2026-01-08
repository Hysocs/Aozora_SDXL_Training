import torch
from torch.optim import Optimizer
import math

class TitanAdamW(Optimizer):
    """
    TitanAdamW: Extreme VRAM Reduction Optimizer for Large Model Training
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
        if not 0.0 <= lr: raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0: raise ValueError(f"Betas must be in [0, 1), got {betas}")
        if not 0.0 <= weight_decay: raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr, 
            betas=betas, 
            weight_decay=weight_decay, 
            eps=eps, 
            debias_strength=debias_strength,
            use_grad_centralization=use_grad_centralization,
            gc_alpha=gc_alpha,
        )
        super(TitanAdamW, self).__init__(params, defaults)

        # 1. Initialize Reusable GPU Buffers
        max_param_size = 0
        self.param_device = None
        
        # 2. Register Hooks & Find Max Size
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    if self.param_device is None: 
                        self.param_device = p.device
                    
                    max_param_size = max(max_param_size, p.numel())
                    
                    # AUTOMATIC HOOK REGISTRATION
                    # We check if the hook is already registered to avoid duplicates
                    if not hasattr(p, '_titan_hook_registered'):
                        if hasattr(p, 'register_post_accumulate_grad_hook'):
                            p.register_post_accumulate_grad_hook(self._cpu_offload_hook)
                            p._titan_hook_registered = True

        if max_param_size > 0:
            self.reusable_exp_avg_gpu = torch.zeros(max_param_size, device=self.param_device, dtype=torch.float32)
            self.reusable_exp_avg_sq_gpu = torch.zeros(max_param_size, device=self.param_device, dtype=torch.float32)
        else:
            self.reusable_exp_avg_gpu, self.reusable_exp_avg_sq_gpu = None, None

    @staticmethod
    def _cpu_offload_hook(param):
        """Internal hook to move gradients to CPU immediately."""
        if param.grad is not None:
            # Move to CPU (Blocking/Sync default)
            grad_cpu = param.grad.detach().cpu()
            
            # Accumulate if needed (for batch size simulation)
            if hasattr(param, '_cpu_grad') and param._cpu_grad is not None:
                param._cpu_grad.add_(grad_cpu)
            else:
                param._cpu_grad = grad_cpu
                
            # Delete GPU gradient to free VRAM
            param.grad = None

    def zero_grad(self, set_to_none: bool = True):
        """Custom zero_grad to handle the shadow _cpu_grad attribute."""
        for group in self.param_groups:
            for p in group['params']:
                if hasattr(p, '_cpu_grad'):
                    if set_to_none:
                        p._cpu_grad = None
                    else:
                        if p._cpu_grad is not None:
                            p._cpu_grad.zero_()
        super().zero_grad(set_to_none)

    def clip_grad_norm(self, max_norm, norm_type=2.0):
        """
        Internal gradient clipping that understands offloaded CPU gradients.
        Calculates norm on CPU to save VRAM.
        """
        params = [p for group in self.param_groups for p in group['params'] 
                 if hasattr(p, '_cpu_grad') and p._cpu_grad is not None]
        
        if len(params) == 0:
            return torch.tensor(0.0)
            
        norm_type = float(norm_type)
        if norm_type == float('inf'):
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
        """Save state ensuring everything is on CPU."""
        params_with_grad = [p for group in self.param_groups for p in group['params'] if p.requires_grad]
        cpu_state = {}
        for i, p in enumerate(params_with_grad):
            if p in self.state:
                state = self.state[p]
                cpu_state[i] = {
                    'step': state.get('step', 0),
                    'exp_avg_cpu': state.get('exp_avg_cpu'),
                    'exp_avg_sq_cpu': state.get('exp_avg_sq_cpu')
                }
        return cpu_state

    def load_cpu_state(self, cpu_state):
        params_with_grad = [p for group in self.param_groups for p in group['params'] if p.requires_grad]
        for i, p in enumerate(params_with_grad):
            if i in cpu_state:
                saved_param_state = cpu_state[i]
                self.state[p] = {
                    'step': saved_param_state['step'],
                    'exp_avg_cpu': saved_param_state['exp_avg_cpu'].to('cpu') if saved_param_state['exp_avg_cpu'] is not None else None,
                    'exp_avg_sq_cpu': saved_param_state['exp_avg_sq_cpu'].to('cpu') if saved_param_state['exp_avg_sq_cpu'] is not None else None
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
                # 1. Retrieve Gradient (Blocking Copy to GPU)
                grad = None
                if hasattr(p, '_cpu_grad') and p._cpu_grad is not None:
                    # Sync transfer CPU -> GPU
                    grad = p._cpu_grad.to(self.param_device).float()
                    # Do not clear _cpu_grad here; zero_grad handles it
                elif p.grad is not None:
                    grad = p.grad.float()
                
                if grad is None: continue
                
                # --- Math Logic ---
                if use_gc and grad.dim() > 1:
                    if grad.dim() >= 3:
                        grad_mean = grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True)
                    else:
                        grad_mean = grad.mean(dim=1, keepdim=True)
                    grad.sub_(grad_mean, alpha=gc_alpha)
                
                state = self.state[p]
                num_param_elements = p.numel()

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg_cpu"] = torch.zeros_like(p, memory_format=torch.preserve_format, device='cpu', dtype=torch.float32)
                    state["exp_avg_sq_cpu"] = torch.zeros_like(p, memory_format=torch.preserve_format, device='cpu', dtype=torch.float32)

                state["step"] += 1
                step, exp_avg_cpu, exp_avg_sq_cpu = state["step"], state["exp_avg_cpu"], state["exp_avg_sq_cpu"]
                
                # View reusable buffers
                exp_avg_gpu_view = self.reusable_exp_avg_gpu[:num_param_elements].view_as(p)
                exp_avg_sq_gpu_view = self.reusable_exp_avg_sq_gpu[:num_param_elements].view_as(p)
                
                # 2. Sync Copy States CPU -> GPU
                exp_avg_gpu_view.copy_(exp_avg_cpu)
                exp_avg_sq_gpu_view.copy_(exp_avg_sq_cpu)
                
                # Math (FP32)
                p_fp32 = p.to(torch.float32)
                
                if weight_decay != 0:
                    p_fp32.mul_(1.0 - lr * weight_decay)
                
                exp_avg_gpu_view.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq_gpu_view.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - math.pow(beta1, step) * debias_strength if debias_strength > 0 else 1.0
                bias_correction2 = 1.0 - math.pow(beta2, step) * debias_strength if debias_strength > 0 else 1.0
                step_size = lr / bias_correction1 if bias_correction1 != 0 else lr
                
                denom = (exp_avg_sq_gpu_view.sqrt() / (math.sqrt(bias_correction2) if bias_correction2 > 0 else 1.0)).add_(eps)
                
                p_fp32.addcdiv_(exp_avg_gpu_view, denom, value=-step_size)
                p.copy_(p_fp32)
                
                # 3. Sync Copy States GPU -> CPU
                exp_avg_cpu.copy_(exp_avg_gpu_view)
                exp_avg_sq_cpu.copy_(exp_avg_sq_gpu_view)

        return loss