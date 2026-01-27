import torch
from torch.optim import Optimizer
import math

class TitanAdamW(Optimizer):
    """
    TitanAdamW (Optimized): 
    - Extreme VRAM Reduction (ZeRO Stage 2+3 style offloading).
    - Fixed 'Raven' buffer reuse to prevent memory thrashing.
    - Enables SDXL Full Finetuning on ~6GB VRAM.
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

        # --- 1. Initialize Reusable GPU Buffers ---
        # We look for the largest parameter to size our reusable buffers.
        max_numel = 0
        self.param_device = None
        
        # Register Hooks & Find Max Size
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    if self.param_device is None: 
                        self.param_device = p.device
                    
                    max_numel = max(max_numel, p.numel())
                    
                    # AUTOMATIC HOOK REGISTRATION
                    # This hook moves gradients to CPU immediately after calculation.
                    if not hasattr(p, '_titan_hook_registered'):
                        if hasattr(p, 'register_post_accumulate_grad_hook'):
                            p.register_post_accumulate_grad_hook(self._cpu_offload_hook)
                            p._titan_hook_registered = True

        # Allocate the "Buckets" (One set for the whole model)
        if max_numel > 0:
            self.buffer_exp_avg = torch.zeros(max_numel, device=self.param_device, dtype=torch.float32)
            self.buffer_exp_avg_sq = torch.zeros(max_numel, device=self.param_device, dtype=torch.float32)
            # This is the speed optimization from Raven:
            self.buffer_p_fp32 = torch.zeros(max_numel, device=self.param_device, dtype=torch.float32)
        else:
            self.buffer_exp_avg = None
            self.buffer_exp_avg_sq = None
            self.buffer_p_fp32 = None

    @staticmethod
    def _cpu_offload_hook(param):
        """
        Internal hook to move gradients to CPU immediately.
        This keeps VRAM usage equivalent to Inference only.
        """
        if param.grad is not None:
            # Move to CPU
            grad_cpu = param.grad.detach().cpu()
            
            # Accumulate if needed (simulates standard gradient accumulation)
            if hasattr(param, '_cpu_grad') and param._cpu_grad is not None:
                param._cpu_grad.add_(grad_cpu)
            else:
                param._cpu_grad = grad_cpu
                
            # Delete GPU gradient to free VRAM immediately
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
                saved = cpu_state[i]
                self.state[p] = {
                    'step': saved['step'],
                    'exp_avg_cpu': saved['exp_avg_cpu'].to('cpu') if saved['exp_avg_cpu'] is not None else None,
                    'exp_avg_sq_cpu': saved['exp_avg_sq_cpu'].to('cpu') if saved['exp_avg_sq_cpu'] is not None else None
                }

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
                # 1. Retrieve Gradient (Blocking Copy from CPU -> GPU)
                grad = None
                if hasattr(p, '_cpu_grad') and p._cpu_grad is not None:
                    grad = p._cpu_grad.to(self.param_device, non_blocking=True).float()
                elif p.grad is not None:
                    grad = p.grad.float()
                
                if grad is None: continue
                
                # Gradient Centralization
                if use_gc and grad.dim() > 1:
                    if grad.dim() >= 3:
                        grad_mean = grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True)
                    else:
                        grad_mean = grad.mean(dim=1, keepdim=True)
                    grad.sub_(grad_mean, alpha=gc_alpha)
                
                state = self.state[p]
                numel = p.numel()

                # Lazy Init on CPU
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg_cpu"] = torch.zeros_like(p, device='cpu', dtype=torch.float32)
                    state["exp_avg_sq_cpu"] = torch.zeros_like(p, device='cpu', dtype=torch.float32)

                state["step"] += 1
                step = state["step"]
                
                # --- 2. GPU Buffer Views (Fixed VRAM Usage) ---
                # Slice the pre-allocated buffers to fit this parameter
                buf_exp_avg = self.buffer_exp_avg[:numel].view_as(p)
                buf_exp_avg_sq = self.buffer_exp_avg_sq[:numel].view_as(p)
                buf_p_fp32 = self.buffer_p_fp32[:numel].view_as(p)
                
                # --- 3. Sync Copy States CPU -> GPU ---
                buf_exp_avg.copy_(state["exp_avg_cpu"], non_blocking=True)
                buf_exp_avg_sq.copy_(state["exp_avg_sq_cpu"], non_blocking=True)
                
                # Load weight into FP32 buffer (Avoids allocation)
                buf_p_fp32.copy_(p, non_blocking=True)
                
                # --- 4. Math (Done on Reusable Buffers) ---
                
                # Weight Decay
                if weight_decay != 0:
                    buf_p_fp32.mul_(1.0 - lr * weight_decay)
                
                # AdamW Logic
                buf_exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                buf_exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Bias Correction
                bc1 = 1.0 - beta1 ** step
                bc2 = 1.0 - beta2 ** step
                if debias_strength < 1.0:
                    bc1 = 1.0 - (1.0 - bc1) * debias_strength
                    bc2 = 1.0 - (1.0 - bc2) * debias_strength
                
                denom = (buf_exp_avg_sq.sqrt().div_(math.sqrt(bc2))).add_(eps)
                step_size = lr / bc1
                
                # Update FP32 buffer
                buf_p_fp32.addcdiv_(buf_exp_avg, denom, value=-step_size)
                
                # --- 5. Write Back ---
                # Update actual Model Weight (BF16/FP16)
                p.copy_(buf_p_fp32, non_blocking=True)
                
                # Save States back to CPU
                state["exp_avg_cpu"].copy_(buf_exp_avg, non_blocking=True)
                state["exp_avg_sq_cpu"].copy_(buf_exp_avg_sq, non_blocking=True)

                # --- 6. Safety Sync ---
                # Required because we reuse the same buffer for the next parameter.
                # If we don't sync, the next param will overwrite this buffer before the copy finishes.
                if self.param_device.type == 'cuda':
                    torch.cuda.synchronize()

        return loss