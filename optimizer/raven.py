import torch
from torch.optim import Optimizer
import math

class RavenAdamW(Optimizer):
    """
    RavenAdamW v4 (Half-Precision Momentum Edition):
    
    KEY OPTIMIZATION: Store optimizer momentum in bfloat16 (bf16)
    ─────────────────────────────────────────────────────
    
    MEMORY BANDWIDTH SAVINGS:
    - CPU→GPU transfer: 2x faster (half the data)
    - CPU RAM usage: 50% reduction  
    - VRAM scratch: unchanged (still FP32 for precision)
    
    PRECISION STRATEGY:
    - STORE:  bf16 (compact, sufficient range)
    - LOAD:   auto-casts to fp32 on GPU copy
    - COMPUTE: full fp32 (maintains Adam accuracy)
    - SAVE:   casts down to bf16
    
    WHY BF16 NOT FP16?
    - BF16: 8-bit exponent (same range as fp32: ±3.4e38)
    - FP16: 5-bit exponent (max ±65504 - OVERFLOWS easily!)
    - exp_avg_sq can reach large values → BF16 required
    
    COMPATIBILITY:
    - Ampere+ GPUs (RTX 30xx, 40xx): Native BF16 hardware
    - Older GPUs: Software emulation (still works, slightly slower)
    - CPU storage: Always supported (PyTorch handles it)
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.98),
        weight_decay: float = 0.06,
        eps: float = 1e-8,
        debias_strength: float = 0.9,
        use_grad_centralization: bool = False,
        gc_alpha: float = 0.9,
        # NEW: Control momentum storage precision
        momentum_dtype: torch.dtype = torch.bfloat16,  # bf16 default, can use fp32
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        
        # Validate dtype choice
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
            use_grad_centralization=use_grad_centralization,
            gc_alpha=gc_alpha,
            momentum_dtype=momentum_dtype,  # Store in defaults for state_dict
        )
        super(RavenAdamW, self).__init__(params, defaults)

        # Find max parameter size for GPU scratch buffers
        self.max_numel = 0
        self.param_device = None
        
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    if self.param_device is None:
                        self.param_device = p.device
                    self.max_numel = max(self.max_numel, p.numel())

        # Allocate GPU SCRATCH buffers (always FP32 for computation precision)
        # Layout: [exp_avg_fp32 | exp_avg_sq_fp32 | param_fp32_copy]
        if self.max_numel > 0:
            self._scratch_buffer = torch.zeros(
                3 * self.max_numel, 
                device=self.param_device, 
                dtype=torch.float32  # ALWAYS fp32 for accurate math
            )
        else:
            self._scratch_buffer = None
        
        # Store dtype for access in step()
        self._momentum_dtype = momentum_dtype

    def _get_scratch_buffers(self, p):
        """Get FP32 scratch buffer views reshaped for parameter p"""
        numel = p.numel()
        return (
            self._scratch_buffer[:numel].view_as(p),           # exp_avg (fp32 workspace)
            self._scratch_buffer[numel:2*numel].view_as(p),     # exp_avg_sq (fp32 workspace)
            self._scratch_buffer[2*numel:3*numel].view_as(p)    # param fp32 copy
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # Extract settings
            lr = group['lr']
            beta1, beta2 = group['betas']
            weight_decay = group['weight_decay']
            eps = group['eps']
            debias_strength = group['debias_strength']
            use_gc = group['use_grad_centralization']
            gc_alpha = group['gc_alpha']
            momentum_dtype = group.get('momentum_dtype', self._momentum_dtype)
            
            # Pre-compute constants
            wd_factor = 1.0 - lr * weight_decay if weight_decay != 0 else 1.0

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                numel = p.numel()
                
                # Get FP32 scratch buffers
                buf_exp_avg, buf_exp_avg_sq, buf_p_fp32 = self._get_scratch_buffers(p)
                
                # ══════════════════════════════════════════════════
                # PHASE 1: GRADIENT PREPARATION
                # ══════════════════════════════════════════════════
                grad = p.grad.float()
                
                if use_gc and grad.dim() > 1:
                    grad_mean = grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True)
                    grad.sub_(grad_mean, alpha=gc_alpha)
                
                # ══════════════════════════════════════════════════
                # PHASE 2: STATE INITIALIZATION
                # ══════════════════════════════════════════════════
                state = self.state[p]
                
                if "step" not in state:
                    state["step"] = 0
                    # 🔥 KEY OPTIMIZATION: Store in half precision!
                    # Uses 2 bytes/element instead of 4
                    state["exp_avg"] = torch.zeros_like(p, device='cpu', dtype=momentum_dtype)
                    state["exp_avg_sq"] = torch.zeros_like(p, device='cpu', dtype=momentum_dtype)
                
                state["step"] += 1
                step = state["step"]
                
                # ══════════════════════════════════════════════════
                # PHASE 3: LOAD STATE (CPU half → GPU FP32)
                # ══════════════════════════════════════════════════
                # Auto upcast: bf16/fp16 tensor → fp32 buffer
                # PyTorch handles dtype promotion during copy_
                # TRANSFER SIZE: HALVED (2 bytes vs 4 bytes per element)!
                buf_exp_avg.copy_(state["exp_avg"], non_blocking=True)
                buf_exp_avg_sq.copy_(state["exp_avg_sq"], non_blocking=True)
                
                # Copy parameter to fp32 workspace
                buf_p_fp32.copy_(p, non_blocking=True)
                
                # ══════════════════════════════════════════════════
                # PHASE 4: MOMENTUM UPDATE (Full FP32 precision)
                # ══════════════════════════════════════════════════
                # Standard Adam momentum (unchanged - must be precise)
                buf_exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                buf_exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                
                # ══════════════════════════════════════════════════
                # PHASE 5: DEBIASING & PARAMETER UPDATE
                # ══════════════════════════════════════════════════
                bc1 = 1.0 - beta1 ** step
                bc2 = 1.0 - beta2 ** step
                
                if debias_strength < 1.0:
                    bc1 = 1.0 - (1.0 - bc1) * debias_strength
                    bc2 = 1.0 - (1.0 - bc2) * debias_strength
                
                sqrt_bc2 = math.sqrt(bc2)
                step_size = lr / bc1
                
                # Weight Decay
                if weight_decay != 0:
                    buf_p_fp32.mul_(wd_factor)
                
                # Adam update
                denom = buf_exp_avg_sq.sqrt().div_(sqrt_bc2).add_(eps)
                buf_p_fp32.addcdiv_(buf_exp_avg, denom, value=-step_size)
                
                # Write updated parameter
                p.copy_(buf_p_fp32)
                
                # ══════════════════════════════════════════════════
                # PHASE 6: SAVE STATE (GPU FP32 → CPU half precision)
                # ══════════════════════════════════════════════════
                # 🔥 KEY: Downcast before saving!
                # fp32 buffer → bf16/fp16 CPU tensor
                # Halves transfer size AND storage footprint
                
                if momentum_dtype == torch.float32:
                    # No casting needed for fp32 mode
                    state["exp_avg"].copy_(buf_exp_avg, non_blocking=True)
                    state["exp_avg_sq"].copy_(buf_exp_avg_sq, non_blocking=True)
                else:
                    # Downcast to half precision for storage
                    # Using .to(dtype) ensures proper conversion
                    state["exp_avg"].copy_(
                        buf_exp_avg.to(momentum_dtype), 
                        non_blocking=True
                    )
                    state["exp_avg_sq"].copy_(
                        buf_exp_avg_sq.to(momentum_dtype), 
                        non_blocking=True
                    )

        return loss

    # ══════════════════════════════════════════════════════════════
    # CHECKPOINTING SUPPORT (handles dtype correctly)
    # ══════════════════════════════════════════════════════════════
    
    def state_dict(self):
        """Save state preserving momentum dtype"""
        state_dict = super().state_dict()
        
        # Add metadata about storage dtype for proper restoration
        state_dict['_momentum_dtype'] = self._momentum_dtype
        
        return state_dict

    def load_state_dict(self, state_dict):
        """Load state handling potential dtype mismatches"""
        
        # Handle legacy state dicts without dtype info
        saved_dtype = state_dict.pop('_momentum_dtype', torch.float32)
        
        # If dtypes don't match, warn and convert
        if saved_dtype != self._momentum_dtype:
            print(
                f"[RavenAdamW] Loading state saved in {saved_dtype}, "
                f"but current optimizer uses {self._momentum_dtype}. "
                f"Converting..."
            )
            
        super().load_state_dict(state_dict)
        
        # Convert loaded states to correct dtype if needed
        if saved_dtype != self._momentum_dtype:
            for group in self.param_groups:
                for p in group['params']:
                    if p.requires_grad and p in self.state:
                        state = self.state[p]
                        if 'exp_avg' in state:
                            state['exp_avg'] = state['exp_avg'].to(self._momentum_dtype)
                            state['exp_avg_sq'] = state['exp_avg_sq'].to(self._momentum_dtype)