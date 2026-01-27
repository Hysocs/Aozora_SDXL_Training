import torch
from torch.optim import Optimizer

class VeloRMS(Optimizer):
    """
    VeloRMS - Minimal Adaptive Optimizer with CPU Offloading
    
    Core idea: Velocity + RMS normalization of velocity, with a small gradient leakage
    to the RMS for stability on sparse/rare updates and reduced initial overshoot.
    - Memory efficient: Uses CPU for state storage, reusable GPU buffers
    - Stable: RMS normalization + leakage prevents explosions on infrequent params
    - Adaptive: Per-parameter effective learning rate
    """
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        momentum: float = 0.86,
        leak: float = 0.16,         # Increased for stronger damping of outliers/warping
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        verbose: bool = False,      # Toggle for logging statistics
        log_every: int = 1
    ):
        if not 0.0 <= lr: raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= momentum < 1.0: raise ValueError(f"Invalid momentum: {momentum}")
        if leak < 0.0: raise ValueError(f"Invalid leak: {leak}")
        
        defaults = dict(lr=lr, momentum=momentum, leak=leak, eps=eps, weight_decay=weight_decay)
        super(VeloRMS, self).__init__(params, defaults)

        self.param_device = None
        self.step_counter = 0
        self.verbose = verbose
        self.log_every = log_every
        
        # --- Reusable GPU Buffers ---
        max_numel = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    if self.param_device is None:
                        self.param_device = p.device
                    max_numel = max(max_numel, p.numel())

        if max_numel > 0:
            self.buffer_vel = torch.zeros(max_numel, device=self.param_device, dtype=torch.float32)
            self.buffer_rms = torch.zeros(max_numel, device=self.param_device, dtype=torch.float32)
            self.buffer_fp32 = torch.zeros(max_numel, device=self.param_device, dtype=torch.float32)
        else:
            self.buffer_vel = None
            self.buffer_rms = None
            self.buffer_fp32 = None
        
        print("\n" + "="*70)
        print("VeloRMS OPTIMIZER - RMS WITH CPU OFFLOADING + GRAD LEAKAGE")
        print(f"  lr={lr}, momentum={momentum}, leak={leak}, eps={eps}")
        print("  State stored on CPU, computation on GPU")
        print(f"  Verbose logging: {'ENABLED' if verbose else 'DISABLED'}")
        print("="*70)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        self.step_counter += 1
        
        # Diagnostics (only computed if verbose=True)
        if self.verbose:
            total_params = 0
            avg_grad_mag = 0.0
            avg_vel_mag = 0.0
            avg_rms = 0.0
            avg_update = 0.0
            max_update = 0.0
        
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            leak = group['leak']
            eps = group['eps']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.float()
                numel = p.numel()
                state = self.state[p]
                
                # --- Initialize state on CPU ---
                if len(state) == 0:
                    state['velocity_cpu'] = torch.zeros_like(p, device='cpu', dtype=torch.float32)
                    state['v_rms_cpu'] = torch.zeros_like(p, device='cpu', dtype=torch.float32)
                    state['master_cpu'] = p.detach().clone().float().cpu()
                
                # --- Load state into GPU buffers ---
                buf_vel = self.buffer_vel[:numel].view_as(p)
                buf_rms = self.buffer_rms[:numel].view_as(p)
                buf_fp32 = self.buffer_fp32[:numel].view_as(p)
                
                buf_vel.copy_(state['velocity_cpu'], non_blocking=True)
                buf_rms.copy_(state['v_rms_cpu'], non_blocking=True)
                buf_fp32.copy_(state['master_cpu'], non_blocking=True)
                
                # --- 1. Update velocity with momentum (scaled to prevent unbounded growth) ---
                buf_vel.mul_(momentum).add_(grad, alpha=1 - momentum)
                
                # --- Diagnostics: capture magnitudes after velocity update ---
                if self.verbose:
                    grad_mag = grad.abs().mean().item()
                    vel_mag = buf_vel.abs().mean().item()
                    avg_grad_mag += grad_mag * numel
                    avg_vel_mag += vel_mag * numel
                
                # --- 2. Update RMS of velocity + small grad**2 leakage ---
                buf_rms.mul_(momentum).addcmul_(buf_vel, buf_vel, value=1 - momentum)
                if leak > 0.0:
                    buf_rms.addcmul_(grad, grad, value=leak)
                
                # --- 3. Normalize velocity by RMS ---
                denom = buf_rms.sqrt().add_(eps)
                update = buf_vel / denom
                
                # --- 4. Apply weight decay (decoupled) ---
                if wd > 0:
                    buf_fp32.mul_(1 - lr * wd)
                
                # --- 5. Apply update ---
                buf_fp32.add_(update, alpha=-lr)
                
                # --- 6. Write back to model and CPU state ---
                p.copy_(buf_fp32)
                state['master_cpu'].copy_(buf_fp32, non_blocking=True)
                state['velocity_cpu'].copy_(buf_vel, non_blocking=True)
                state['v_rms_cpu'].copy_(buf_rms, non_blocking=True)
                
                # --- More diagnostics ---
                if self.verbose:
                    total_params += numel
                    avg_rms += denom.mean().item() * numel
                    update_mag = update.abs().mean().item()
                    avg_update += update_mag * numel
                    max_update = max(max_update, update.abs().max().item())
        
        if self.param_device and self.param_device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Logging (only if verbose=True)
        if self.verbose and self.step_counter % self.log_every == 0 and total_params > 0:
            avg_grad_mag /= total_params
            avg_vel_mag /= total_params
            avg_rms /= total_params
            avg_update /= total_params
            vel_grad_ratio = avg_vel_mag / (avg_grad_mag + 1e-8)
            
            print(f"\n[VeloRMS STEP {self.step_counter:6d}]")
            print(f"  Avg Grad Mag   : {avg_grad_mag:.6f}")
            print(f"  Avg Vel Mag    : {avg_vel_mag:.6f}  (post-velocity update, pre-norm)")
            print(f"  Vel/Grad Ratio : {vel_grad_ratio:.3f}x  <-- if >>2-3x, velocity buildup (risk of overshoot)")
            print(f"  Avg RMS        : {avg_rms:.6f} (normalization denominator)")
            print(f"  Avg Update     : {avg_update:.6f} (normalized update magnitude)")
            print(f"  Max Update     : {max_update:.6f} (largest single update) <-- if consistently high, likely warping source")
            print(f"  Effective LR   : ~{lr * avg_update:.6f}")
        
        return loss
    
    def save_cpu_state(self):
        """Save CPU state for checkpointing"""
        cpu_state = {}
        for i, group in enumerate(self.param_groups):
            for j, p in enumerate(group['params']):
                if p.requires_grad and p in self.state:
                    state = self.state[p]
                    cpu_state[(i, j)] = {
                        'velocity_cpu': state.get('velocity_cpu'),
                        'v_rms_cpu': state.get('v_rms_cpu'),
                        'master_cpu': state.get('master_cpu')
                    }
        return cpu_state

    def load_cpu_state(self, cpu_state):
        """Load CPU state from checkpoint"""
        for i, group in enumerate(self.param_groups):
            for j, p in enumerate(group['params']):
                if (i, j) in cpu_state:
                    data = cpu_state[(i, j)]
                    self.state[p] = {
                        'velocity_cpu': data['velocity_cpu'],
                        'v_rms_cpu': data['v_rms_cpu'],
                        'master_cpu': data['master_cpu']
                    }