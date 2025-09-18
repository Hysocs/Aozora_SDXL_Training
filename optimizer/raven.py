import torch
from torch.optim import Optimizer
from collections import deque
import numpy as np
import math

class Raven(Optimizer):
    """
    The Definitive VRAM-Safe and Accelerated AdamW Optimizer.

    This version uses a single, pre-allocated, reusable GPU buffer for AdamW states,
    offloading them to the CPU to save VRAM.

    V2 Features:
    - Integrated Lookahead mechanism for proactive stabilization.
    - Integrated Adaptive Dampening mechanism with a dual-condition trigger
      to prevent reactive spikes without choking the optimizer during fine-tuning.
    """
    def __init__(
        self,
        params,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.01,
        eps: float = 1e-8,
        # --- Lookahead Parameters ---
        use_lookahead: bool = False,
        la_steps: int = 6,
        la_alpha: float = 0.5,
        # --- Adaptive Dampening Parameters ---
        use_adaptive_dampening: bool = False,
        ad_dampening_factor: float = 0.1,
        ad_sigma_threshold: float = 3.0,
        ad_percentile_threshold: float = 95.0, # The fix for the "choking" issue
        ad_history_window: int = 100,
    ):
        # --- Validation ---
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Betas must be in [0, 1), got {betas}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if use_lookahead:
            if not 0.0 <= la_alpha < 1.0: raise ValueError(f"Lookahead alpha must be in [0, 1)")
            if not la_steps >= 1: raise ValueError(f"Lookahead steps must be >= 1")
        if use_adaptive_dampening:
            if not 0.0 < ad_dampening_factor <= 1.0: raise ValueError(f"Dampening factor must be in (0, 1]")
            if not ad_sigma_threshold > 0: raise ValueError(f"Sigma threshold must be > 0")
            if not 0.0 < ad_percentile_threshold < 100.0: raise ValueError(f"Percentile threshold must be in (0, 100)")
            if not ad_history_window > 20: raise ValueError(f"History window must be > 20 for stable stats")

        defaults = dict(
            betas=betas, weight_decay=weight_decay, eps=eps,
            use_lookahead=use_lookahead, la_steps=la_steps, la_alpha=la_alpha,
            use_adaptive_dampening=use_adaptive_dampening, ad_dampening_factor=ad_dampening_factor,
            ad_sigma_threshold=ad_sigma_threshold, ad_percentile_threshold=ad_percentile_threshold,
        )
        super(Raven, self).__init__(params, defaults)

        # --- Adaptive Dampening State ---
        if use_adaptive_dampening:
            self.grad_norm_history = deque(maxlen=ad_history_window)

        # --- VRAM-Saving Kernel ---
        max_param_size = 0
        self.param_device = None
        for group in self.param_groups:
            for p in group['params']:
                if self.param_device is None: self.param_device = p.device
                max_param_size = max(max_param_size, p.numel())

        if max_param_size > 0:
            self.reusable_exp_avg_gpu = torch.zeros(max_param_size, device=self.param_device, dtype=torch.float32)
            self.reusable_exp_avg_sq_gpu = torch.zeros(max_param_size, device=self.param_device, dtype=torch.float32)
        else:
            self.reusable_exp_avg_gpu = None
            self.reusable_exp_avg_sq_gpu = None

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # --- Adaptive Dampening Calculation (runs once per step) ---
        lr_dampening_factor = 1.0
        if self.defaults.get('use_adaptive_dampening'):
            params_with_grad = [p for group in self.param_groups for p in group['params'] if p.grad is not None]
            if params_with_grad:
                total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2).cpu() for p in params_with_grad]), 2).item()
                
                # Check for outliers only if we have enough history for stable statistics
                if len(self.grad_norm_history) >= self.grad_norm_history.maxlen:
                    # Condition 1: Is the spike statistically significant (relative check)?
                    mean_norm = np.mean(self.grad_norm_history)
                    std_norm = np.std(self.grad_norm_history)
                    statistical_threshold = mean_norm + self.defaults['ad_sigma_threshold'] * std_norm
                    is_stat_significant = total_norm > statistical_threshold
                    
                    # Condition 2: Is the spike absolutely large (minimum threshold check)?
                    # This prevents the system from choking on small gradients.
                    absolute_threshold = np.percentile(self.grad_norm_history, self.defaults['ad_percentile_threshold'])
                    is_absolutely_large = total_norm > absolute_threshold

                    if is_stat_significant and is_absolutely_large:
                        lr_dampening_factor = self.defaults['ad_dampening_factor']
                        print(f"\n[RAVEN EYE] Grad norm spike detected! Norm: {total_norm:.2f} > (Stat Thresh: {statistical_threshold:.2f} AND Abs Thresh: {absolute_threshold:.2f}). Dampening LR.")

                self.grad_norm_history.append(total_norm)

        # --- Main Parameter Loop ---
        for group in self.param_groups:
            effective_lr = group['lr'] * lr_dampening_factor

            for p in group["params"]:
                if p.grad is None: continue
                grad = p.grad.float()
                if grad.is_sparse:
                    raise RuntimeError("Raven does not support sparse gradients.")

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg_cpu"] = torch.zeros_like(p, memory_format=torch.preserve_format, device='cpu', dtype=torch.bfloat16)
                    state["exp_avg_sq_cpu"] = torch.zeros_like(p, memory_format=torch.preserve_format, device='cpu', dtype=torch.float32)
                    if group['use_lookahead']:
                        state['slow_param_cpu'] = torch.empty_like(p, device='cpu').copy_(p.data)
                        state['la_step_counter'] = 0

                state["step"] += 1
                
                if group["weight_decay"] != 0:
                    p.mul_(1.0 - effective_lr * group["weight_decay"])

                # --- Core AdamW Update ---
                exp_avg_cpu = state["exp_avg_cpu"]
                exp_avg_sq_cpu = state["exp_avg_sq_cpu"]
                beta1, beta2 = group["betas"]
                
                num_param_elements = p.numel()
                exp_avg_gpu_view = self.reusable_exp_avg_gpu[:num_param_elements].view_as(p)
                exp_avg_sq_gpu_view = self.reusable_exp_avg_sq_gpu[:num_param_elements].view_as(p)

                exp_avg_gpu_view.copy_(exp_avg_cpu, non_blocking=True)
                exp_avg_sq_gpu_view.copy_(exp_avg_sq_cpu, non_blocking=True)
                
                exp_avg_gpu_view.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq_gpu_view.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                
                denom = (exp_avg_sq_gpu_view.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])
                step_size = effective_lr / bias_correction1
                
                p.addcdiv_(exp_avg_gpu_view, denom, value=-step_size)
                
                exp_avg_cpu.copy_(exp_avg_gpu_view, non_blocking=True)
                exp_avg_sq_cpu.copy_(exp_avg_sq_gpu_view, non_blocking=True)

                # --- Lookahead Update ---
                if group['use_lookahead']:
                    state['la_step_counter'] += 1
                    if state['la_step_counter'] >= group['la_steps']:
                        slow_p_cpu = state['slow_param_cpu']
                        fast_p_cpu = p.data.cpu()
                        slow_p_cpu.add_(fast_p_cpu - slow_p_cpu, alpha=group['la_alpha'])
                        p.data.copy_(slow_p_cpu)
                        state['la_step_counter'] = 0

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return loss