import math
import torch
from torch.optim import Optimizer

class Raven(Optimizer):
    """Implements Raven algorithm.

    This implementation is based on:
    `Adafactor: Adaptive Learning Rates with Sublinear Memory Cost`
    (see https://arxiv.org/abs/1804.04235)

    Note that this optimizer internally adjusts the learning rate
    depending on the *scale_parameter*, *relative_step* and
    *warmup_init* options. To use a manual (external) learning rate
    schedule you should set `scale_parameter=False` and
    `relative_step=False`.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): external learning rate (default: None)
        eps (tuple[float, float]): regularization constants for square gradient
            and parameter scale respectively (default: (1e-30, 1e-3))
        clip_threshold (float): threshold of root mean square of
            final gradient update (default: 1.0)
        decay_rate (float): coefficient used to compute running averages of square
            gradient (default: -0.8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        scale_parameter (bool): if True, learning rate is scaled by root mean square of
            parameter (default: True)
        relative_step (bool): if True, time-dependent learning rate is computed
            instead of external learning rate (default: True)
        warmup_init (bool): time-dependent learning rate computation depends on
            whether warm-up initialization is being used (default: False)
        zero_divisor (float): Additive for rsqrt stability (default: 0.0)
        center_gradients (bool): If True, centers gradients by subtracting mean (default: False)
        adaptive_decay (bool): If True, uses phase-adaptive decay rate for better early/late training (default: True)
        sign_perturbation (bool): If True, applies probabilistic sign-robust perturbation to gradients (default: True)
    """

    def __init__(
        self,
        params,
        lr=None,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        weight_decay=0.0,
        scale_parameter=True,
        relative_step=True,
        warmup_init=False,
        zero_divisor=0.0,
        center_gradients=False,
        adaptive_decay=True,
        sign_perturbation=True,
    ):
        if lr is not None and relative_step:
            raise ValueError("Cannot combine manual lr and relative_step options")
        if warmup_init and not relative_step:
            raise ValueError("warmup_init requires relative_step=True")

        defaults = dict(
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            decay_rate=decay_rate,
            weight_decay=weight_decay,
            scale_parameter=scale_parameter,
            relative_step=relative_step,
            warmup_init=warmup_init,
            zero_divisor=zero_divisor,
            center_gradients=center_gradients,
            adaptive_decay=adaptive_decay,
            sign_perturbation=sign_perturbation,
        )
        super(Raven, self).__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return False

    def _get_lr(self, param_group, param_state):
        rel_step_sz = param_group["lr"]
        if param_group["relative_step"]:
            min_step = (
                1e-6 * param_state["step"] if param_group["warmup_init"] else 1e-2
            )
            rel_step_sz = min(min_step, 1.0 / math.sqrt(param_state["step"]))
        param_scale = 1.0
        if param_group["scale_parameter"]:
            param_scale = max(param_group["eps"][1], param_state["RMS"])
        return param_scale * rel_step_sz

    def _get_options(self, param_group, param_shape):
        factored = len(param_shape) >= 2
        return factored

    def _rms(self, tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    def _approx_sq_grad(self, exp_avg_sq_row, exp_avg_sq_col, zero_divisor):
        # Fused in-place computation for speed
        row_with_eps = exp_avg_sq_row + zero_divisor
        row_normed = row_with_eps.div_(row_with_eps.mean(dim=-1, keepdim=True))
        r_factor = row_normed.rsqrt_().unsqueeze(-1)
        c_factor = (exp_avg_sq_col + zero_divisor).unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        eps0, eps1 = self.defaults["eps"]  # Cache eps for reuse
        for group in self.param_groups:
            do_wd = group["weight_decay"] != 0
            # Generate one random value per group per step
            rand_val = torch.rand(1).item() if group["sign_perturbation"] else 0.0

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                is_low_prec = grad.dtype in {torch.float16, torch.bfloat16}
                if is_low_prec:
                    grad = grad.float()  # Cast once per param

                # Gradient centralization (in-place for speed/stability)
                if group["center_gradients"]:
                    if len(grad.shape) >= 3:  # Spatial (e.g., UNet)
                        mean_grad = grad.mean(dim=[0, 2, 3], keepdim=True)
                    else:
                        mean_grad = grad.mean(dim=0, keepdim=True)
                    torch.sub(grad, mean_grad, out=grad)

                # Sign perturbation (apply before squaring, reuse group rand)
                if group["sign_perturbation"] and rand_val < 0.1:
                    grad.mul_(-1.0)  # Flip sign to mimic Lion's robustness

                state = self.state[p]
                grad_shape = grad.shape
                factored = self._get_options(group, grad_shape)

                # State setup (lazy cast)
                if len(state) == 0:
                    state["step"] = 0
                    state["RMS"] = 0.0
                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1], dtype=grad.dtype, device=grad.device)
                        state["exp_avg_sq_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:], dtype=grad.dtype, device=grad.device)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(grad)

                # Reload state with single cast if needed
                if factored:
                    state["exp_avg_sq_row"] = state["exp_avg_sq_row"].to(grad)
                    state["exp_avg_sq_col"] = state["exp_avg_sq_col"].to(grad)
                else:
                    state["exp_avg_sq"] = state["exp_avg_sq"].to(grad)

                # Param prep (fp32 once)
                p_data_fp32 = p.data.float() if p.data.dtype in {torch.float16, torch.bfloat16} else p.data

                # Core step
                state["step"] += 1
                state["RMS"] = self._rms(p_data_fp32)
                lr = self._get_lr(group, state)
                group["lr"] = lr  # Update group LR

                # Phase-adaptive decay (starts slower, approaches base) if enabled
                if group["adaptive_decay"]:
                    effective_decay = group["decay_rate"] * (1.0 - 0.2 / math.log1p(state["step"]))
                    beta2t = 1.0 - math.pow(state["step"], effective_decay)
                else:
                    beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])
                alpha = 1.0 - beta2t  # Precompute for EMA

                # In-place squared grad + eps
                update = grad.clone()  # Temp for update; reuse where possible
                update.square_().add_(eps0)

                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]
                    exp_avg_sq_row.mul_(beta2t).add_(update.mean(dim=-1), alpha=alpha)
                    exp_avg_sq_col.mul_(beta2t).add_(update.mean(dim=-2), alpha=alpha)
                    exp_avg_sq_row.clamp_min_(eps0)  # Stability clamp
                    exp_avg_sq_col.clamp_min_(eps0)

                    sq_grad_approx = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col, group["zero_divisor"])
                    update.copy_(sq_grad_approx.mul(grad))  # In-place mul
                else:
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2t).add_(update, alpha=alpha)
                    exp_avg_sq.clamp_min_(eps0)  # Stability clamp
                    denom = (exp_avg_sq + group["zero_divisor"]).rsqrt()
                    update.copy_(denom.mul(grad))  # Fused rsqrt-mul

                # Clip and scale (fused)
                clip_factor = self._rms(update) / group["clip_threshold"]
                update.div_(clip_factor.clamp_min_(1.0)).mul_(lr)

                # Stability guard on update
                torch.nan_to_num(update, nan=0.0, posinf=1e3, neginf=-1e3, out=update)

                # Weight decay (decoupled, lr-scaled, matching PyTorch Adafactor)
                if do_wd:
                    p_data_fp32.add_(p_data_fp32, alpha=-group["weight_decay"] * lr)

                # Apply update
                p_data_fp32.add_(-update)

                # Copy back if low prec
                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p.data.copy_(p_data_fp32)

        return loss