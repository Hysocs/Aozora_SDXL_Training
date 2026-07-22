import torch

from .raven import RavenAdamW


class RavenPinnedAdamW(RavenAdamW):
    """RavenAdamW with page-locked CPU momentum buffers.

    The optimizer math and state precision are identical to RavenAdamW. Only
    the CPU allocation strategy changes, making this an isolated A/B test of
    whether pinned transfers help on the current Windows/CUDA configuration.
    """

    @staticmethod
    def _pinned_empty_like(tensor, dtype):
        if not torch.cuda.is_available():
            return torch.empty_like(tensor, device="cpu", dtype=dtype)
        return torch.empty_like(
            tensor,
            device="cpu",
            dtype=dtype,
            pin_memory=True,
        )

    def _new_cpu_state_tensor(self, p, dtype):
        return self._pinned_empty_like(p, dtype).zero_()

    def _restore_cpu_state_tensor(self, tensor):
        restored = self._pinned_empty_like(tensor, self._momentum_dtype)
        restored.copy_(tensor, non_blocking=False)
        return restored
