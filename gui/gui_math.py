import math


def repeated_image_count(datasets):
    """Return the number of samples seen after applying dataset repeats."""
    return sum(int(dataset.get("image_count", 0)) * int(dataset.get("repeats", 1))
               for dataset in datasets)


def training_calculations(max_steps, gradient_accumulation_steps, batch_size, total_images):
    """Calculate GUI training totals using the sampler's micro-step semantics."""
    max_steps = int(max_steps)
    gradient_accumulation_steps = int(gradient_accumulation_steps)
    batch_size = int(batch_size)
    total_images = int(total_images)

    optimizer_steps = max_steps // gradient_accumulation_steps if gradient_accumulation_steps > 0 else 0
    steps_per_epoch = math.ceil(total_images / batch_size) if total_images > 0 and batch_size > 0 else 0
    epochs = max_steps / steps_per_epoch if steps_per_epoch else math.inf
    return optimizer_steps, steps_per_epoch, epochs


def epoch_marker_interval(max_steps, batch_size, total_images):
    """Return (micro-steps per epoch, number of full epoch markers)."""
    _, steps_per_epoch, _ = training_calculations(max_steps, 1, batch_size, total_images)
    marker_count = (int(max_steps) - 1) // steps_per_epoch if max_steps > 0 and steps_per_epoch else 0
    return steps_per_epoch, marker_count


def logit_shift_ticket_weights(bin_size, shift, total_timesteps=1000):
    """Bin masses reproducing uniform tickets passed through a sigma shift."""
    bin_size = max(1, int(bin_size))
    total_timesteps = max(1, int(total_timesteps))
    shift = max(0.01, float(shift))

    def inverse_shift(y):
        return y / (shift - (shift - 1.0) * y)

    weights = []
    for start in range(0, total_timesteps, bin_size):
        y0 = start / total_timesteps
        y1 = min(start + bin_size, total_timesteps) / total_timesteps
        weights.append(max(0.0, inverse_shift(y1) - inverse_shift(y0)))
    return weights
