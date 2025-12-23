"""Data generation utilities for the multiscale XOR task.

The implementation mirrors the author-provided logic: a first signal is
presented for ``delay`` time steps on the first channel group and the second
signal is repeatedly presented on the second channel group for ``coding_time``
steps followed by ``remain_time`` steps of silence. The class label at each
step is the XOR of the two signal identities, but only the time steps that fall
in an evaluation window contribute to loss/accuracy (see ``make_eval_mask``).
"""
from __future__ import annotations

import math
import torch


def _sample_spikes(batch: int, steps: int, channels: int, rate: float, device: torch.device) -> torch.Tensor:
    return torch.rand(batch, steps, channels, device=device) <= rate


def generate_batch(
    batch_size: int,
    time_steps: int,
    channel_size: int,
    coding_time: int,
    remain_time: int,
    delay: int,
    low_rate: float,
    high_rate: float,
    noise_rate: float,
    device: torch.device,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a single batch for the multiscale XOR task.

    Args:
        batch_size: number of sequences.
        time_steps: total simulation steps.
        channel_size: number of input channels per signal (input dim = 2 * channel_size).
        coding_time: active duration for the second signal in each cycle.
        remain_time: gap between consecutive second-signal windows.
        delay: duration of the first signal (start_time in the paper/spec).
        low_rate: firing rate for logical 0.
        high_rate: firing rate for logical 1.
        noise_rate: background noise rate applied to all channels.
        device: device for returned tensors.
        seed: optional per-batch seed for deterministic sampling.

    Returns:
        x: (B, T, 2 * channel_size) float32 spike tensor.
        y: (B, T) int64 labels; only evaluation-mask steps are relevant for loss/acc.
    """
    if seed is not None:
        torch.manual_seed(seed)

    if time_steps <= delay:
        raise ValueError("time_steps must exceed delay so that the second signal can appear")

    device = torch.device(device)
    cycle_len = coding_time + remain_time
    total_cycles = max(1, math.ceil((time_steps - delay) / cycle_len))

    # Signal identities
    s1_labels = torch.randint(0, 2, (batch_size,), device=device)
    s2_labels = torch.randint(0, 2, (batch_size, total_cycles), device=device)

    # Base noise across all channels
    inputs = _sample_spikes(batch_size, time_steps, 2 * channel_size, noise_rate, device)
    targets = torch.zeros(batch_size, time_steps, dtype=torch.int64, device=device)

    # First signal occupies the first channel group for `delay` steps
    s1_rates = torch.where(s1_labels == 1, high_rate, low_rate).view(-1, 1, 1)
    s1_spikes = torch.rand(batch_size, delay, channel_size, device=device) <= s1_rates
    inputs[:, :delay, :channel_size] |= s1_spikes

    # Repeated presentation of the second signal
    for cycle_idx in range(total_cycles):
        start = delay + cycle_idx * cycle_len
        end = min(start + coding_time, time_steps)
        if start >= time_steps:
            break
        current_label = s2_labels[:, cycle_idx]
        rate = torch.where(current_label == 1, high_rate, low_rate).view(-1, 1, 1)
        stim = torch.rand(batch_size, end - start, channel_size, device=device) <= rate
        inputs[:, start:end, channel_size:] |= stim
        targets[:, start:end] = (s1_labels.view(-1, 1) ^ current_label.view(-1, 1)).expand(-1, end - start)

    return inputs.float(), targets


def make_eval_mask(time_steps: int, coding_time: int, remain_time: int, delay: int) -> torch.Tensor:
    """Create a boolean mask selecting evaluation steps.

    A step contributes to loss/accuracy if ``t >= delay`` and the step lies in the
    active portion of a cycle (``coding_time`` active followed by ``remain_time`` rest).
    """
    t = torch.arange(time_steps)
    after_delay = t >= delay
    cycle_offset = (t - delay) % (coding_time + remain_time)
    in_active = cycle_offset < coding_time
    return after_delay & in_active
