import torch


def _xor_labels(channel_rate: tuple[float, float], device: torch.device) -> torch.Tensor:
    label = torch.zeros(len(channel_rate), len(channel_rate), dtype=torch.int64, device=device)
    label[1, 0] = 1
    label[0, 1] = 1
    return label


def generate_batch(
    batch_size: int,
    delay: int,
    time_steps: int,
    channel_size: int,
    coding_time: int,
    noise_rate: float = 0.01,
    channel_rate: tuple[float, float] = (0.2, 0.6),
    *,
    device: torch.device,
    generator: torch.Generator | None = None,
    target_mode: str = "last_step",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a single delayed XOR batch on the requested device.

    Args:
        batch_size: number of sequences.
        delay: gap (in steps) between the two coding windows.
        time_steps: total simulation steps (must allow both coding blocks).
        channel_size: number of input channels.
        coding_time: duration of each coding window.
        noise_rate: background spike probability.
        channel_rate: spike probability for low/high states (must be length 2).
        device: target device for all returned tensors.
        generator: optional torch.Generator for reproducible sampling.
        target_mode: "after_cue" masks all steps after the second cue; "last_step"
            keeps only the final step (default, matching the DH-SNN delayed XOR
            evaluation).

    Returns:
        inputs: (B, T, C) binary spikes on ``device``.
        targets: (B, T) integer class labels.
        valid_mask: (B, T) bool mask selecting steps contributing to the loss.
    """
    assert len(channel_rate) == 2, "Delayed XOR expects exactly two input channels"
    if time_steps < coding_time * 2 + delay:
        raise ValueError("time_steps too small for given delay and coding_time")

    label_matrix = _xor_labels(channel_rate, device=device)

    noise = torch.rand(
        batch_size, time_steps, channel_size, device=device, generator=generator
    )
    inputs = (noise <= noise_rate)

    targets = torch.zeros(batch_size, time_steps, dtype=torch.int64, device=device)
    valid_mask = torch.zeros(batch_size, time_steps, dtype=torch.bool, device=device)

    init_pattern = torch.randint(
        len(channel_rate), size=(batch_size,), device=device, generator=generator
    )
    delayed_pattern = torch.randint(
        len(channel_rate), size=(batch_size,), device=device, generator=generator
    )

    start = coding_time + delay
    end = start + coding_time
    if end > time_steps:
        raise ValueError(
            "Delayed window exceeds time_steps; increase time_steps or decrease delay."
        )

    channel_tensor = torch.tensor(channel_rate, device=device)
    init_probs = channel_tensor[init_pattern].view(batch_size, 1, 1)
    delayed_probs = channel_tensor[delayed_pattern].view(batch_size, 1, 1)

    init_noise = torch.rand(
        batch_size, coding_time, channel_size, device=device, generator=generator
    )
    delayed_noise = torch.rand(
        batch_size, coding_time, channel_size, device=device, generator=generator
    )

    inputs[:, :coding_time, :] |= init_noise <= init_probs
    inputs[:, start:end, :] |= delayed_noise <= delayed_probs

    labels = label_matrix[init_pattern, delayed_pattern]
    targets[:, start:] = labels.unsqueeze(1)

    if target_mode == "after_cue":
        valid_mask[:, start:] = True
    elif target_mode == "last_step":
        valid_mask[:, -1] = True
    else:
        raise ValueError(f"Unknown target_mode: {target_mode}")

    return inputs.float(), targets, valid_mask
