import torch


def _xor_labels(channel_rate: tuple[float, float]):
    label = torch.zeros(len(channel_rate), len(channel_rate), dtype=torch.int64)
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a single delayed XOR batch.

    Args:
        batch_size: number of sequences.
        delay: gap (in steps) between the two coding windows.
        time_steps: total simulation steps (must allow both coding blocks).
        channel_size: number of input channels.
        coding_time: duration of each coding window.
        noise_rate: background spike probability.
        channel_rate: spike probability for low/high states.

    Returns:
        inputs: (B, T, C) binary spikes.
        targets: (B, T) integer class labels with valid steps marked after second cue.
        valid_mask: (B, T) bool mask selecting steps contributing to the loss.
    """
    device = torch.device("cpu")
    if time_steps < coding_time * 2 + delay:
        raise ValueError("time_steps too small for given delay and coding_time")

    label_matrix = _xor_labels(channel_rate)

    inputs = (torch.rand(batch_size, time_steps, channel_size) <= noise_rate).to(device)
    targets = torch.zeros(batch_size, time_steps, dtype=torch.int64, device=device)
    valid_mask = torch.zeros(batch_size, time_steps, dtype=torch.bool, device=device)

    init_pattern = torch.randint(len(channel_rate), size=(batch_size,))
    init_probs = torch.tensor(channel_rate)[init_pattern]
    prob_matrix = torch.ones(coding_time, channel_size, batch_size) * init_probs
    add_patterns = torch.bernoulli(prob_matrix).permute(2, 0, 1).bool()
    inputs[:, :coding_time, :] |= add_patterns

    pattern = torch.randint(len(channel_rate), size=(batch_size,))
    delayed_probs = torch.tensor(channel_rate)[pattern]
    delayed_matrix = torch.ones(coding_time, channel_size, batch_size) * delayed_probs
    delayed_patterns = torch.bernoulli(delayed_matrix).permute(2, 0, 1).bool()

    start = coding_time + delay
    end = start + coding_time

    if end > time_steps:
        raise ValueError("Delayed window exceeds time_steps; increase time_steps or decrease delay.")

    for i in range(batch_size):
        inputs[i, start:end, :] |= delayed_patterns[i]
        targets[i, start:,] = label_matrix[init_pattern[i], pattern[i]]
        valid_mask[i, start:] = True

    return inputs.float(), targets, valid_mask
