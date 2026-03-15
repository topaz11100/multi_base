from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


def gaussian(x: torch.Tensor, mu: float = 0.0, sigma: float = 0.5) -> torch.Tensor:
    # NOTE: torch.sqrt on scalar tensor for device correctness
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(
        2 * torch.tensor(math.pi, device=x.device, dtype=x.dtype)
    ) / sigma


class MultiGaussianSpike(torch.autograd.Function):
    """
    Heaviside spike with Multi-Gaussian surrogate gradient (as used in DH-SNN author code).

    Forward: out = 1[x > 0]
    Backward: grad = grad_output * gamma * MG(x)
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, lens: float = 0.5, gamma: float = 0.5) -> torch.Tensor:
        ctx.save_for_backward(input)
        ctx.lens = float(lens)
        ctx.gamma = float(gamma)
        return (input > 0).to(input.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (input,) = ctx.saved_tensors
        lens = ctx.lens
        gamma = ctx.gamma
        grad_input = grad_output.clone()

        # Multi-Gaussian surrogate (Yin et al. / many SNN works; also in DH-SNN repo)
        scale = 6.0
        hight = 0.15

        temp = gaussian(input, mu=0.0, sigma=lens) * (1.0 + hight) \
            - gaussian(input, mu=lens, sigma=scale * lens) * hight \
            - gaussian(input, mu=-lens, sigma=scale * lens) * hight

        return grad_input * temp.to(grad_output.dtype) * gamma, None, None


def spike_mg(x: torch.Tensor, lens: float = 0.5, gamma: float = 0.5) -> torch.Tensor:
    return MultiGaussianSpike.apply(x, lens, gamma)


class FastSigmoidSpike(torch.autograd.Function):
    """Heaviside spike with fast-sigmoid surrogate: d/dx ~ 1/(1+|x|)^2"""

    @staticmethod
    def forward(ctx, input: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
        ctx.save_for_backward(input)
        ctx.gamma = float(gamma)
        return (input > 0).to(input.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (input,) = ctx.saved_tensors
        gamma = ctx.gamma
        grad = 1.0 / (1.0 + input.abs()) ** 2
        return grad_output * grad.to(grad_output.dtype) * gamma, None


def spike_fast_sigmoid(x: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    return FastSigmoidSpike.apply(x, gamma)


@dataclass
class SpikeFn:
    name: str = "mg"
    lens: float = 0.5
    gamma: float = 0.5
    fs_gamma: float = 1.0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.name == "mg":
            return spike_mg(x, lens=self.lens, gamma=self.gamma)
        if self.name in ("fs", "fast_sigmoid"):
            return spike_fast_sigmoid(x, gamma=self.fs_gamma)
        if self.name in ("linear",):
            # Piecewise linear surrogate: (1 - |x|)+
            out = (x > 0).to(x.dtype)
            # Use straight-through gradient via custom op is better; keep simple:
            return out + (F.relu(1 - x.abs()) - F.relu(1 - x.abs()).detach())  # type: ignore
        raise ValueError(f"Unknown spike surrogate: {self.name}")
