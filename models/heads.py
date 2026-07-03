"""Reward / value / policy heads with DreamerV3 stability tricks.

Rewards and values are predicted as two-hot categorical distributions over
symlog-spaced bins (symlog + two-hot regression, Hafner et al. 2023) so sparse
and large-magnitude returns train stably. Final layers are zero-initialized so
initial predictions are exactly 0.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def two_hot(x: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Scalar batch (B,) -> (B, num_bins) two-hot weights over `bins`."""
    x = x.clamp(bins[0], bins[-1])
    idx = torch.searchsorted(bins, x.contiguous()).clamp(1, len(bins) - 1)
    lo, hi = bins[idx - 1], bins[idx]
    w_hi = ((x - lo) / (hi - lo).clamp(min=1e-8)).clamp(0, 1)
    out = torch.zeros(*x.shape, len(bins), device=x.device)
    out.scatter_(-1, (idx - 1)[..., None], (1 - w_hi)[..., None])
    out.scatter_add_(-1, idx[..., None], w_hi[..., None])
    return out


def _mlp(in_dim: int, hidden: int, out_dim: int, zero_last: bool) -> nn.Sequential:
    last = nn.Linear(hidden, out_dim)
    if zero_last:
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), last
    )


class DistHead(nn.Module):
    """Two-hot symlog regression head (reward or value)."""

    def __init__(self, in_dim: int, hidden: int = 512, num_bins: int = 255,
                 low: float = -20.0, high: float = 20.0):
        super().__init__()
        self.register_buffer("bins", torch.linspace(low, high, num_bins))
        self.net = _mlp(in_dim, hidden, num_bins, zero_last=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        """Expected value in real (symexp'd) units."""
        return symexp((logits.softmax(-1) * self.bins).sum(-1))

    def loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """CE against the two-hot encoding of symlog(target)."""
        t = two_hot(symlog(target.detach()), self.bins)
        return -(t * logits.log_softmax(-1)).sum(-1).mean()


class PolicyHead(nn.Module):
    def __init__(self, in_dim: int, num_actions: int, hidden: int = 512):
        super().__init__()
        self.net = _mlp(in_dim, hidden, num_actions, zero_last=True)

    def forward(self, state: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.net(state))


class StateEncoder(nn.Module):
    """Frame token grid (B, N) -> agent state vector (B, out_dim)."""

    def __init__(self, vocab_size: int, tokens_per_frame: int = 64,
                 dim: int = 256, out_dim: int = 512):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.pos = nn.Parameter(torch.zeros(1, tokens_per_frame, dim))
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim)
        )
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net((self.emb(tokens) + self.pos).mean(dim=1))
