"""Stage 1: FSQ autoencoder over single frames.

FSQ (Mentzer et al., 2023) replaces the VQ codebook with a fixed grid: each
latent channel is bounded to [-1, 1] and rounded to a small number of levels,
so there is no codebook collapse and no commitment loss — just MSE.

Default levels (8, 6, 5) give a 240-code vocabulary; a 64x64 frame becomes an
8x8 grid of token indices for the dynamics model.
"""

import math

import torch
import torch.nn as nn


def img_to_float(frames: torch.Tensor) -> torch.Tensor:
    """uint8 (..., H, W, C) -> float (..., C, H, W) in [-1, 1]."""
    return frames.float().div(127.5).sub(1.0).movedim(-1, -3)


def float_to_img(x: torch.Tensor) -> torch.Tensor:
    """float (..., C, H, W) in [-1, 1] -> uint8 (..., H, W, C)."""
    return x.movedim(-3, -1).add(1.0).mul(127.5).round().clamp(0, 255).to(torch.uint8)


class FSQ(nn.Module):
    """Finite Scalar Quantization; straight-through gradients."""

    def __init__(self, levels: tuple[int, ...]):
        super().__init__()
        lv = torch.tensor(levels, dtype=torch.float32)
        self.register_buffer("levels", lv)
        self.register_buffer(
            "basis", torch.cumprod(torch.cat([torch.ones(1), lv[:-1]]), dim=0)
        )
        self.dim = len(levels)
        self.codebook_size = int(lv.prod().item())

    def _bound(self, z: torch.Tensor) -> torch.Tensor:
        half_l = (self.levels - 1) * (1 - 1e-3) / 2
        offset = torch.where(self.levels % 2 == 0, 0.5, 0.0)
        shift = torch.atanh(offset / half_l)
        return torch.tanh(z + shift) * half_l - offset

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """(..., dim) unbounded -> quantized, normalized to [-1, 1]."""
        zb = self._bound(z)
        zq = zb + (torch.round(zb) - zb).detach()
        return zq / (self.levels // 2)

    def codes_to_indices(self, zq: torch.Tensor) -> torch.Tensor:
        half_width = self.levels // 2
        digits = torch.round(zq * half_width + half_width)
        return (digits * self.basis).sum(-1).long()

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        digits = (indices.unsqueeze(-1) // self.basis) % self.levels
        return (digits - self.levels // 2) / (self.levels // 2)


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class Tokenizer(nn.Module):
    """64x64x3 frame <-> 8x8 grid of FSQ token indices (3 stride-2 stages)."""

    def __init__(self, levels: tuple[int, ...] = (8, 6, 5), base_ch: int = 64):
        super().__init__()
        self.fsq = FSQ(levels)
        c = base_ch
        self.encoder = nn.Sequential(
            nn.Conv2d(3, c, 3, padding=1),
            ResBlock(c),
            nn.Conv2d(c, 2 * c, 4, stride=2, padding=1),
            ResBlock(2 * c),
            nn.Conv2d(2 * c, 4 * c, 4, stride=2, padding=1),
            ResBlock(4 * c),
            nn.Conv2d(4 * c, 4 * c, 4, stride=2, padding=1),
            ResBlock(4 * c),
            nn.GroupNorm(8, 4 * c),
            nn.SiLU(),
            nn.Conv2d(4 * c, self.fsq.dim, 1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(self.fsq.dim, 4 * c, 1),
            ResBlock(4 * c),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(4 * c, 4 * c, 3, padding=1),
            ResBlock(4 * c),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(4 * c, 2 * c, 3, padding=1),
            ResBlock(2 * c),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(2 * c, c, 3, padding=1),
            ResBlock(c),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, 3, 3, padding=1),
        )

    @property
    def vocab_size(self) -> int:
        return self.fsq.codebook_size

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """image (B, 3, H, W) -> quantized latents (B, fsq_dim, h, w)."""
        z = self.encoder(x)
        return self.fsq(z.movedim(1, -1)).movedim(-1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.quantize(x))

    @torch.no_grad()
    def encode_indices(self, x: torch.Tensor) -> torch.Tensor:
        """image (B, 3, H, W) -> token indices (B, h, w)."""
        return self.fsq.codes_to_indices(self.quantize(x).movedim(1, -1))

    @torch.no_grad()
    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """token indices (B, h, w) -> image (B, 3, H, W) in [-1, 1]."""
        zq = self.fsq.indices_to_codes(indices).movedim(-1, 1)
        return self.decoder(zq)


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Inputs in [-1, 1]."""
    mse = torch.mean((pred - target) ** 2).item()
    return 10 * math.log10(4.0 / max(mse, 1e-10))
