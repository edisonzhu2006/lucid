"""Stage 6 ablation: DIAMOND-style diffusion dynamics in pixel space.

EDM formulation (Karras et al. 2022) as used by DIAMOND (Alonso et al. 2024):
the network denoises the next frame conditioned on the past C frames (channel
concat) and the latent action (embedding -> AdaGN). Sampling needs only a few
Euler steps (DIAMOND ships with 3), so the ablation stays interactive.

Same role as DynamicsTransformer but predicts pixels, not tokens.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierFeatures(nn.Module):
    def __init__(self, dim: int, scale: float = 16.0):
        super().__init__()
        self.register_buffer("freqs", torch.randn(dim // 2) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x[:, None] * self.freqs[None] * 2 * math.pi
        return torch.cat([a.sin(), a.cos()], dim=-1)


class AdaResBlock(nn.Module):
    """Residual block with scale/shift conditioning (AdaGN)."""

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(emb_dim, 2 * out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb(emb)[..., None, None].chunk(2, dim=1)
        h = self.conv2(F.silu(self.norm2(h) * (1 + scale) + shift))
        return self.skip(x) + h


class UNet(nn.Module):
    def __init__(self, in_ch: int, base_ch: int = 64, mults=(1, 2, 4),
                 emb_dim: int = 256, num_actions: int = 8):
        super().__init__()
        self.noise_emb = nn.Sequential(
            FourierFeatures(emb_dim), nn.Linear(emb_dim, emb_dim), nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.action_emb = nn.Embedding(num_actions + 1, emb_dim)
        chans = [base_ch * m for m in mults]
        self.stem = nn.Conv2d(in_ch, chans[0], 3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i, ch in enumerate(chans):
            prev = chans[max(i - 1, 0)]
            self.down_blocks.append(AdaResBlock(prev, ch, emb_dim))
            self.downs.append(
                nn.Conv2d(ch, ch, 4, stride=2, padding=1) if i < len(chans) - 1 else nn.Identity()
            )
        self.mid = AdaResBlock(chans[-1], chans[-1], emb_dim)
        self.up_blocks = nn.ModuleList()
        self.ups = nn.ModuleList()
        for i in reversed(range(len(chans))):
            self.ups.append(
                nn.Upsample(scale_factor=2, mode="nearest") if i < len(chans) - 1 else nn.Identity()
            )
            # incoming h is chans[i] (mid out or previous up out) + chans[i] skip
            self.up_blocks.append(AdaResBlock(2 * chans[i], chans[max(i - 1, 0)], emb_dim))
        self.out_norm = nn.GroupNorm(8, chans[0])
        self.out_conv = nn.Conv2d(chans[0], 3, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x, c_noise, action):
        emb = self.noise_emb(c_noise) + self.action_emb(action)
        h = self.stem(x)
        skips = []
        for block, down in zip(self.down_blocks, self.downs):
            h = block(h, emb)
            skips.append(h)
            h = down(h)
        h = self.mid(h, emb)
        for block, up in zip(self.up_blocks, self.ups):
            h = up(h)
            h = block(torch.cat([h, skips.pop()], dim=1), emb)
        return self.out_conv(F.silu(self.out_norm(h)))


class DiffusionDynamics(nn.Module):
    """context frames (B, C, 3, H, W) + latent action -> next frame (B, 3, H, W)."""

    def __init__(self, context_frames: int = 4, num_actions: int = 8,
                 base_ch: int = 64, mults=(1, 2, 4), emb_dim: int = 256,
                 sigma_data: float = 0.5, p_mean: float = -0.4, p_std: float = 1.2,
                 sigma_min: float = 2e-3, sigma_max: float = 20.0):
        super().__init__()
        self.context_frames = context_frames
        self.null_action = num_actions
        self.sigma_data, self.p_mean, self.p_std = sigma_data, p_mean, p_std
        self.sigma_min, self.sigma_max = sigma_min, sigma_max
        self.net = UNet(3 * (context_frames + 1), base_ch, mults, emb_dim, num_actions)

    def forward(self, x_noisy, sigma, context, action):
        return self.denoise(x_noisy, sigma, context, action)

    def denoise(self, x_noisy, sigma, context, action):
        """EDM-preconditioned denoiser D(x; sigma)."""
        sigma = sigma.reshape(-1, 1, 1, 1)
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (sigma**2 + self.sigma_data**2).sqrt()
        c_noise = sigma.flatten().log() / 4
        inp = torch.cat([c_in * x_noisy, context.flatten(1, 2)], dim=1)
        return c_skip * x_noisy + c_out * self.net(inp, c_noise, action)

    def training_loss(self, context, target, action) -> torch.Tensor:
        b = target.shape[0]
        sigma = (torch.randn(b, device=target.device) * self.p_std + self.p_mean).exp()
        noise = torch.randn_like(target) * sigma.reshape(-1, 1, 1, 1)
        denoised = self.denoise(target + noise, sigma, context, action)
        weight = ((sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2).reshape(-1, 1, 1, 1)
        return (weight * (denoised - target) ** 2).mean()

    def _karras_sigmas(self, steps: int, rho: float = 7.0, device="cpu"):
        ramp = torch.linspace(0, 1, steps, device=device)
        s = (self.sigma_max ** (1 / rho) + ramp * (self.sigma_min ** (1 / rho) - self.sigma_max ** (1 / rho))) ** rho
        return torch.cat([s, torch.zeros(1, device=device)])

    @torch.no_grad()
    def sample(self, context, action, steps: int = 3) -> torch.Tensor:
        """Euler sampler over the Karras schedule -> next frame in [-1, 1]."""
        b, _, _, h, w = context.shape
        sigmas = self._karras_sigmas(steps, device=context.device)
        x = torch.randn(b, 3, h, w, device=context.device) * sigmas[0]
        for i in range(steps):
            sigma = sigmas[i].expand(b)
            d = (x - self.denoise(x, sigma, context, action)) / sigmas[i]
            x = x + (sigmas[i + 1] - sigmas[i]) * d
        return x.clamp(-1, 1)

    @torch.no_grad()
    def rollout(self, context, action_seq, steps: int = 3) -> torch.Tensor:
        """action_seq (B, T) -> imagined frames (B, T, 3, H, W)."""
        frames = []
        for t in range(action_seq.shape[1]):
            nxt = self.sample(context, action_seq[:, t], steps)
            frames.append(nxt)
            context = torch.cat([context, nxt[:, None]], dim=1)[:, -self.context_frames :]
        return torch.stack(frames, dim=1)
