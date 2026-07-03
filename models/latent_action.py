"""Stage 2: unsupervised latent action model (the Genie trick).

Given (x_t, x_{t+1}), an encoder must compress "what changed" into ONE code
from a tiny codebook; a decoder reconstructs x_{t+1} from x_t + that code.
The bottleneck forces the codes to discover the environment's actions with no
action labels anywhere.

Failure mode to watch: codebook collapse (one code takes all). Mitigations
here: EMA codebook updates + dead-code revival; monitor usage entropy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tokenizer import ResBlock


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_codes: int = 8, dim: int = 32, decay: float = 0.95,
                 beta: float = 0.25, revive_frac: float = 0.1):
        super().__init__()
        self.num_codes, self.dim, self.decay, self.beta = num_codes, dim, decay, beta
        self.revive_frac = revive_frac
        embed = torch.randn(num_codes, dim)
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())
        self.register_buffer("cluster_size", torch.ones(num_codes))
        self.register_buffer("inited", torch.zeros((), dtype=torch.bool))

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(B, dim) -> (quantized, indices, commitment loss)."""
        if self.training and not self.inited:
            with torch.no_grad():  # k-means-style init: codes start on the data
                take = torch.randperm(z.shape[0])[: self.num_codes]
                take = take.repeat((self.num_codes // len(take)) + 1)[: self.num_codes]
                self.embed.copy_(z[take])
                self.embed_avg.copy_(z[take])
                self.inited.fill_(True)
        dist = torch.cdist(z, self.embed)
        idx = dist.argmin(-1)
        zq = self.embed[idx]
        if self.training:
            with torch.no_grad():
                onehot = F.one_hot(idx, self.num_codes).float()
                self.cluster_size.mul_(self.decay).add_(onehot.sum(0), alpha=1 - self.decay)
                self.embed_avg.mul_(self.decay).add_(onehot.T @ z, alpha=1 - self.decay)
                n = self.cluster_size.sum()
                smoothed = (self.cluster_size + 1e-5) / (n + self.num_codes * 1e-5) * n
                self.embed.copy_(self.embed_avg / smoothed.unsqueeze(-1))
                # revive codes whose usage fell below revive_frac of uniform share
                dead = self.cluster_size < self.revive_frac * z.shape[0] / self.num_codes
                if dead.any():
                    take = torch.randint(0, z.shape[0], (int(dead.sum()),), device=z.device)
                    self.embed[dead] = z[take]
                    self.embed_avg[dead] = z[take]
                    self.cluster_size[dead] = 1.0
        loss = self.beta * F.mse_loss(z, zq.detach())
        zq = z + (zq - z).detach()
        return zq, idx, loss

    @torch.no_grad()
    def usage_entropy(self) -> float:
        p = self.cluster_size / self.cluster_size.sum()
        return -(p * (p + 1e-10).log()).sum().item()


def _conv_encoder(in_ch: int, c: int) -> nn.Sequential:
    """in_ch x 64 x 64 -> 4c x 8 x 8."""
    return nn.Sequential(
        nn.Conv2d(in_ch, c, 3, padding=1),
        ResBlock(c),
        nn.Conv2d(c, 2 * c, 4, stride=2, padding=1),
        ResBlock(2 * c),
        nn.Conv2d(2 * c, 4 * c, 4, stride=2, padding=1),
        ResBlock(4 * c),
        nn.Conv2d(4 * c, 4 * c, 4, stride=2, padding=1),
        ResBlock(4 * c),
    )


class LatentActionModel(nn.Module):
    def __init__(self, num_codes: int = 8, code_dim: int = 32, base_ch: int = 64):
        super().__init__()
        c = base_ch
        self.num_codes = num_codes
        self.action_encoder = nn.Sequential(
            _conv_encoder(6, c),
            nn.GroupNorm(8, 4 * c),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(4 * c, code_dim),
        )
        self.vq = VectorQuantizerEMA(num_codes, code_dim)
        self.frame_encoder = _conv_encoder(3, c)
        self.action_proj = nn.Linear(code_dim, 4 * c)
        self.decoder = nn.Sequential(
            ResBlock(4 * c),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(4 * c, 2 * c, 3, padding=1),
            ResBlock(2 * c),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(2 * c, c, 3, padding=1),
            ResBlock(c),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(c, c, 3, padding=1),
            ResBlock(c),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, 3, 3, padding=1),
        )

    def decode(self, x_t: torch.Tensor, zq: torch.Tensor) -> torch.Tensor:
        h = self.frame_encoder(x_t) + self.action_proj(zq)[..., None, None]
        return self.decoder(h)

    def forward(self, x_t, x_tp1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """-> (predicted x_{t+1}, code indices, vq loss)."""
        z = self.action_encoder(torch.cat([x_t, x_tp1], dim=1))
        zq, idx, vq_loss = self.vq(z)
        return self.decode(x_t, zq), idx, vq_loss

    @torch.no_grad()
    def encode_action(self, x_t, x_tp1) -> torch.Tensor:
        z = self.action_encoder(torch.cat([x_t, x_tp1], dim=1))
        return self.vq(z)[1]

    @torch.no_grad()
    def decode_with_code(self, x_t: torch.Tensor, code: int) -> torch.Tensor:
        zq = self.vq.embed[code].expand(x_t.shape[0], -1)
        return self.decode(x_t, zq)
