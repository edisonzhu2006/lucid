"""Stage 3: autoregressive-in-time, MaskGIT-in-space dynamics over tokens.

Predict frame t's token grid given the previous frames' tokens and the latent
action for each transition. Bidirectional attention over the whole window;
training masks a random cosine-scheduled subset of the target frame's tokens;
generation fills an all-masked frame in a few confidence-ordered steps
(MaskGIT, Chang et al. 2022) so the interactive loop hits real framerates.

Action convention: actions[:, f] is the latent code that PRODUCED frame f
(the f-1 -> f transition); frame 0 of a window takes the null action.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim), nn.Dropout(dropout)
        )

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class DynamicsTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_actions: int,
        tokens_per_frame: int = 64,
        context_frames: int = 8,
        dim: int = 512,
        depth: int = 8,
        heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_id = vocab_size
        self.null_action = num_actions
        self.tokens_per_frame = tokens_per_frame
        self.context_frames = context_frames
        self.token_emb = nn.Embedding(vocab_size + 1, dim)
        self.action_emb = nn.Embedding(num_actions + 1, dim)
        self.spatial_emb = nn.Parameter(torch.zeros(1, 1, tokens_per_frame, dim))
        self.temporal_emb = nn.Parameter(torch.zeros(1, context_frames + 1, 1, dim))
        self.blocks = nn.ModuleList(Block(dim, heads, dropout) for _ in range(depth))
        self.out_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.action_emb.weight, std=0.02)

    def forward(self, tokens: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """tokens (B, F, N) with mask_id allowed; actions (B, F) -> logits (B, F, N, V)."""
        b, f, n = tokens.shape
        x = (
            self.token_emb(tokens)
            + self.spatial_emb
            + self.temporal_emb[:, -f:]
            + self.action_emb(actions).unsqueeze(2)
        )
        x = x.reshape(b, f * n, -1)
        for block in self.blocks:
            x = block(x)
        return self.head(self.out_norm(x)).reshape(b, f, n, -1)

    def mask_final_frame(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Cosine-scheduled random mask on the final frame -> (input tokens, mask)."""
        b, f, n = tokens.shape
        ratio = torch.cos(0.5 * math.pi * torch.rand(b, device=tokens.device))
        num_mask = (ratio * n).long().clamp(min=1)
        scores = torch.rand(b, n, device=tokens.device)
        masked = scores <= scores.sort(-1).values.gather(1, (num_mask - 1)[:, None])
        inp = tokens.clone()
        inp[:, -1][masked] = self.mask_id
        return inp, masked

    def training_loss(self, tokens: torch.Tensor, actions: torch.Tensor,
                      forward=None) -> dict:
        """CE on masked final-frame slots.

        Pass forward=ddp_model under DDP so gradient sync hooks fire.
        """
        inp, masked = self.mask_final_frame(tokens)
        logits = (forward or self)(inp, actions)[:, -1]
        loss = F.cross_entropy(logits[masked], tokens[:, -1][masked])
        with torch.no_grad():
            acc = (logits[masked].argmax(-1) == tokens[:, -1][masked]).float().mean()
        return {"loss": loss, "masked_acc": acc}

    @torch.no_grad()
    def generate(
        self,
        context: torch.Tensor,
        context_actions: torch.Tensor,
        action: torch.Tensor,
        steps: int = 8,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """MaskGIT-decode the next frame.

        context (B, C, N) clean tokens; context_actions (B, C); action (B,)
        for the new transition -> (B, N) tokens.
        """
        b, c, n = context.shape
        device = context.device
        cur = torch.full((b, n), self.mask_id, dtype=torch.long, device=device)
        actions = torch.cat([context_actions, action[:, None]], dim=1)
        for i in range(steps):
            tokens = torch.cat([context, cur[:, None]], dim=1)
            logits = self(tokens, actions)[:, -1] / max(temperature, 1e-6)
            probs = logits.softmax(-1)
            sampled = torch.multinomial(probs.reshape(b * n, -1), 1).reshape(b, n)
            conf = probs.gather(-1, sampled[..., None]).squeeze(-1)
            still_masked = cur == self.mask_id
            conf = torch.where(still_masked, conf, torch.inf)  # keep decided slots
            cur = torch.where(still_masked, sampled, cur)
            keep_ratio = math.cos(0.5 * math.pi * (i + 1) / steps)
            num_remask = int(n * keep_ratio)
            if num_remask > 0 and i < steps - 1:
                remask = conf.argsort(-1)[:, :num_remask]  # lowest confidence
                cur.scatter_(1, remask, self.mask_id)
        return cur

    @torch.no_grad()
    def rollout(
        self,
        context: torch.Tensor,
        context_actions: torch.Tensor,
        action_seq: torch.Tensor,
        steps: int = 8,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Iterate generate() over action_seq (B, T) -> (B, T, N) token frames."""
        frames = []
        ctx, ctx_a = context, context_actions
        for t in range(action_seq.shape[1]):
            nxt = self.generate(ctx, ctx_a, action_seq[:, t], steps, temperature)
            frames.append(nxt)
            ctx = torch.cat([ctx, nxt[:, None]], dim=1)[:, -self.context_frames :]
            ctx_a = torch.cat([ctx_a, action_seq[:, t : t + 1]], dim=1)[:, -self.context_frames :]
        return torch.stack(frames, dim=1)
