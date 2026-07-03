"""Stage 3 entrypoint: train the MaskGIT dynamics transformer over tokens.

Requires the frozen Stage-1 tokenizer and Stage-2 latent action model.
Gate: multi-step rollouts stay coherent for tens of frames — inspect
rollout_*.png strips.

Run:  python -m train.train_dynamics
"""

import itertools

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from data.datamodule import make_dataloader
from models.dynamics_transformer import DynamicsTransformer
from models.latent_action import LatentActionModel
from models.tokenizer import Tokenizer, float_to_img, img_to_float
from train import common


def load_frozen(ckpt_path, cls, key="model", device="cpu", **overrides):
    ckpt = common.load_checkpoint(ckpt_path, map_location=device)
    kwargs = {**ckpt["cfg"]["model"], **overrides}
    if cls is Tokenizer:
        kwargs = {"levels": tuple(kwargs["levels"]), "base_ch": kwargs["base_ch"]}
    model = cls(**kwargs).to(device)
    model.load_state_dict(ckpt[key])
    model.eval().requires_grad_(False)
    return model


@torch.no_grad()
def encode_batch(tokenizer, lam, frames, null_action):
    """frames uint8 (B, T, H, W, C) -> (tokens (B, T, N), actions (B, T))."""
    b, t = frames.shape[:2]
    x = img_to_float(frames).flatten(0, 1)
    tokens = tokenizer.encode_indices(x).flatten(1).reshape(b, t, -1)
    pairs_t = x.reshape(b, t, *x.shape[1:])
    acts = lam.encode_action(
        pairs_t[:, :-1].flatten(0, 1), pairs_t[:, 1:].flatten(0, 1)
    ).reshape(b, t - 1)
    null = torch.full((b, 1), null_action, dtype=torch.long, device=acts.device)
    return tokens, torch.cat([null, acts], dim=1)


@torch.no_grad()
def save_rollout_strip(model, tokenizer, tokens, actions, horizon, decode_steps, path):
    """Ground-truth context + imagined continuation, one row per sample."""
    import imageio.v3 as iio

    c = model.context_frames
    ctx, ctx_a = tokens[:4, :c], actions[:4, :c]
    action_seq = actions[:4, -1:].expand(-1, horizon).contiguous()  # repeat last action
    gen = model.rollout(ctx, ctx_a, action_seq, steps=decode_steps)
    all_tokens = torch.cat([ctx, gen], dim=1)
    b, t, n = all_tokens.shape
    side = int(n**0.5)
    decoded = tokenizer.decode_indices(all_tokens.reshape(b * t, side, side)).clamp(-1, 1)
    h, w = decoded.shape[-2:]
    imgs = float_to_img(decoded).cpu().numpy().reshape(b, t, h, w, 3)
    rows = [np.concatenate(list(r), axis=1) for r in imgs]
    iio.imwrite(path, np.concatenate(rows, axis=0))


@hydra.main(version_base=None, config_path="../configs", config_name="dynamics")
def main(cfg: DictConfig) -> None:
    device, rank, world = common.setup(cfg.seed)
    out_dir = common.resolve_dir(cfg.out_dir)
    run = common.maybe_wandb(cfg, rank)

    tokenizer = load_frozen(common.resolve_dir(cfg.tokenizer_ckpt), Tokenizer, device=device)
    lam = load_frozen(common.resolve_dir(cfg.lam_ckpt), LatentActionModel, device=device)

    seq_len = cfg.model.context_frames + 1
    sample = tokenizer.encode_indices(torch.zeros(1, 3, 64, 64, device=device))
    tokens_per_frame = sample.numel()
    model = DynamicsTransformer(
        vocab_size=tokenizer.vocab_size,
        num_actions=lam.num_codes,
        tokens_per_frame=tokens_per_frame,
        context_frames=cfg.model.context_frames,
        dim=cfg.model.dim,
        depth=cfg.model.depth,
        heads=cfg.model.heads,
        dropout=cfg.model.dropout,
    ).to(device)
    model = common.wrap_ddp(model, device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / cfg.warmup_steps)
    )

    loader = make_dataloader(
        common.resolve_dir(cfg.data_dir),
        batch_size=cfg.batch_size,
        seq_len=seq_len,
        num_workers=cfg.num_workers,
        seed=cfg.seed + rank,
    )
    batches = itertools.chain.from_iterable(itertools.repeat(loader))

    for step in range(1, cfg.steps + 1):
        frames = next(batches)["frames"].to(device, non_blocking=True)
        tokens, actions = encode_batch(tokenizer, lam, frames, common.unwrap(model).null_action)
        out = common.unwrap(model).training_loss(tokens, actions, forward=model)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()

        if rank == 0 and step % cfg.log_every == 0:
            metrics = {"loss": out["loss"].item(), "masked_acc": out["masked_acc"].item()}
            print(f"step {step}  " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
            if run:
                run.log(metrics, step=step)
        if rank == 0 and step % cfg.rollout_every == 0:
            out_dir.mkdir(parents=True, exist_ok=True)
            save_rollout_strip(
                common.unwrap(model), tokenizer, tokens, actions,
                cfg.rollout_horizon, cfg.decode_steps, out_dir / f"rollout_{step:07d}.png",
            )
        if rank == 0 and (step % cfg.ckpt_every == 0 or step == cfg.steps):
            common.save_checkpoint(out_dir / "dynamics.pt", step, cfg, model=model)

    if run:
        run.finish()


if __name__ == "__main__":
    main()
