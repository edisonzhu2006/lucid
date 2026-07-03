"""Stage 6 ablation entrypoint: DIAMOND-style diffusion dynamics on pixels.

Same data and latent actions as the transformer; compare on FVD, rollout
drift, and downstream agent score.

Run:  python -m train.train_dynamics_diffusion
"""

import itertools

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from data.datamodule import make_dataloader
from models.dynamics_diffusion import DiffusionDynamics
from models.latent_action import LatentActionModel
from models.tokenizer import float_to_img, img_to_float
from train import common
from train.train_dynamics import load_frozen


@torch.no_grad()
def save_rollout_strip(model, frames_float, actions, horizon, sample_steps, path):
    import imageio.v3 as iio

    c = model.context_frames
    ctx = frames_float[:4, :c]
    action_seq = actions[:4, -1:].expand(-1, horizon).contiguous()
    gen = model.rollout(ctx, action_seq, steps=sample_steps)
    seq = torch.cat([ctx, gen], dim=1).clamp(-1, 1)
    imgs = float_to_img(seq.flatten(0, 1)).cpu().numpy()
    b, t = seq.shape[:2]
    imgs = imgs.reshape(b, t, *imgs.shape[1:])
    rows = [np.concatenate(list(r), axis=1) for r in imgs]
    iio.imwrite(path, np.concatenate(rows, axis=0))


@hydra.main(version_base=None, config_path="../configs", config_name="dynamics_diffusion")
def main(cfg: DictConfig) -> None:
    device, rank, world = common.setup(cfg.seed)
    out_dir = common.resolve_dir(cfg.out_dir)
    run = common.maybe_wandb(cfg, rank)

    lam = load_frozen(common.resolve_dir(cfg.lam_ckpt), LatentActionModel, device=device)
    model = DiffusionDynamics(
        context_frames=cfg.model.context_frames,
        num_actions=lam.num_codes,
        base_ch=cfg.model.base_ch,
        mults=tuple(cfg.model.mults),
        emb_dim=cfg.model.emb_dim,
    ).to(device)
    model = common.wrap_ddp(model, device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / cfg.warmup_steps)
    )

    seq_len = cfg.model.context_frames + 1
    loader = make_dataloader(
        common.resolve_dir(cfg.data_dir), batch_size=cfg.batch_size,
        seq_len=seq_len, num_workers=cfg.num_workers, seed=cfg.seed + rank,
    )
    batches = itertools.chain.from_iterable(itertools.repeat(loader))
    m = common.unwrap(model)

    for step in range(1, cfg.steps + 1):
        frames = img_to_float(next(batches)["frames"]).to(device, non_blocking=True)
        with torch.no_grad():
            b, t = frames.shape[:2]
            acts = lam.encode_action(
                frames[:, :-1].flatten(0, 1), frames[:, 1:].flatten(0, 1)
            ).reshape(b, t - 1)
        context, target, action = frames[:, :-1], frames[:, -1], acts[:, -1]
        # forward through the (possibly DDP-wrapped) model for gradient sync
        sigma = (torch.randn(b, device=device) * m.p_std + m.p_mean).exp()
        noise = torch.randn_like(target) * sigma.reshape(-1, 1, 1, 1)
        denoised = model(target + noise, sigma, context, action)
        weight = ((sigma**2 + m.sigma_data**2) / (sigma * m.sigma_data) ** 2).reshape(-1, 1, 1, 1)
        loss = (weight * (denoised - target) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()

        if rank == 0 and step % cfg.log_every == 0:
            print(f"step {step}  loss={loss.item():.4f}")
            if run:
                run.log({"loss": loss.item()}, step=step)
        if rank == 0 and step % cfg.rollout_every == 0:
            out_dir.mkdir(parents=True, exist_ok=True)
            save_rollout_strip(
                m, frames, acts, cfg.rollout_horizon, cfg.sample_steps,
                out_dir / f"rollout_{step:07d}.png",
            )
        if rank == 0 and (step % cfg.ckpt_every == 0 or step == cfg.steps):
            common.save_checkpoint(out_dir / "dynamics_diffusion.pt", step, cfg, model=model)

    if run:
        run.finish()


if __name__ == "__main__":
    main()
