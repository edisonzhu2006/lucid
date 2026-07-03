"""Stage 2 entrypoint: train the latent action model on frame pairs.

Gate: fix a frame, sweep the codes, and every code should produce one
consistent, distinguishable movement — inspect sweep_*.png and watch
code_entropy (collapse to one code is the failure mode).

Run:  python -m train.train_latent_action
"""

import itertools

import hydra
import torch
from omegaconf import DictConfig

from data.datamodule import make_dataloader
from eval.controllability import controllability_probe, save_sweep_figure
from models.latent_action import LatentActionModel
from models.tokenizer import img_to_float
from train import common


@hydra.main(version_base=None, config_path="../configs", config_name="latent_action")
def main(cfg: DictConfig) -> None:
    device, rank, world = common.setup(cfg.seed)
    out_dir = common.resolve_dir(cfg.out_dir)
    run = common.maybe_wandb(cfg, rank)

    model = LatentActionModel(cfg.model.num_codes, cfg.model.code_dim, cfg.model.base_ch).to(device)
    model = common.wrap_ddp(model, device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    loader = make_dataloader(
        common.resolve_dir(cfg.data_dir),
        batch_size=cfg.batch_size,
        seq_len=2,
        num_workers=cfg.num_workers,
        seed=cfg.seed + rank,
    )
    batches = itertools.chain.from_iterable(itertools.repeat(loader))

    for step in range(1, cfg.steps + 1):
        frames = img_to_float(next(batches)["frames"]).to(device, non_blocking=True)
        x_t, x_tp1 = frames[:, 0], frames[:, 1]
        pred, idx, vq_loss = model(x_t, x_tp1)
        recon_loss = torch.nn.functional.mse_loss(pred, x_tp1)
        loss = recon_loss + vq_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        lam = common.unwrap(model)
        if rank == 0 and step % cfg.log_every == 0:
            metrics = {
                "recon_loss": recon_loss.item(),
                "vq_loss": vq_loss.item(),
                "code_entropy": lam.vq.usage_entropy(),
                "codes_in_batch": idx.unique().numel(),
            }
            print(f"step {step}  " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
            if run:
                run.log(metrics, step=step)
        if rank == 0 and step % cfg.sweep_every == 0:
            out_dir.mkdir(parents=True, exist_ok=True)
            probe_x = x_t[:6].detach()
            save_sweep_figure(lam, probe_x, out_dir / f"sweep_{step:07d}.png")
            probe = controllability_probe(lam, probe_x)
            print(f"step {step}  " + "  ".join(f"{k}={v:.3f}" for k, v in probe.items()))
            if run:
                run.log(probe, step=step)
        if rank == 0 and (step % cfg.ckpt_every == 0 or step == cfg.steps):
            common.save_checkpoint(out_dir / "latent_action.pt", step, cfg, model=model)

    if run:
        run.finish()


if __name__ == "__main__":
    main()
