"""Stage 1 entrypoint: train the FSQ tokenizer on single frames.

Gate: reconstructions are sharp — inspect the grids written to out_dir.

Run:  python -m train.train_tokenizer
      python -m train.train_tokenizer steps=200 batch_size=16 model.base_ch=16
      torchrun --nproc-per-node=8 -m train.train_tokenizer   # cluster
"""

import itertools

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from data.datamodule import make_dataloader
from models.tokenizer import Tokenizer, float_to_img, img_to_float, psnr
from train import common


def save_recon_grid(x, recon, path, n=8):
    """Side-by-side original|reconstruction PNG."""
    import imageio.v3 as iio

    orig = float_to_img(x[:n]).cpu().numpy()
    rec = float_to_img(recon[:n]).cpu().numpy()
    rows = [np.concatenate(list(imgs), axis=1) for imgs in (orig, rec)]
    iio.imwrite(path, np.concatenate(rows, axis=0))


@hydra.main(version_base=None, config_path="../configs", config_name="tokenizer")
def main(cfg: DictConfig) -> None:
    device, rank, world = common.setup(cfg.seed)
    out_dir = common.resolve_dir(cfg.out_dir)
    run = common.maybe_wandb(cfg, rank)

    model = Tokenizer(tuple(cfg.model.levels), cfg.model.base_ch).to(device)
    model = common.wrap_ddp(model, device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    loader = make_dataloader(
        common.resolve_dir(cfg.data_dir),
        batch_size=cfg.batch_size,
        seq_len=1,
        num_workers=cfg.num_workers,
        seed=cfg.seed + rank,
    )
    batches = itertools.chain.from_iterable(itertools.repeat(loader))

    for step in range(1, cfg.steps + 1):
        frames = next(batches)["frames"].squeeze(1)
        x = img_to_float(frames).to(device, non_blocking=True)
        recon = model(x)
        loss = torch.nn.functional.mse_loss(recon, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if rank == 0 and step % cfg.log_every == 0:
            metrics = {"loss": loss.item(), "psnr": psnr(recon.detach(), x)}
            print(f"step {step}  " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
            if run:
                run.log(metrics, step=step)
        if rank == 0 and step % cfg.sample_every == 0:
            out_dir.mkdir(parents=True, exist_ok=True)
            save_recon_grid(x, recon.detach(), out_dir / f"recon_{step:07d}.png")
        if rank == 0 and (step % cfg.ckpt_every == 0 or step == cfg.steps):
            common.save_checkpoint(out_dir / "tokenizer.pt", step, cfg, model=model)

    if run:
        run.finish()


if __name__ == "__main__":
    main()
