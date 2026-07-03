"""Shared training utilities: device/DDP setup, wandb, checkpoints.

Every stage entrypoint runs single-process on CPU/GPU as-is, and multi-node
under torchrun (DDP activates when RANK is set by the launcher).
"""

import os
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel


def setup(seed: int) -> tuple[torch.device, int, int]:
    """Returns (device, rank, world_size); initializes DDP under torchrun."""
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    if torch.cuda.is_available():
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")
    torch.manual_seed(seed + rank)
    return device, rank, world


def wrap_ddp(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        ids = [device.index] if device.type == "cuda" else None
        return DistributedDataParallel(model, device_ids=ids)
    return model


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "module", model)


def maybe_wandb(cfg: DictConfig, rank: int):
    if rank != 0 or cfg.wandb.mode == "disabled":
        return None
    import wandb

    return wandb.init(
        project=cfg.wandb.project,
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True),
    )


def save_checkpoint(path: Path, step: int, cfg: DictConfig, **modules) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: unwrap(m).state_dict() for name, m in modules.items()}
    torch.save(
        {"step": step, "cfg": OmegaConf.to_container(cfg, resolve=True), **state}, path
    )


def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


def resolve_dir(cfg_path: str) -> Path:
    """Resolve a config path relative to the launch dir (hydra chdirs)."""
    import hydra.utils

    p = Path(cfg_path)
    return p if p.is_absolute() else Path(hydra.utils.get_original_cwd()) / p
