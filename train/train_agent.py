"""Stage 5 entrypoint: reward head, then actor-critic trained in imagination.

Phase A fits the reward model on held-out TRUE rewards (the only real-env
signal in the agent stack). Phase B trains the agent purely inside the frozen
world model. Evaluate with eval/real_game_eval.py — the gate is beating the
random floor in real CoinRun.

Run:  python -m train.train_agent
"""

import itertools

import hydra
import torch
from omegaconf import DictConfig

from behavior.imagination import ImaginationAgent, ImaginationTrainer, RewardModel
from data.datamodule import make_dataloader
from models.latent_action import LatentActionModel
from models.tokenizer import Tokenizer, img_to_float
from train import common
from train.train_dynamics import encode_batch, load_frozen
from interactive.play import load_models


def infinite(loader):
    return itertools.chain.from_iterable(itertools.repeat(loader))


@hydra.main(version_base=None, config_path="../configs", config_name="agent")
def main(cfg: DictConfig) -> None:
    device, rank, world = common.setup(cfg.seed)
    out_dir = common.resolve_dir(cfg.out_dir)
    run = common.maybe_wandb(cfg, rank)

    tokenizer, dynamics = load_models(
        common.resolve_dir(cfg.tokenizer_ckpt), common.resolve_dir(cfg.dynamics_ckpt), device
    )
    lam = load_frozen(common.resolve_dir(cfg.lam_ckpt), LatentActionModel, device=device)
    tokens_per_frame = dynamics.tokens_per_frame

    # ---- Phase A: reward model on held-out true rewards --------------------
    reward_model = RewardModel(tokenizer.vocab_size, tokens_per_frame).to(device)
    r_opt = torch.optim.AdamW(reward_model.parameters(), lr=cfg.reward.lr)
    r_loader = infinite(make_dataloader(
        common.resolve_dir(cfg.data_dir_heldout), batch_size=cfg.reward.batch_size,
        seq_len=2, num_workers=cfg.num_workers, frames_only=False, seed=cfg.seed,
    ))
    for step in range(1, cfg.reward.steps + 1):
        batch = next(r_loader)
        frames = batch["frames"].to(device, non_blocking=True)
        # rewards[t] arrives with the NEXT observed state: pair frames[:,1] / rewards[:,0]
        rewards = batch["rewards"][:, 0].to(device, non_blocking=True)
        tokens = tokenizer.encode_indices(img_to_float(frames[:, 1])).flatten(1)
        loss = reward_model.loss(tokens, rewards)
        r_opt.zero_grad(set_to_none=True)
        loss.backward()
        r_opt.step()
        if rank == 0 and step % cfg.log_every == 0:
            with torch.no_grad():
                mae = (reward_model.predict(tokens) - rewards).abs().mean().item()
            print(f"[reward] step {step}  loss={loss.item():.4f}  mae={mae:.4f}")
            if run:
                run.log({"reward/loss": loss.item(), "reward/mae": mae}, step=step)

    # ---- Phase B: actor-critic purely in imagination -----------------------
    agent_kwargs = dict(
        vocab_size=tokenizer.vocab_size, num_actions=lam.num_codes,
        tokens_per_frame=tokens_per_frame, dim=cfg.agent.dim,
        state_dim=cfg.agent.state_dim, hidden=cfg.agent.hidden,
    )
    agent = ImaginationAgent(**agent_kwargs).to(device)
    trainer = ImaginationTrainer(
        dynamics, reward_model, agent,
        horizon=cfg.imagination.horizon, gamma=cfg.imagination.gamma,
        lam=cfg.imagination["lambda"], entropy_coef=cfg.imagination.entropy_coef,
        actor_lr=cfg.imagination.actor_lr, critic_lr=cfg.imagination.critic_lr,
        target_decay=cfg.imagination.target_decay,
        decode_steps=cfg.imagination.decode_steps, device=device,
    )
    ctx_len = dynamics.context_frames
    i_loader = infinite(make_dataloader(
        common.resolve_dir(cfg.data_dir), batch_size=cfg.imagination.batch_size,
        seq_len=ctx_len, num_workers=cfg.num_workers, seed=cfg.seed + 1,
    ))
    for step in range(1, cfg.imagination.steps + 1):
        frames = next(i_loader)["frames"].to(device, non_blocking=True)
        ctx, ctx_actions = encode_batch(tokenizer, lam, frames, dynamics.null_action)
        metrics = trainer.update(ctx, ctx_actions)
        if rank == 0 and step % cfg.log_every == 0:
            print(f"[agent] step {step}  " +
                  "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
            if run:
                run.log({f"agent/{k}": v for k, v in metrics.items()}, step=step)
        if rank == 0 and (step % cfg.ckpt_every == 0 or step == cfg.imagination.steps):
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"agent": agent.state_dict(), "reward_model": reward_model.state_dict(),
                 "agent_kwargs": agent_kwargs, "step": step,
                 "cfg": {"agent_kwargs": agent_kwargs}},
                out_dir / "agent.pt",
            )

    if run:
        run.finish()


if __name__ == "__main__":
    main()
