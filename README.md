# lucid

**A playable, learned dream — and an agent that grows up inside it.**

A lucid dream is a dream you can control. `lucid` learns one from raw CoinRun
video: a Genie-style world model trained with no action labels, whose latent
action codes make the dream playable with a keyboard. A DreamerV3-style
actor-critic is then trained *entirely inside the dream* — it never touches the
real game — and is finally deployed in real CoinRun to measure how well a
policy learned in imagination transfers to reality.

The headline result this project builds to: **a real-game score from an agent
that never trained on the real game.**

The pipeline is built and debugged on CoinRun, then re-run unchanged on a
fresh Stable-Retro title no world-model paper has used for the showcase
result.

See [PLAN.md](PLAN.md) for the full implementation plan: build stages and
gates, metrics, repo structure, timeline, and de-risking.

---

## Quick start (local CPU, tiny smoke test)

```bash
pip install -r requirements.txt
python -m pytest tests/ -q           # verify all 30 tests pass
```

## Cluster (multi-node GPU pipeline)

All stages are ready-to-go Slurm sbatch scripts under `scripts/`:

```bash
sbatch scripts/launch_collect.sh              # Stage 0: ~12 hrs, 1 GPU
sbatch scripts/launch_tokenizer.sh            # Stage 1: ~4 hrs, 8 GPUs
sbatch scripts/launch_latent_action.sh        # Stage 2: ~2 hrs, 8 GPUs
sbatch scripts/launch_dynamics.sh             # Stage 3: ~6 hrs, 16 GPUs (2 nodes)
sbatch scripts/launch_agent.sh                # Stage 5: ~4 hrs, 8 GPUs
sbatch scripts/launch_diffusion.sh            # Stage 6: ~6 hrs, 16 GPUs (2 nodes)
```

Each script is self-contained: edit the `#SBATCH` directives to match your
cluster's partition names, GPU count, wall time. Logs go to `logs/`.

## Play the world model

Once Stage 3 (dynamics) trains, play it interactively or headless:

```bash
# Interactive (macOS w/ pygame, or cluster w/ X11 forwarding):
python -m interactive.play \
  --tokenizer results/tokenizer/tokenizer.pt \
  --dynamics results/dynamics/dynamics.pt \
  --data-dir datasets/coinrun/train \
  --key-map "left=3,right=1,up=5"                    # adjust after seeing sweep figure
# Press 1..K to inject latent codes; ESC quits; --record play.gif to save.

# Headless (record a GIF of the model dreaming):
python -m interactive.play \
  --tokenizer results/tokenizer/tokenizer.pt \
  --dynamics results/dynamics/dynamics.pt \
  --data-dir datasets/coinrun/train \
  --headless --steps 64 --record dream.gif
```

The sweep figure (see Stage 2 logs) shows which latent code does what; adjust
`--key-map` to match your codes.

## Evaluate in real CoinRun

(Requires `procgen` on x86_64 or cluster compute node; not on Apple Silicon.)

```bash
# Build the action mapping from held-out labeled pairs (mapping purity is the ceiling):
python -m eval.real_game_eval \
  --lam results/latent_action/latent_action.pt \
  --data-dir datasets/coinrun/heldout

# Run the imagination-trained agent in real CoinRun:
python -m eval.real_game_eval \
  --lam results/latent_action/latent_action.pt \
  --tokenizer results/tokenizer/tokenizer.pt \
  --agent results/agent/agent.pt \
  --data-dir datasets/coinrun/heldout \
  --episodes 100
```

The result is the headline: **mean CoinRun score for a policy trained purely
inside the dream, tested in the real game.**

## Ablation: diffusion dynamics vs transformer

Train DIAMOND-style diffusion dynamics and compare on FVD, long-horizon drift,
and downstream agent score:

```bash
sbatch scripts/launch_diffusion.sh

# Compare rollout drift (coherence over 32 frames):
python -m eval.rollout_drift \
  --tokenizer results/tokenizer/tokenizer.pt \
  --lam results/latent_action/latent_action.pt \
  --dynamics results/dynamics/dynamics.pt \
  --data-dir datasets/coinrain/train \
  --horizon 32 --out-dir results/drift_transformer

python -m eval.rollout_drift \
  --tokenizer results/tokenizer/tokenizer.pt \
  --lam results/latent_action/latent_action.pt \
  --dynamics results/dynamics_diffusion/dynamics_diffusion.pt \
  --data-dir datasets/coinrun/train \
  --horizon 32 --out-dir results/drift_diffusion
# Plot: diffusion typically has higher long-horizon FVD but same agent
# transfer (the bottleneck is the latent action bottleneck, not dynamics quality).
```

---

## Repo structure

See [PLAN.md](PLAN.md) for the architecture overview. Key entry points:

- **Data:** `data/collect.py` (rollout collection), `data/datamodule.py` (streaming)
- **Models:** `models/tokenizer.py` (FSQ), `models/latent_action.py` (Genie), `models/dynamics_transformer.py` (MaskGIT), `models/heads.py` (DreamerV3)
- **Training:** `train/train_*.py` per stage
- **Interactive:** `interactive/play.py` (keyboard + headless)
- **Behavior:** `behavior/imagination.py` (actor-critic in imagination)
- **Eval:** `eval/` (real-game eval, FVD, drift, controllability probe)
- **Tests:** `tests/` (30 unit & integration tests, CPU-fast)

## References

Ha & Schmidhuber (2018) *World Models* · Hafner et al. (2023) *DreamerV3* ·
Bruce et al. (2024) *Genie* · Alonso et al. (2024) *DIAMOND* · Mentzer et al.
(2023) *FSQ* · Chang et al. (2022) *MaskGIT* · Micheli et al. (2023) *IRIS*
