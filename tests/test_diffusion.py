import torch

from models.dynamics_diffusion import DiffusionDynamics


def tiny_model():
    torch.manual_seed(0)
    return DiffusionDynamics(
        context_frames=2, num_actions=4, base_ch=16, mults=(1, 2), emb_dim=32
    )


def make_batch(b=8, size=32):
    torch.manual_seed(1)
    context = torch.rand(b, 2, 3, size, size) * 2 - 1
    action = torch.randint(0, 4, (b,))
    # deterministic toy dynamics: next frame = mean of context + action brightness
    target = context.mean(dim=1) + (action.float().reshape(-1, 1, 1, 1) - 1.5) * 0.1
    return context, target.clamp(-1, 1), action


def test_training_loss_finite():
    model = tiny_model()
    context, target, action = make_batch()
    loss = model.training_loss(context, target, action)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_sample_and_rollout_shapes():
    model = tiny_model().eval()
    context, _, action = make_batch(b=4)
    nxt = model.sample(context, action, steps=2)
    assert nxt.shape == (4, 3, 32, 32)
    assert nxt.min() >= -1 and nxt.max() <= 1
    seq = model.rollout(context, torch.randint(0, 4, (4, 3)), steps=2)
    assert seq.shape == (4, 3, 3, 32, 32)


def test_overfits_toy_dynamics():
    model = tiny_model()
    context, target, action = make_batch(b=16)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    first = None
    for _ in range(80):
        loss = model.training_loss(context, target, action)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first or loss.item()
    assert loss.item() < 0.6 * first, f"no convergence: {first:.3f} -> {loss.item():.3f}"
