import numpy as np
import torch

from interactive.play import WorldModelSession, init_frames_from_shard
from models.dynamics_transformer import DynamicsTransformer
from models.tokenizer import Tokenizer
from tests.conftest import make_frames


def tiny_stack():
    torch.manual_seed(0)
    tokenizer = Tokenizer(base_ch=16).eval()
    dynamics = DynamicsTransformer(
        vocab_size=tokenizer.vocab_size, num_actions=6, tokens_per_frame=64,
        context_frames=3, dim=64, depth=2, heads=4,
    ).eval()
    return tokenizer, dynamics


def test_session_steps_produce_frames():
    tokenizer, dynamics = tiny_stack()
    session = WorldModelSession(
        tokenizer, dynamics, make_frames(5), decode_steps=2, device="cpu"
    )
    for code in [0, 3, 5]:
        frame = session.step(code)
        assert frame.shape == (64, 64, 3) and frame.dtype == np.uint8
    assert len(session.ctx) == dynamics.context_frames
    assert list(session.ctx_actions)[-3:] == [0, 3, 5]


def test_session_bootstraps_short_context():
    tokenizer, dynamics = tiny_stack()
    session = WorldModelSession(  # 1 frame < context_frames=3
        tokenizer, dynamics, make_frames(1), decode_steps=2, device="cpu"
    )
    assert session.step(1).shape == (64, 64, 3)


def test_init_frames_from_shard(shard_dir):
    frames = init_frames_from_shard(shard_dir, context=4)
    assert frames.shape == (4, 64, 64, 3) and frames.dtype == np.uint8
