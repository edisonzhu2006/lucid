import numpy as np

from eval.fvd import frechet_distance, gaussian_stats
from eval.rollout_drift import drift_curves


def test_frechet_identical_is_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 16))
    mu, sigma = gaussian_stats(x)
    assert abs(frechet_distance(mu, sigma, mu, sigma)) < 1e-6


def test_frechet_mean_shift():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20000, 8))
    y = x + 2.0  # same covariance, shifted mean: FD = ||dmu||^2 = 8*4
    d = frechet_distance(*gaussian_stats(x), *gaussian_stats(y))
    assert abs(d - 32.0) < 1.0


def test_frechet_detects_scale():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(20000, 8))
    y = rng.normal(scale=2.0, size=(20000, 8))
    assert frechet_distance(*gaussian_stats(x), *gaussian_stats(y)) > 1.0


def test_drift_curves():
    rng = np.random.default_rng(0)
    gt = rng.integers(0, 255, (4, 6, 16, 16, 3), dtype=np.uint8)
    # prediction degrades with horizon: step 0 exact, later steps noisier
    pred = gt.copy()
    for t in range(1, 6):
        noise = rng.normal(0, 10 * t, pred[:, t].shape)
        pred[:, t] = np.clip(pred[:, t] + noise, 0, 255).astype(np.uint8)
    out = drift_curves(gt, pred)
    assert len(out["mse"]) == 6 and len(out["psnr"]) == 6
    assert out["mse"][0] < 1e-9
    assert out["mse"][-1] > out["mse"][1]  # drift grows with horizon
    assert np.isfinite(out["psnr"]).all()
