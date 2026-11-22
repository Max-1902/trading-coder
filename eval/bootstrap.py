"""Percentile bootstrap over tasks."""

from __future__ import annotations

import numpy as np

N_BOOT = 10_000
SEED = 0


def ci(values, n_boot: int = N_BOOT, seed: int = SEED, alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean and 95% CI of per-task values (resampling tasks with replacement)."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, (n_boot, v.size))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(v.mean()), float(lo), float(hi)


def paired_diff_ci(a, b, n_boot: int = N_BOOT, seed: int = SEED, alpha: float = 0.05):
    """CI of mean(a - b) where a, b are per-task values for the same tasks."""
    return ci(np.asarray(a, float) - np.asarray(b, float), n_boot, seed, alpha)
