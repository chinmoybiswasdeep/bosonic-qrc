"""Empirical diagnostics, not proofs of the quantum echo-state property."""

import numpy as np


def feature_diagnostics(features):
    x = np.asarray(features, float)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Need at least two finite feature rows")
    centered = x - x.mean(axis=0)
    s = np.linalg.svd(centered, compute_uv=False)
    spectrum = s * s / (len(x) - 1)
    p = spectrum / spectrum.sum() if spectrum.sum() else spectrum
    p = p[p > 0]
    rank = float(np.exp(-np.sum(p * np.log(p)))) if len(p) else 0.0
    positive = s[s > max(s[0], 1.0) * 1e-12]
    return {
        "effective_rank": rank,
        "covariance_spectrum": spectrum.tolist(),
        "condition_number_nonzero": float(positive[0] / positive[-1]) if len(positive) else None,
        "rank_deficient": len(positive) < x.shape[1],
        "variance_collapsed_columns": int(np.sum(x.var(axis=0) < 1e-12)),
        "maximum_absolute_feature": float(np.max(abs(x))),
    }


def trace_distance(a, b):
    return float(np.sum(abs(np.linalg.eigvalsh(np.asarray(a) - np.asarray(b)))) / 2)


def contraction(factory, inputs, initial_states):
    models = [factory().reset(initial_state=s) for s in initial_states]
    if len(models) < 2:
        raise ValueError("Provide at least two initial states")
    curves, quantum = [], []
    for value in inputs:
        rows = [m.step(value).features for m in models]
        curves.append(
            [float(np.linalg.norm(a - b)) for i, a in enumerate(rows) for b in rows[i + 1 :]]
        )
        if all(isinstance(m.state, np.ndarray) and m.state.ndim == 2 for m in models):
            quantum.append(
                [
                    trace_distance(a.state, b.state)
                    for i, a in enumerate(models)
                    for b in models[i + 1 :]
                ]
            )
    return {
        "metric": "feature Euclidean distance",
        "distances": curves,
        "trace_distances": quantum if quantum else None,
    }


def fading_memory(factory, inputs, *, perturbation=0.01, at=0):
    u = np.asarray(inputs, float)
    if not 0 <= at < len(u) or not np.isfinite(perturbation) or perturbation == 0:
        raise ValueError("Invalid perturbation")
    v = u.copy()
    v[at] += perturbation
    a, b = factory(), factory()
    x, y = a.run_sequence(u).features, b.run_sequence(v).features
    return {
        "at": at,
        "perturbation": perturbation,
        "feature_distance": np.linalg.norm(x - y, axis=1).tolist(),
    }


def bootstrap_summary(values, *, seed=0, draws=2000):
    a = np.asarray(values, float)
    if a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all():
        raise ValueError("Need at least two finite independent seed results")
    rng = np.random.default_rng(seed)
    means = rng.choice(a, (draws, len(a)), replace=True).mean(axis=1)
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)),
        "median": float(np.median(a)),
        "bootstrap95": np.quantile(means, [0.025, 0.975]).tolist(),
        "values": a.tolist(),
        "bootstrap_seed": seed,
    }
