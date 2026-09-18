"""Leakage-safe temporal task runner used by the main EOC study."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .tasks import (
    effective_feature_rank,
    ipc_targets,
    mackey_glass,
    narma10,
    nonlinear_channel,
)


@dataclass(frozen=True)
class TaskProtocol:
    train: int = 36
    validation: int = 18
    test: int = 24
    washout: int = 8
    embargo: int = 4
    maximum_delay: int = 3
    maximum_degree: int = 2
    alphas: tuple[float, ...] = (1e-7, 1e-5, 1e-3, 1e-1)

    @property
    def total_length(self):
        return self.train + self.validation + self.test + 3 * self.washout + 2 * self.embargo


def _segments(protocol: TaskProtocol):
    start = 0
    result = {}
    for name, length in (
        ("train", protocol.train),
        ("validation", protocol.validation),
        ("test", protocol.test),
    ):
        stop = start + protocol.washout + length
        result[name] = (start, stop)
        start = stop + (protocol.embargo if name != "test" else 0)
    return result


def _run_split(factory, inputs, target, bounds, washout):
    start, stop = bounds
    model = factory()
    model.reset()
    rows = [model.step(float(value))["features"] for value in inputs[start:stop]]
    return np.asarray(rows)[washout:], np.asarray(target[start:stop])[washout:]


def evaluate_target(factory, inputs, target, protocol: TaskProtocol):
    segments = _segments(protocol)
    split = {
        name: _run_split(factory, inputs, target, bounds, protocol.washout)
        for name, bounds in segments.items()
    }
    train_x, train_y = split["train"]
    validation_x, validation_y = split["validation"]
    test_x, test_y = split["test"]
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    keep = scale > 1e-12
    if not np.any(keep):
        return {
            "test_nmse": 1.0,
            "test_r2": 0.0,
            "selected_alpha": None,
            "effective_rank": 0,
        }
    scale = scale[keep]

    def transform(values):
        return (values[:, keep] - mean[keep]) / scale

    train_x = transform(train_x)
    validation_x = transform(validation_x)
    test_x = transform(test_x)
    target_mean = float(train_y.mean())
    identity = np.eye(train_x.shape[1])
    candidates = []
    for alpha in protocol.alphas:
        coefficients = np.linalg.solve(
            train_x.T @ train_x + alpha * identity,
            train_x.T @ (train_y - target_mean),
        )
        prediction = validation_x @ coefficients + target_mean
        candidates.append((float(np.mean((validation_y - prediction) ** 2)), alpha))
    selected = min(candidates)[1]
    coefficients = np.linalg.solve(
        train_x.T @ train_x + selected * identity,
        train_x.T @ (train_y - target_mean),
    )
    prediction = test_x @ coefficients + target_mean
    variance = float(np.var(test_y))
    nmse = float(np.mean((test_y - prediction) ** 2) / variance) if variance > 0 else 1.0
    return {
        "test_nmse": nmse,
        "test_r2": float(1 - nmse),
        "selected_alpha": selected,
        "effective_rank": effective_feature_rank(train_x),
        "split_bounds": segments,
        "preprocessing_fit": "train only",
        "hyperparameter_selection": "validation only",
        "test_usage": "single untouched evaluation",
    }


def run_task_suite(factory, seed: int, protocol: TaskProtocol, input_domain: str):
    length = protocol.total_length
    rng = np.random.default_rng(seed)
    if input_domain == "dv":
        raw = rng.uniform(0, 1, length)
        centered = 2 * raw - 1
        drive = raw
    else:
        centered = rng.uniform(-1, 1, length)
        drive = centered
    results = {}

    linear_scores = []
    for delay in range(1, protocol.maximum_delay + 1):
        target = np.roll(centered, delay)
        target[:delay] = 0
        score = evaluate_target(factory, drive, target, protocol)
        results[f"linear_memory_delay_{delay}"] = score
        linear_scores.append(max(0.0, score["test_r2"]))
    results["linear_memory_capacity"] = float(sum(linear_scores))

    ipc = ipc_targets(centered, protocol.maximum_delay, protocol.maximum_degree)
    nonlinear_scores = []
    degree_capacity: dict[str, float] = {}
    for label, target in ipc.items():
        score = evaluate_target(factory, drive, target, protocol)
        results[f"ipc:{label}"] = score
        degree = label.split(";")[0]
        value = max(0.0, score["test_r2"])
        degree_capacity[degree] = degree_capacity.get(degree, 0.0) + value
        if degree != "degree=1":
            nonlinear_scores.append(value)
    results["ipc_capacity_by_degree"] = degree_capacity
    results["nonlinear_memory_capacity"] = float(sum(nonlinear_scores))

    bits = rng.integers(0, 2, length)
    parity_drive = bits.astype(float) if input_domain == "dv" else 2 * bits - 1
    parity_target = np.asarray(
        [
            np.sum(bits[max(0, index - protocol.maximum_delay) : index + 1]) % 2
            for index in range(length)
        ],
        float,
    )
    results["delayed_parity"] = evaluate_target(factory, parity_drive, parity_target, protocol)

    narma_input, narma_target = narma10(length, seed + 101)
    narma_drive = 2 * narma_input if input_domain == "dv" else 4 * narma_input - 1
    results["narma10"] = evaluate_target(factory, narma_drive, narma_target, protocol)

    mg_input, mg_target = mackey_glass(length, seed + 211)
    mg_scaled = np.clip((mg_input - 1.0) / 0.6, -1, 1)
    if input_domain == "dv":
        mg_scaled = (mg_scaled + 1) / 2
    results["mackey_glass"] = evaluate_target(factory, mg_scaled, mg_target, protocol)

    channel_input, channel_target = nonlinear_channel(length, seed + 307)
    scale = max(np.max(np.abs(channel_input)), 1e-12)
    channel_drive = np.clip(channel_input / scale, -1, 1)
    if input_domain == "dv":
        channel_drive = (channel_drive + 1) / 2
    results["nonlinear_channel_equalization"] = evaluate_target(
        factory, channel_drive, channel_target, protocol
    )
    return {
        "seed": seed,
        "protocol": asdict(protocol),
        "tasks": results,
        "advertised_tasks_exercised": [
            "linear memory",
            "nonlinear memory",
            "delayed parity",
            "NARMA10",
            "Mackey-Glass",
            "nonlinear channel equalization",
            "polynomial IPC including cross delays",
        ],
    }


def summarize_against_chaos(rows):
    metrics = (
        "linear_memory_capacity",
        "nonlinear_memory_capacity",
        "narma10",
        "mackey_glass",
        "delayed_parity",
        "nonlinear_channel_equalization",
    )
    summary = {}
    coordinates = np.asarray([row["d_eoc"] for row in rows], float)

    def samples(performance, metric):
        if "seeds" in performance:
            return np.asarray([samples(seed, metric)[0] for seed in performance["seeds"]], float)
        task = performance["tasks"].get(metric)
        value = task if isinstance(task, (int, float)) else task["test_r2"]
        return np.asarray([float(value)])

    def effect_size(left, right):
        if len(left) < 2 or len(right) < 2:
            return None
        pooled = np.sqrt((np.var(left, ddof=1) + np.var(right, ddof=1)) / 2)
        return float((np.mean(left) - np.mean(right)) / pooled) if pooled > 0 else None

    for metric in metrics:
        samples_by_point = [samples(row["performance"], metric) for row in rows]
        values = np.asarray([np.mean(value) for value in samples_by_point])
        correlation = (
            float(np.corrcoef(coordinates, values)[0, 1])
            if len(values) > 1 and np.std(values) > 0
            else None
        )
        by_region = {}
        for region in sorted({row["region"] for row in rows}):
            selected = np.concatenate(
                [value for row, value in zip(rows, samples_by_point) if row["region"] == region]
            )
            by_region[region] = {
                "mean": float(np.mean(selected)),
                "count": len(selected),
                "ci95": np.quantile(selected, [0.025, 0.975]).tolist(),
            }
        edge_samples = (
            np.concatenate(
                [
                    value
                    for row, value in zip(rows, samples_by_point)
                    if row["region"] == "crossover"
                ]
            )
            if any(row["region"] == "crossover" for row in rows)
            else np.asarray([])
        )
        regular_samples = (
            np.concatenate(
                [value for row, value in zip(rows, samples_by_point) if row["region"] == "regular"]
            )
            if any(row["region"] == "regular" for row in rows)
            else np.asarray([])
        )
        chaotic_samples = (
            np.concatenate(
                [value for row, value in zip(rows, samples_by_point) if row["region"] == "chaotic"]
            )
            if any(row["region"] == "chaotic" for row in rows)
            else np.asarray([])
        )
        summary[metric] = {
            "pearson_correlation_with_d_eoc": correlation,
            "by_region": by_region,
            "crossover_minus_regular_cohens_d": effect_size(edge_samples, regular_samples),
            "crossover_minus_chaotic_cohens_d": effect_size(edge_samples, chaotic_samples),
            "peak_d_eoc": float(coordinates[int(np.argmax(values))]),
            "effect_size_status": "descriptive; a publication claim also requires size and cutoff robustness",
        }
    return summary
