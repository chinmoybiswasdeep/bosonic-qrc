"""Complete quantum-channel construction and physically labelled diagnostics."""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs


def compose_kraus(*stages: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
    """Compose Kraus stages in application order without dropping noise."""
    current = (np.eye(stages[0][0].shape[1], dtype=complex),)
    for stage in stages:
        current = tuple(right @ left for left in current for right in stage)
    return current


def apply_kraus_raw(density: np.ndarray, kraus: tuple[np.ndarray, ...]):
    return sum(
        (operator @ density @ operator.conj().T for operator in kraus), np.zeros_like(density)
    )


def channel_superoperator(kraus: tuple[np.ndarray, ...]):
    dimension = kraus[0].shape[1]
    result = np.zeros((dimension**2, dimension**2), complex)
    for operator in kraus:
        result += np.kron(operator.conj(), operator)
    return result


def traceless_basis(dimension: int):
    rows = []
    for left in range(dimension):
        for right in range(left + 1, dimension):
            symmetric = np.zeros((dimension, dimension), complex)
            symmetric[left, right] = symmetric[right, left] = 1 / np.sqrt(2)
            rows.append(symmetric.reshape(-1, order="F"))
            antisymmetric = np.zeros((dimension, dimension), complex)
            antisymmetric[left, right] = -1j / np.sqrt(2)
            antisymmetric[right, left] = 1j / np.sqrt(2)
            rows.append(antisymmetric.reshape(-1, order="F"))
    for index in range(1, dimension):
        diagonal = np.zeros((dimension, dimension), complex)
        diagonal[:index, :index] += np.eye(index) / np.sqrt(index * (index + 1))
        diagonal[index, index] = -index / np.sqrt(index * (index + 1))
        rows.append(diagonal.reshape(-1, order="F"))
    return np.column_stack(rows)


def complete_channel_diagnostics(
    kraus: tuple[np.ndarray, ...],
    *,
    tp_tolerance: float = 1e-9,
    cp_tolerance: float = 1e-10,
):
    """Diagnose the complete input-dependent map before any normalization."""
    dimension = kraus[0].shape[1]
    completeness = sum(
        (operator.conj().T @ operator for operator in kraus),
        np.zeros((dimension, dimension), complex),
    )
    tp_residual = float(np.linalg.norm(completeness - np.eye(dimension)))
    vectors = np.column_stack([operator.reshape(-1, order="F") for operator in kraus])
    gram = vectors.conj().T @ vectors
    minimum_choi_eigenvalue = float(min(0, np.linalg.eigvalsh(gram).min()))

    def forward(vector):
        matrix = vector.reshape((dimension, dimension), order="F")
        return apply_kraus_raw(matrix, kraus).reshape(-1, order="F")

    def adjoint(vector):
        matrix = vector.reshape((dimension, dimension), order="F")
        result = sum(
            (operator.conj().T @ matrix @ operator for operator in kraus),
            np.zeros_like(matrix),
        )
        return result.reshape(-1, order="F")

    dense = dimension <= 20
    if dense:
        superoperator = channel_superoperator(kraus)
        eigenvalues, eigenvectors = np.linalg.eig(superoperator)
        normality_scale = max(np.linalg.norm(superoperator) ** 2, 1e-30)
        non_normality = float(
            np.linalg.norm(
                superoperator.conj().T @ superoperator - superoperator @ superoperator.conj().T
            )
            / normality_scale
        )
    else:
        operator = LinearOperator(
            (dimension**2, dimension**2), matvec=forward, rmatvec=adjoint, dtype=complex
        )
        eigenvalues, eigenvectors = eigs(operator, k=4, which="LM", tol=1e-9)
        probe_rng = np.random.default_rng(719)
        differences = []
        scales = []
        for _ in range(6):
            probe = probe_rng.normal(size=dimension**2) + 1j * probe_rng.normal(size=dimension**2)
            differences.append(np.linalg.norm(adjoint(forward(probe)) - forward(adjoint(probe))))
            scales.append(np.linalg.norm(forward(probe)) ** 2)
        non_normality = float(np.mean(differences) / max(np.mean(scales), 1e-30))
    order = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    fixed_index = int(np.argmin(np.abs(eigenvalues - 1)))
    fixed = eigenvectors[:, fixed_index].reshape((dimension, dimension), order="F")
    fixed = (fixed + fixed.conj().T) / 2
    fixed_trace = np.trace(fixed)
    if abs(fixed_trace) > 1e-14:
        fixed = fixed / fixed_trace
    fixed_residual = float(np.linalg.norm(apply_kraus_raw(fixed, kraus) - fixed))
    nonstationary = np.delete(eigenvalues, fixed_index)
    subleading = float(np.max(np.abs(nonstationary))) if len(nonstationary) else 0.0
    if dense:
        q = traceless_basis(dimension)
        restricted = q.conj().T @ superoperator @ q
        contraction = float(np.linalg.svd(restricted, compute_uv=False)[0])
        contraction_method = "exact projected singular value"
    else:
        probe_rng = np.random.default_rng(811)
        vector = probe_rng.normal(size=dimension**2) + 1j * probe_rng.normal(size=dimension**2)
        identity_vector = np.eye(dimension).reshape(-1, order="F")
        vector -= identity_vector * np.vdot(identity_vector, vector) / dimension
        vector /= np.linalg.norm(vector)
        contraction = 0.0
        for _ in range(30):
            image = forward(vector)
            image -= identity_vector * np.vdot(identity_vector, image) / dimension
            contraction = float(np.linalg.norm(image))
            back = adjoint(image)
            back -= identity_vector * np.vdot(identity_vector, back) / dimension
            vector = back / max(np.linalg.norm(back), 1e-30)
        contraction_method = "30-step power estimate on traceless subspace"
    passed = tp_residual <= tp_tolerance and minimum_choi_eigenvalue >= -cp_tolerance
    return {
        "dimension": dimension,
        "kraus_count": len(kraus),
        "tp_residual": tp_residual,
        "minimum_choi_eigenvalue": minimum_choi_eigenvalue,
        "cp_residual": max(0.0, -minimum_choi_eigenvalue),
        "fixed_point_residual": fixed_residual,
        "fixed_point_minimum_eigenvalue": float(np.linalg.eigvalsh(fixed).min()),
        "subleading_eigenvalue_modulus": subleading,
        "eigenvalue_mixing_gap": float(1 - subleading),
        "traceless_hilbert_schmidt_contraction": contraction,
        "non_normality": non_normality,
        "dense_superoperator": dense,
        "contraction_method": contraction_method,
        "validation_passed": passed,
        "tp_tolerance": tp_tolerance,
        "cp_tolerance": cp_tolerance,
        "singular_value_note": "largest singular value after projection to the traceless Hilbert-Schmidt subspace; not called a channel gap",
    }


def staged_channel_diagnostics(
    stages: tuple[tuple[np.ndarray, ...], ...],
    *,
    tp_tolerance: float = 1e-9,
    cp_tolerance: float = 1e-10,
):
    """Diagnose a complete channel without expanding its Cartesian Kraus product.

    Each stage is an explicit Kraus map.  Complete positivity is certified from
    the stage Choi Gram matrices, and composition preserves that certificate.
    """
    dimension = stages[0][0].shape[1]
    if dimension <= 20:
        return complete_channel_diagnostics(
            compose_kraus(*stages),
            tp_tolerance=tp_tolerance,
            cp_tolerance=cp_tolerance,
        )

    def forward_matrix(matrix):
        for stage in stages:
            matrix = apply_kraus_raw(matrix, stage)
        return matrix

    def adjoint_matrix(matrix):
        for stage in reversed(stages):
            matrix = sum(
                (operator.conj().T @ matrix @ operator for operator in stage),
                np.zeros_like(matrix),
            )
        return matrix

    def forward(vector):
        matrix = vector.reshape((dimension, dimension), order="F")
        return forward_matrix(matrix).reshape(-1, order="F")

    def adjoint(vector):
        matrix = vector.reshape((dimension, dimension), order="F")
        return adjoint_matrix(matrix).reshape(-1, order="F")

    identity = np.eye(dimension, dtype=complex)
    tp_residual = float(np.linalg.norm(adjoint_matrix(identity) - identity))
    stage_gram_minima = []
    for stage in stages:
        vectors = np.column_stack([operator.reshape(-1, order="F") for operator in stage])
        stage_gram_minima.append(float(np.linalg.eigvalsh(vectors.conj().T @ vectors).min()))
    minimum_choi_eigenvalue = float(min(0.0, *stage_gram_minima))

    identity_vector = identity.reshape(-1, order="F")

    def project(vector):
        return vector - identity_vector * np.vdot(identity_vector, vector) / dimension

    full_operator = LinearOperator(
        (dimension**2, dimension**2), matvec=forward, rmatvec=adjoint, dtype=complex
    )
    _, fixed_vector = eigs(
        full_operator,
        k=1,
        which="LM",
        v0=identity_vector / np.sqrt(dimension),
        tol=1e-7,
        maxiter=300,
        ncv=12,
    )
    fixed = fixed_vector[:, 0].reshape((dimension, dimension), order="F")
    fixed = (fixed + fixed.conj().T) / 2
    fixed /= np.trace(fixed)
    for _ in range(200):
        if np.linalg.norm(forward_matrix(fixed) - fixed) < 1e-11:
            break
        fixed = forward_matrix(fixed)
    fixed_residual = float(np.linalg.norm(forward_matrix(fixed) - fixed))

    spectral_rng = np.random.default_rng(613)
    initial = project(
        spectral_rng.normal(size=dimension**2) + 1j * spectral_rng.normal(size=dimension**2)
    )
    initial /= np.linalg.norm(initial)
    krylov_dimension = min(320 if dimension <= 40 else 192, dimension**2 - 1)
    vectors = np.zeros((dimension**2, krylov_dimension + 1), complex)
    hessenberg = np.zeros((krylov_dimension + 1, krylov_dimension), complex)
    vectors[:, 0] = initial
    arnoldi_steps = 0
    for column in range(krylov_dimension):
        candidate = project(forward(vectors[:, column]))
        for _ in range(2):
            overlaps = vectors[:, : column + 1].conj().T @ candidate
            hessenberg[: column + 1, column] += overlaps
            candidate -= vectors[:, : column + 1] @ overlaps
        norm = np.linalg.norm(candidate)
        hessenberg[column + 1, column] = norm
        arnoldi_steps = column + 1
        if norm < 1e-13:
            break
        vectors[:, column + 1] = candidate / norm
    ritz_values, ritz_vectors = np.linalg.eig(hessenberg[:arnoldi_steps, :arnoldi_steps])
    dominant = int(np.argmax(np.abs(ritz_values)))
    subleading = float(abs(ritz_values[dominant]))
    subleading_residual = float(
        abs(hessenberg[arnoldi_steps, arnoldi_steps - 1] * ritz_vectors[-1, dominant])
    )

    probe_rng = np.random.default_rng(719)
    differences = []
    scales = []
    for _ in range(3):
        probe = probe_rng.normal(size=dimension**2) + 1j * probe_rng.normal(size=dimension**2)
        differences.append(np.linalg.norm(adjoint(forward(probe)) - forward(adjoint(probe))))
        scales.append(np.linalg.norm(forward(probe)) ** 2)
    non_normality = float(np.mean(differences) / max(np.mean(scales), 1e-30))

    vector = project(probe_rng.normal(size=dimension**2) + 1j * probe_rng.normal(size=dimension**2))
    vector /= np.linalg.norm(vector)
    contraction = 0.0
    for _ in range(12):
        image = project(forward(vector))
        contraction = float(np.linalg.norm(image))
        back = project(adjoint(image))
        vector = back / max(np.linalg.norm(back), 1e-30)

    passed = tp_residual <= tp_tolerance and minimum_choi_eigenvalue >= -cp_tolerance
    return {
        "dimension": dimension,
        "kraus_count": int(np.prod([len(stage) for stage in stages])),
        "tp_residual": tp_residual,
        "minimum_choi_eigenvalue": minimum_choi_eigenvalue,
        "cp_residual": max(0.0, -minimum_choi_eigenvalue),
        "cp_certificate": "positive stage Choi Gram matrices; CP is closed under composition",
        "stage_choi_gram_minima": stage_gram_minima,
        "fixed_point_residual": fixed_residual,
        "fixed_point_minimum_eigenvalue": float(np.linalg.eigvalsh(fixed).min()),
        "subleading_eigenvalue_modulus": subleading,
        "subleading_eigenvalue_ritz_residual": subleading_residual,
        "subleading_eigenvalue_method": f"{arnoldi_steps}-step Arnoldi on traceless subspace",
        "eigenvalue_mixing_gap": float(1 - subleading),
        "traceless_hilbert_schmidt_contraction": contraction,
        "non_normality": non_normality,
        "dense_superoperator": False,
        "contraction_method": "12-step power estimate on traceless subspace",
        "validation_passed": passed,
        "tp_tolerance": tp_tolerance,
        "cp_tolerance": cp_tolerance,
        "singular_value_note": "largest singular value after projection to the traceless Hilbert-Schmidt subspace; not called a channel gap",
    }


def echo_state_curves(factory, input_sequences, initial_states):
    """Trace-distance contraction for paired physical states and common inputs."""
    from .core import trace_distance

    curves = []
    for inputs in input_sequences:
        models = [factory() for _ in initial_states]
        for model, state in zip(models, initial_states):
            model.state = np.asarray(state, complex).copy()
        curve = []
        for value in inputs:
            for model in models:
                model.step(float(value))
            curve.append(trace_distance(models[0].state, models[1].state))
        curves.append(curve)
    values = np.asarray(curves)
    return {
        "curves": values.tolist(),
        "mean": np.mean(values, axis=0).tolist(),
        "ci95": np.quantile(values, [0.025, 0.975], axis=0).T.tolist(),
        "input_realizations": len(values),
        "common_random_numbers": True,
    }
