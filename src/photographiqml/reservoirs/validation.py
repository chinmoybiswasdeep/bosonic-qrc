"""Independent references for the declared small scientific subset."""

import numpy as np


def collision_kraus(angle, measurement_angle, *, entangle=True, feedforward=True):
    """K_b=H X^b (cos(a/2) I+(-1)^b exp(-i phi)sin(a/2) Z)/sqrt(2).

    Ancilla is cos(a/2)|0>+sin(a/2)|1>; disable CZ by replacing Z with I.
    Completeness sum K_b^dagger K_b=I establishes trace preservation.
    """
    identity = np.eye(2, dtype=complex)
    z = np.diag([1, -1]) if entangle else identity
    x = np.array([[0, 1], [1, 0]])
    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    return tuple(
        h
        @ (x if bit and feedforward else identity)
        @ (
            np.cos(angle / 2) * identity
            + (-1) ** bit * np.exp(-1j * measurement_angle) * np.sin(angle / 2) * z
        )
        / np.sqrt(2)
        for bit in (0, 1)
    )


def raw_piquasso_collision(joint, weights, rotations):
    """Independent public Piquasso preparation/gates, before ancilla measurement."""
    import piquasso as pq

    state = pq.GaussianState(
        d=len(joint.nodes), connector=pq.NumpyConnector(), config=pq.Config(hbar=2)
    )
    state.xpxp_mean_vector = joint.mean
    state.xpxp_covariance_matrix = 2 * joint.covariance
    with pq.Program() as program:
        for i, angle in enumerate(rotations):
            pq.Q(i) | pq.Phaseshifter(phi=angle)
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                pq.Q(i, weights.shape[0] + j) | pq.ControlledZ(s=float(weights[i, j]))
    result = (
        pq.GaussianSimulator(d=len(joint.nodes), config=pq.Config(hbar=2))
        .execute(program, initial_state=state)
        .state
    )
    return result.xpxp_mean_vector, result.xpxp_covariance_matrix / 2


def gaussian_one_mode_reference(
    mean,
    covariance,
    *,
    value,
    variance_q,
    variance_p,
    coupling,
    rotation,
    angle,
    feedforward,
    eta,
    thermal,
    detector_noise=0.0,
):
    """Independent 4D symplectic/Wigner derivation with affine measurement feedback."""
    c, s = np.cos(rotation), np.sin(rotation)
    transform = np.eye(4)
    transform[:2, :2] = [[c, -s], [s, c]]
    cz = np.eye(4)
    cz[1, 2] = cz[3, 0] = coupling
    transform = cz @ transform
    mu = np.r_[mean, 2 * value, 0.0]
    cov = np.zeros((4, 4))
    cov[:2, :2] = covariance
    cov[2:, 2:] = np.diag([variance_q, variance_p])
    rows = np.eye(4)[:2]
    gain = feedforward * coupling
    rows[0, 2:] += gain * np.array([np.cos(angle), np.sin(angle)])
    output_mean = np.sqrt(eta) * rows @ transform @ mu
    output_cov = eta * rows @ transform @ cov @ transform.T @ rows.T
    output_cov += (1 - eta) * (2 * thermal + 1) * np.eye(2)
    output_cov[0, 0] += eta * gain**2 * detector_noise
    return output_mean, output_cov
