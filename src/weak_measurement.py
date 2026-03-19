from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from scipy.linalg import expm

from .data import StateParams, rho_ab


I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
Sdg = np.array([[1, 0], [0, -1j]], dtype=complex)

BITSTRINGS_3Q = [f"{i:03b}" for i in range(8)]


def _ry(theta: float) -> np.ndarray:
    ct = np.cos(theta / 2)
    st = np.sin(theta / 2)
    return np.array([[ct, -st], [st, ct]], dtype=complex)


@lru_cache(maxsize=256)
def _weak_unitary_cached(g_rounded: float) -> np.ndarray:
    return expm(-1j * float(g_rounded) * np.kron(X, Y) / 2.0)


def weak_unitary_bp(g: float) -> np.ndarray:
    return _weak_unitary_cached(round(float(g), 10))


def pointer_basis_rotation(basis: str) -> np.ndarray:
    b = basis.upper()
    if b == "Z":
        return I2
    if b == "X":
        return H
    if b == "Y":
        return H @ Sdg  # apply Sdg then H
    raise ValueError(f"Unknown pointer basis: {basis}")


def simulate_outcome_probabilities(
    rho_ab_in: np.ndarray,
    g: float,
    pointer_basis: str,
    b_basis_angle: float = 0.0,
) -> dict[str, float]:
    """
    Exact 3-qubit (A,B,P) probabilities for terminal Z measurements.
    Bit order is A,B,P.
    """
    ket0 = np.array([1.0, 0.0], dtype=complex)
    rho_p0 = np.outer(ket0, ket0.conj())

    rho0 = np.kron(rho_ab_in, rho_p0)

    # U_weak on B,P; identity on A.
    Uweak = np.kron(I2, weak_unitary_bp(g))

    # Local compression postselection implementation: apply H on A before terminal measurement.
    UH_A = np.kron(H, np.eye(4, dtype=complex))

    # Optional B basis rotation and pointer basis readout rotation.
    Ub = np.kron(np.kron(I2, _ry(b_basis_angle)), I2)
    Up = np.kron(np.kron(I2, I2), pointer_basis_rotation(pointer_basis))

    U = Up @ Ub @ UH_A @ Uweak
    rho_fin = U @ rho0 @ U.conj().T

    probs_vec = np.real(np.diag(rho_fin))
    probs_vec = np.clip(probs_vec, 0.0, None)
    z = np.sum(probs_vec)
    if z <= 0:
        probs_vec = np.ones(8) / 8.0
    else:
        probs_vec = probs_vec / z
    return {BITSTRINGS_3Q[i]: float(probs_vec[i]) for i in range(8)}


def outcome_prob_vector(
    rho_ab_in: np.ndarray,
    g: float,
    pointer_basis: str,
    b_basis_angle: float = 0.0,
) -> np.ndarray:
    probs = simulate_outcome_probabilities(rho_ab_in, g, pointer_basis, b_basis_angle)
    return np.array([probs[k] for k in BITSTRINGS_3Q], dtype=float)


def sample_counts_from_probs(
    probs: dict[str, float], shots: int, rng: np.random.Generator
) -> dict[str, int]:
    vec = np.array([probs[k] for k in BITSTRINGS_3Q], dtype=float)
    vec = np.clip(vec, 0.0, None)
    vec = vec / np.sum(vec)
    draw = rng.multinomial(shots, vec)
    return {BITSTRINGS_3Q[i]: int(draw[i]) for i in range(8)}


def simulate_counts_for_state(
    params: StateParams,
    g: float,
    pointer_basis: str,
    shots: int,
    rng: np.random.Generator,
    noise: str = "ideal",
    b_basis_angle: float = 0.0,
    depolarizing_strength: float = 0.03,
) -> tuple[dict[str, int], dict[str, float]]:
    rho = rho_ab(params)
    probs = simulate_outcome_probabilities(
        rho,
        g=g,
        pointer_basis=pointer_basis,
        b_basis_angle=b_basis_angle,
    )

    mode = noise.lower()
    if mode in {"depolarizing", "from_backend"}:
        lam = float(np.clip(depolarizing_strength, 0.0, 1.0))
        u = 1.0 / 8.0
        probs = {k: (1.0 - lam) * v + lam * u for k, v in probs.items()}

    counts = sample_counts_from_probs(probs, shots=shots, rng=rng)
    return counts, probs


def conditional_state_after_A_plus(rho_ab_in: np.ndarray) -> tuple[np.ndarray, float]:
    plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
    gamma = np.outer(plus, plus.conj())
    M = np.kron(gamma, I2)
    num = M @ rho_ab_in @ M
    p_plus = float(np.real(np.trace(num)))
    if p_plus <= 1e-15:
        return np.eye(2, dtype=complex) / 2.0, 0.0

    resh = num.reshape(2, 2, 2, 2)
    rho_b = np.zeros((2, 2), dtype=complex)
    for a in range(2):
        rho_b += resh[a, :, a, :]
    rho_b = rho_b / p_plus
    rho_b = 0.5 * (rho_b + rho_b.conj().T)
    rho_b = rho_b / np.trace(rho_b)
    return rho_b, p_plus


def weak_values_family(params: StateParams) -> tuple[float, float]:
    p = float(np.clip(params.p, 0.0, 1.0))
    th = float(np.clip(params.theta, 1e-9, np.pi / 4 - 1e-9))
    w0 = float(p * np.tan(th))
    w1 = float(p / np.tan(th))
    return w0, w1


def _cond_pointer_mean(counts: dict[str, int], b_val: int) -> tuple[float, int]:
    num = 0
    den = 0
    for bitstring, ct in counts.items():
        a = int(bitstring[0])
        b = int(bitstring[1])
        p = int(bitstring[2])
        if a == 0 and b == b_val:
            den += int(ct)
            num += int(ct) * (1 if p == 0 else -1)
    if den == 0:
        return 0.0, 0
    return float(num / den), den


def feature_block_from_counts_by_basis(counts_by_basis: dict[str, dict[str, int]]) -> dict[str, float]:
    if not counts_by_basis:
        raise ValueError("counts_by_basis cannot be empty")

    ref_basis = "X" if "X" in counts_by_basis else next(iter(counts_by_basis.keys()))
    ref = counts_by_basis[ref_basis]
    total = max(1, sum(ref.values()))

    p_a0b0 = sum(ct for bs, ct in ref.items() if bs[0] == "0" and bs[1] == "0") / total
    p_a0b1 = sum(ct for bs, ct in ref.items() if bs[0] == "0" and bs[1] == "1") / total

    out: dict[str, float] = {
        "p_a0b0": float(p_a0b0),
        "p_a0b1": float(p_a0b1),
    }
    for basis in ["X", "Y", "Z"]:
        c = counts_by_basis.get(basis)
        if c is None:
            out[f"m0_{basis}"] = 0.0
            out[f"m1_{basis}"] = 0.0
            out[f"valid0_{basis}"] = 0.0
            out[f"valid1_{basis}"] = 0.0
            continue
        m0, d0 = _cond_pointer_mean(c, b_val=0)
        m1, d1 = _cond_pointer_mean(c, b_val=1)
        out[f"m0_{basis}"] = float(m0)
        out[f"m1_{basis}"] = float(m1)
        out[f"valid0_{basis}"] = 1.0 if d0 > 0 else 0.0
        out[f"valid1_{basis}"] = 1.0 if d1 > 0 else 0.0
    return out


def aggregate_feature_vector(
    measurement_counts: dict[tuple[float, str], dict[str, int]],
    g_grid: list[float],
    include_z: bool = False,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    feats: list[float] = []
    names: list[str] = []
    validity: list[float] = []

    bases = ["X", "Y"] + (["Z"] if include_z else [])

    for g in g_grid:
        block_inputs: dict[str, dict[str, int]] = {}
        for b in bases:
            key = (float(g), b)
            if key in measurement_counts:
                block_inputs[b] = measurement_counts[key]
        block = feature_block_from_counts_by_basis(block_inputs)

        keys = ["p_a0b0", "p_a0b1"]
        for b in bases:
            keys.extend([f"m0_{b}", f"m1_{b}"])
        for k in keys:
            names.append(f"g{g:.3f}_{k}")
            feats.append(float(block[k]))

        validity.extend(
            [
                float(block.get("valid0_X", 1.0)),
                float(block.get("valid1_X", 1.0)),
                float(block.get("valid0_Y", 1.0)),
                float(block.get("valid1_Y", 1.0)),
            ]
        )

    meta = {
        "valid_fraction": float(np.mean(validity)) if validity else 1.0,
    }
    return np.array(feats, dtype=float), names, meta
