from __future__ import annotations

from typing import Any

import numpy as np


I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = {"I": I2, "X": X, "Y": Y, "Z": Z}


def concurrence(rho: np.ndarray) -> float:
    sy = Y
    yy = np.kron(sy, sy)
    m = rho @ yy @ rho.conj() @ yy
    vals = np.linalg.eigvals(m)
    vals = np.real_if_close(vals, tol=1e6)
    vals = np.real(vals)
    vals = np.maximum(vals, 0.0)
    roots = np.sort(np.sqrt(vals))[::-1]
    c = float(max(0.0, roots[0] - roots[1] - roots[2] - roots[3]))
    return c


def partial_transpose_2q(rho: np.ndarray, sys: str = "B") -> np.ndarray:
    if rho.shape != (4, 4):
        raise ValueError("rho must be 4x4 for two qubits")
    resh = rho.reshape(2, 2, 2, 2)
    if sys.upper() == "A":
        pt = np.transpose(resh, (2, 1, 0, 3))
    else:
        pt = np.transpose(resh, (0, 3, 2, 1))
    return pt.reshape(4, 4)


def negativity(rho: np.ndarray) -> float:
    pt = partial_transpose_2q(rho, sys="B")
    eig = np.linalg.eigvalsh(0.5 * (pt + pt.conj().T))
    neg = float(np.sum(np.abs(eig[eig < 0.0])))
    return neg


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def coverage(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    y = np.asarray(y_true)
    lo_arr = np.asarray(lo)
    hi_arr = np.asarray(hi)
    return float(np.mean((y >= lo_arr) & (y <= hi_arr)))


def shots_to_threshold(abs_err: np.ndarray, shot_budgets: np.ndarray, eps: float) -> int:
    idx = np.where(np.asarray(abs_err) <= eps)[0]
    if idx.size == 0:
        return int(shot_budgets[-1])
    return int(shot_budgets[int(idx[0])])


def kl_shift(obs: np.ndarray, pred: np.ndarray, eps: float = 1e-12) -> float:
    o = np.asarray(obs, dtype=float)
    p = np.asarray(pred, dtype=float)
    o = o / max(np.sum(o), eps)
    p = p / max(np.sum(p), eps)
    return float(np.sum(o * np.log((o + eps) / (p + eps))))


def project_psd_trace_one(rho: np.ndarray) -> np.ndarray:
    h = 0.5 * (rho + rho.conj().T)
    evals, evecs = np.linalg.eigh(h)
    evals = np.maximum(np.real(evals), 0.0)
    if np.sum(evals) <= 1e-12:
        return np.eye(h.shape[0], dtype=complex) / h.shape[0]
    rho_psd = (evecs * evals) @ evecs.conj().T
    rho_psd = rho_psd / np.trace(rho_psd)
    return rho_psd


def _basis_rotation(label: str) -> np.ndarray:
    label = label.upper()
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    Sdg = np.array([[1, 0], [0, -1j]], dtype=complex)
    if label == "Z":
        return I2
    if label == "X":
        return H
    if label == "Y":
        return H @ Sdg  # apply Sdg then H
    raise ValueError(f"Unknown basis {label}")


def _probs_for_basis(rho: np.ndarray, basis_a: str, basis_b: str) -> np.ndarray:
    Ua = _basis_rotation(basis_a)
    Ub = _basis_rotation(basis_b)
    U = np.kron(Ua, Ub)
    rho_rot = U @ rho @ U.conj().T
    probs = np.real(np.diag(rho_rot))
    probs = np.clip(probs, 0.0, None)
    s = np.sum(probs)
    if s <= 0:
        probs = np.ones(4) / 4
    else:
        probs = probs / s
    return probs


def simulate_tomography_counts(
    rho: np.ndarray, shots: int, rng: np.random.Generator
) -> dict[tuple[str, str], dict[str, int]]:
    settings = [(a, b) for a in "XYZ" for b in "XYZ"]
    out: dict[tuple[str, str], dict[str, int]] = {}
    for a, b in settings:
        probs = _probs_for_basis(rho, a, b)
        draws = rng.multinomial(shots, probs)
        out[(a, b)] = {
            "00": int(draws[0]),
            "01": int(draws[1]),
            "10": int(draws[2]),
            "11": int(draws[3]),
        }
    return out


def _exp_from_counts(counts: dict[str, int]) -> tuple[float, float, float]:
    total = max(1, sum(counts.values()))
    p00 = counts.get("00", 0) / total
    p01 = counts.get("01", 0) / total
    p10 = counts.get("10", 0) / total
    p11 = counts.get("11", 0) / total
    e_a = (p00 + p01) - (p10 + p11)
    e_b = (p00 + p10) - (p01 + p11)
    e_ab = (p00 + p11) - (p01 + p10)
    return float(e_a), float(e_b), float(e_ab)


def reconstruct_rho_linear_psd(tomo_counts: dict[tuple[str, str], dict[str, int]]) -> np.ndarray:
    # Correlations t_{ij}
    t: dict[tuple[str, str], float] = {}
    ra: dict[str, list[float]] = {"X": [], "Y": [], "Z": []}
    rb: dict[str, list[float]] = {"X": [], "Y": [], "Z": []}

    for a in "XYZ":
        for b in "XYZ":
            e_a, e_b, e_ab = _exp_from_counts(tomo_counts[(a, b)])
            t[(a, b)] = e_ab
            ra[a].append(e_a)
            rb[b].append(e_b)

    r = {k: float(np.mean(v)) for k, v in ra.items()}
    s = {k: float(np.mean(v)) for k, v in rb.items()}

    rho = np.zeros((4, 4), dtype=complex)
    rho += np.kron(I2, I2)
    for i in "XYZ":
        rho += r[i] * np.kron(PAULI[i], I2)
        rho += s[i] * np.kron(I2, PAULI[i])
    for i in "XYZ":
        for j in "XYZ":
            rho += t[(i, j)] * np.kron(PAULI[i], PAULI[j])

    rho *= 0.25
    return project_psd_trace_one(rho)


def tomography_entanglement_estimates(
    rho_true: np.ndarray, shots: int, rng: np.random.Generator
) -> dict[str, Any]:
    counts = simulate_tomography_counts(rho_true, shots=shots, rng=rng)
    rho_hat = reconstruct_rho_linear_psd(counts)
    return {
        "rho_hat": rho_hat,
        "c_hat": concurrence(rho_hat),
        "n_hat": negativity(rho_hat),
        "counts": counts,
    }
