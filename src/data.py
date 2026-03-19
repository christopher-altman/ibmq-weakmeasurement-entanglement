from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .metrics import concurrence, negativity


@dataclass(frozen=True)
class StateParams:
    p: float
    theta: float


def psi_theta(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([c, 0.0, 0.0, s], dtype=complex)


def rho_b_theta(theta: float) -> np.ndarray:
    c2 = float(np.cos(theta) ** 2)
    s2 = float(np.sin(theta) ** 2)
    return np.array([[c2, 0.0], [0.0, s2]], dtype=complex)


def rho_ab(params: StateParams) -> np.ndarray:
    p = float(np.clip(params.p, 0.0, 1.0))
    theta = float(np.clip(params.theta, 0.0, np.pi / 4))
    psi = psi_theta(theta)
    pure = np.outer(psi, psi.conj())
    mix = 0.5 * np.kron(np.eye(2, dtype=complex), rho_b_theta(theta))
    rho = p * pure + (1.0 - p) * mix
    rho = 0.5 * (rho + rho.conj().T)
    rho = rho / np.trace(rho)
    return rho


def state_monotones(params: StateParams) -> tuple[float, float]:
    rho = rho_ab(params)
    return concurrence(rho), negativity(rho)


def sample_state_params(
    n: int,
    seed: int,
    theta_range: tuple[float, float] = (0.0, np.pi / 4),
    p_range: tuple[float, float] = (0.0, 1.0),
) -> list[StateParams]:
    rng = np.random.default_rng(seed)
    ps = rng.uniform(p_range[0], p_range[1], size=n)
    thetas = rng.uniform(theta_range[0], theta_range[1], size=n)
    return [StateParams(float(p), float(t)) for p, t in zip(ps, thetas)]


def _deterministic_pool(total: int, seed: int) -> list[StateParams]:
    # Deterministic stratified grid then seeded permutation.
    n_theta = int(np.ceil(np.sqrt(total)))
    n_p = int(np.ceil(total / max(n_theta, 1)))

    theta_grid = np.linspace(0.0, np.pi / 4, num=n_theta, endpoint=True)
    p_grid = np.linspace(0.0, 1.0, num=n_p, endpoint=True)

    pool: list[StateParams] = []
    for t in theta_grid:
        for p in p_grid:
            pool.append(StateParams(float(p), float(t)))

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pool))
    sel = [pool[int(i)] for i in idx[:total]]
    return sel


def build_paper_anchor_dataset(
    seed: int = 7,
    total_states: int = 415,
    n_train: int = 349,
    n_test: int = 66,
) -> dict[str, list[dict[str, float | str]]]:
    if n_train + n_test > total_states:
        raise ValueError("n_train + n_test must be <= total_states")

    states = _deterministic_pool(total_states, seed)
    train_states = states[:n_train]
    test_states = states[n_train : n_train + n_test]

    def _format(items: Iterable[StateParams], split: str) -> list[dict[str, float | str]]:
        out = []
        for i, sp in enumerate(items):
            c, n = state_monotones(sp)
            out.append(
                {
                    "state_id": f"{split}_{i:04d}",
                    "split": split,
                    "p": float(sp.p),
                    "theta": float(sp.theta),
                    "c_true": float(c),
                    "n_true": float(n),
                }
            )
        return out

    return {
        "train": _format(train_states, "train"),
        "test": _format(test_states, "test"),
    }


def hardware_mixture_components(params: StateParams) -> list[tuple[str, float]]:
    """
    Decompose rho_AB into classical mixture over pure-circuit components.

    Components:
      entangled: |psi_theta>
      a0b0, a0b1, a1b0, a1b1 product terms
    """
    p = float(np.clip(params.p, 0.0, 1.0))
    theta = float(np.clip(params.theta, 0.0, np.pi / 4))
    c2 = float(np.cos(theta) ** 2)
    s2 = float(np.sin(theta) ** 2)
    w_prod = 0.5 * (1.0 - p)
    comps = [
        ("entangled", p),
        ("a0b0", w_prod * c2),
        ("a0b1", w_prod * s2),
        ("a1b0", w_prod * c2),
        ("a1b1", w_prod * s2),
    ]
    total = sum(w for _, w in comps)
    if total <= 0:
        return [("entangled", 1.0)]
    return [(k, float(w / total)) for k, w in comps if w > 0]


def allocate_shots_by_weights(weights: list[tuple[str, float]], shots: int) -> list[tuple[str, int]]:
    base = [(k, int(np.floor(w * shots))) for k, w in weights]
    used = sum(x for _, x in base)
    rem = shots - used
    if rem > 0:
        order = sorted(weights, key=lambda x: x[1], reverse=True)
        for i in range(rem):
            tag = order[i % len(order)][0]
            for j, (k, cnt) in enumerate(base):
                if k == tag:
                    base[j] = (k, cnt + 1)
                    break
    return base
