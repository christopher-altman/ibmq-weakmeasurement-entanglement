from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .data import StateParams, rho_ab
from .metrics import concurrence, negativity
from .weak_measurement import BITSTRINGS_3Q, outcome_prob_vector, simulate_counts_for_state


@dataclass(frozen=True)
class DesignSetting:
    g: float
    pointer_basis: str
    b_basis_angle: float = 0.0


class ParticleBelief:
    def __init__(
        self,
        p_grid: np.ndarray,
        theta_grid: np.ndarray,
        candidates: list[DesignSetting],
    ) -> None:
        self.particles: list[StateParams] = [
            StateParams(float(p), float(t)) for p in p_grid for t in theta_grid
        ]
        n = len(self.particles)
        self.weights = np.ones(n, dtype=float) / max(1, n)
        self.candidates = candidates
        self._prob_cache: dict[tuple[int, float, str, float], np.ndarray] = {}

    def _particle_probs(self, idx: int, setting: DesignSetting) -> np.ndarray:
        key = (idx, float(setting.g), setting.pointer_basis.upper(), float(setting.b_basis_angle))
        if key not in self._prob_cache:
            st = self.particles[idx]
            vec = outcome_prob_vector(
                rho_ab(st),
                g=setting.g,
                pointer_basis=setting.pointer_basis,
                b_basis_angle=setting.b_basis_angle,
            )
            self._prob_cache[key] = np.asarray(vec, dtype=float)
        return self._prob_cache[key]

    @staticmethod
    def _entropy(prob: np.ndarray, eps: float = 1e-12) -> float:
        p = np.asarray(prob, dtype=float)
        p = np.clip(p, eps, None)
        p = p / np.sum(p)
        return float(-np.sum(p * np.log(p)))

    def information_gain(self, setting: DesignSetting) -> float:
        mix = np.zeros(8, dtype=float)
        h_cond = 0.0
        for i, w in enumerate(self.weights):
            pi = self._particle_probs(i, setting)
            mix += w * pi
            h_cond += w * self._entropy(pi)
        return self._entropy(mix) - h_cond

    def select_setting(self, policy: str, round_index: int) -> DesignSetting:
        pol = policy.lower()
        if pol == "fixed":
            return self.candidates[round_index % len(self.candidates)]
        if pol != "adaptive":
            raise ValueError(f"Unknown policy: {policy}")

        scored = [(self.information_gain(s), s) for s in self.candidates]
        # Deterministic tie-break: higher IG, then smaller g, then basis order X<Y<Z.
        basis_rank = {"X": 0, "Y": 1, "Z": 2}
        scored.sort(
            key=lambda item: (
                -item[0],
                float(item[1].g),
                basis_rank.get(item[1].pointer_basis.upper(), 99),
                float(item[1].b_basis_angle),
            )
        )
        return scored[0][1]

    def update(self, setting: DesignSetting, counts: dict[str, int], eps: float = 1e-12) -> None:
        cvec = np.array([counts.get(bs, 0) for bs in BITSTRINGS_3Q], dtype=float)
        if np.sum(cvec) <= 0:
            return

        logw = np.log(np.clip(self.weights, eps, None))
        for i in range(len(self.particles)):
            probs = np.clip(self._particle_probs(i, setting), eps, None)
            logw[i] += float(np.sum(cvec * np.log(probs)))

        m = np.max(logw)
        w = np.exp(logw - m)
        z = np.sum(w)
        if z <= eps:
            self.weights[:] = 1.0 / len(self.weights)
        else:
            self.weights[:] = w / z

    def posterior_mean(self) -> StateParams:
        pvals = np.array([s.p for s in self.particles], dtype=float)
        tvals = np.array([s.theta for s in self.particles], dtype=float)
        return StateParams(float(np.sum(self.weights * pvals)), float(np.sum(self.weights * tvals)))

    def posterior_map(self) -> StateParams:
        idx = int(np.argmax(self.weights))
        return self.particles[idx]

    def posterior_entropy(self, eps: float = 1e-12) -> float:
        w = np.clip(self.weights, eps, None)
        w = w / np.sum(w)
        return float(-np.sum(w * np.log(w)))

    def multimodal(self, min_peaks: int = 2, threshold_ratio: float = 0.6) -> bool:
        if len(self.weights) < 2:
            return False
        idx = np.argsort(self.weights)[::-1]
        top = self.weights[idx[: max(min_peaks, 2)]]
        if top.size < 2:
            return False
        return bool(top[1] >= threshold_ratio * top[0])


def build_candidate_settings(g_grid: list[float], pointer_bases: tuple[str, ...] = ("X", "Y")) -> list[DesignSetting]:
    cands: list[DesignSetting] = []
    for g in g_grid:
        for b in pointer_bases:
            cands.append(DesignSetting(g=float(g), pointer_basis=b, b_basis_angle=0.0))
    return cands


def predict_outcome_dist(state_params: StateParams, setting: DesignSetting) -> np.ndarray:
    return outcome_prob_vector(
        rho_ab(state_params),
        g=setting.g,
        pointer_basis=setting.pointer_basis,
        b_basis_angle=setting.b_basis_angle,
    )


def estimate_state_with_policy(
    true_params: StateParams,
    g_grid: list[float],
    policy: str,
    shots: int,
    seed: int,
    rounds: int,
    batch_shots: int | None = None,
    noise: str = "ideal",
    p_points: int = 31,
    theta_points: int = 31,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    candidates = build_candidate_settings(g_grid)
    p_grid = np.linspace(0.0, 1.0, p_points)
    theta_grid = np.linspace(0.0, np.pi / 4, theta_points)
    belief = ParticleBelief(p_grid=p_grid, theta_grid=theta_grid, candidates=candidates)

    if batch_shots is None or batch_shots <= 0:
        batch_shots = max(1, shots // max(1, rounds))

    history: list[dict[str, Any]] = []
    measurement_counts: dict[tuple[float, str], dict[str, int]] = {}
    used = 0

    for r in range(rounds):
        if used >= shots:
            break
        s = belief.select_setting(policy=policy, round_index=r)
        current = min(batch_shots, shots - used)
        counts, probs = simulate_counts_for_state(
            params=true_params,
            g=s.g,
            pointer_basis=s.pointer_basis,
            shots=current,
            rng=rng,
            noise=noise,
            b_basis_angle=s.b_basis_angle,
        )
        belief.update(s, counts)
        used += current

        key = (float(s.g), s.pointer_basis.upper())
        if key not in measurement_counts:
            measurement_counts[key] = {k: 0 for k in BITSTRINGS_3Q}
        for k, v in counts.items():
            measurement_counts[key][k] = measurement_counts[key].get(k, 0) + int(v)

        history.append(
            {
                "round": r,
                "setting": {"g": s.g, "basis": s.pointer_basis, "b_basis_angle": s.b_basis_angle},
                "shots": int(current),
                "counts": {k: int(v) for k, v in counts.items()},
                "posterior_entropy": belief.posterior_entropy(),
                "ig": belief.information_gain(s),
                "probs": {k: float(p) for k, p in zip(BITSTRINGS_3Q, probs)},
            }
        )

    est = belief.posterior_mean()
    rho_est = rho_ab(est)
    c_hat = concurrence(rho_est)
    n_hat = negativity(rho_est)

    return {
        "p_hat": float(est.p),
        "theta_hat": float(est.theta),
        "c_hat": float(c_hat),
        "n_hat": float(n_hat),
        "posterior_entropy": belief.posterior_entropy(),
        "posterior_multimodal": bool(belief.multimodal()),
        "weights": belief.weights.copy(),
        "particles": belief.particles,
        "history": history,
        "measurement_counts": measurement_counts,
        "shots_used": int(used),
    }
