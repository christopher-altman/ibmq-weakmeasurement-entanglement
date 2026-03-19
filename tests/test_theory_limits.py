from __future__ import annotations

import numpy as np

from src.data import StateParams, rho_ab, state_monotones
from src.design import DesignSetting, estimate_state_with_policy, predict_outcome_dist
from src.weak_measurement import simulate_outcome_probabilities


def _conditional_joint_bp_given_a0(probs: dict[str, float]) -> np.ndarray:
    arr = np.zeros((2, 2), dtype=float)
    norm = 0.0
    for bs, p in probs.items():
        a, b, ptr = int(bs[0]), int(bs[1]), int(bs[2])
        if a == 0:
            arr[b, ptr] += p
            norm += p
    if norm <= 1e-12:
        return np.ones((2, 2), dtype=float) / 4.0
    return arr / norm


def _mutual_information_2x2(joint: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(joint, dtype=float)
    p = p / max(np.sum(p), eps)
    pb = np.sum(p, axis=1, keepdims=True)
    pp = np.sum(p, axis=0, keepdims=True)
    return float(np.sum(p * np.log((p + eps) / (pb @ pp + eps))))


def test_zero_coupling_no_information_limit() -> None:
    params = StateParams(p=0.8, theta=0.22)
    probs0 = simulate_outcome_probabilities(rho_ab(params), g=0.0, pointer_basis="X", b_basis_angle=0.0)
    j0 = _conditional_joint_bp_given_a0(probs0)
    mi0 = _mutual_information_2x2(j0)

    setting = DesignSetting(g=0.0, pointer_basis="X", b_basis_angle=0.0)
    dist = predict_outcome_dist(params, setting)
    assert np.isclose(np.sum(dist), 1.0, atol=1e-10)
    assert mi0 < 1e-6


def test_strong_coupling_projective_tendency() -> None:
    params = StateParams(p=0.8, theta=0.22)
    probs0 = simulate_outcome_probabilities(rho_ab(params), g=0.0, pointer_basis="Z", b_basis_angle=0.0)
    probsS = simulate_outcome_probabilities(rho_ab(params), g=np.pi / 2 - 1e-3, pointer_basis="Z", b_basis_angle=0.0)

    mi0 = _mutual_information_2x2(_conditional_joint_bp_given_a0(probs0))
    miS = _mutual_information_2x2(_conditional_joint_bp_given_a0(probsS))
    assert miS > mi0 + 1e-3


def test_known_state_concurrence_recovery() -> None:
    params = StateParams(p=0.92, theta=0.28)
    c_true, _ = state_monotones(params)
    out = estimate_state_with_policy(
        true_params=params,
        g_grid=[0.0, 0.2, 0.35, 0.5, 0.8],
        policy="adaptive",
        shots=4000,
        seed=17,
        rounds=12,
        noise="ideal",
        p_points=21,
        theta_points=21,
    )
    assert abs(float(out["c_hat"]) - c_true) <= 0.12
