from __future__ import annotations

import numpy as np

from src.data import StateParams, rho_ab
from src.metrics import reconstruct_rho_linear_psd, simulate_tomography_counts
from src.models import apply_conformal_interval, split_conformal_calibrate
from src.weak_measurement import feature_block_from_counts_by_basis


def test_density_matrix_validity() -> None:
    params = StateParams(p=0.73, theta=0.31)
    rho = rho_ab(params)
    assert np.allclose(rho, rho.conj().T, atol=1e-10)
    assert np.isclose(np.trace(rho), 1.0, atol=1e-10)
    vals = np.linalg.eigvalsh(rho)
    assert np.min(vals) >= -1e-10


def test_feature_extraction_zero_bins_safe() -> None:
    counts_x = {
        "000": 10,
        "001": 5,
        "010": 0,
        "011": 0,
        "100": 0,
        "101": 0,
        "110": 0,
        "111": 0,
    }
    block = feature_block_from_counts_by_basis({"X": counts_x})
    assert "m0_X" in block and "m1_X" in block
    assert block["valid1_X"] == 0.0
    assert np.isfinite(block["m1_X"])


def test_tomography_reconstruction_reasonable() -> None:
    rng = np.random.default_rng(123)
    rho_true = rho_ab(StateParams(p=0.85, theta=0.25))
    counts = simulate_tomography_counts(rho_true, shots=6000, rng=rng)
    rho_hat = reconstruct_rho_linear_psd(counts)
    assert np.isclose(np.trace(rho_hat), 1.0, atol=1e-8)
    diff = np.linalg.norm(rho_true - rho_hat, ord="fro")
    assert diff < 0.35


def test_conformal_interval_monotone_with_alpha() -> None:
    rng = np.random.default_rng(9)
    y = rng.normal(0.0, 1.0, size=80)
    yhat = y + rng.normal(0.0, 0.2, size=80)
    q90 = split_conformal_calibrate(y, yhat, alpha=0.10)
    q95 = split_conformal_calibrate(y, yhat, alpha=0.05)
    assert q95 >= q90

    y0 = np.array([0.2, -0.1, 0.4])
    lo90, hi90 = apply_conformal_interval(y0, q90)
    lo95, hi95 = apply_conformal_interval(y0, q95)
    assert np.all((hi95 - lo95) >= (hi90 - lo90))
