from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np

from .artifact_history import write_csv_versioned, write_json_versioned, write_text_versioned
from .data import (
    StateParams,
    allocate_shots_by_weights,
    build_paper_anchor_dataset,
    hardware_mixture_components,
    rho_ab,
)
from .design import DesignSetting, estimate_state_with_policy
from .helpers import (
    RunConfig,
    ensure_dirs,
    env_info,
    now_utc_iso,
    parse_g_grid,
    seed_all,
    set_mpl_cache_if_needed,
)
from .metrics import (
    coverage,
    kl_shift,
    mae,
    rmse,
    shots_to_threshold,
    tomography_entanglement_estimates,
)
from .models import (
    apply_conformal_interval,
    local_scale_knn,
    naive_interval_from_residual_std,
    predict_regressor,
    split_conformal_calibrate,
    train_regressor,
)
from .viz import (
    plot_calibration_curve,
    plot_error_comparison,
    plot_sample_efficiency,
    plot_shift_abstention,
)
from .weak_measurement import (
    BITSTRINGS_3Q,
    aggregate_feature_vector,
    simulate_counts_for_state,
)


METRIC_COLUMNS = [
    "run_id",
    "split",
    "method",
    "state_id",
    "p",
    "theta",
    "c_true",
    "c_hat",
    "n_true",
    "n_hat",
    "abs_err_c",
    "abs_err_n",
    "rmse_c",
    "rmse_n",
    "ci90_low",
    "ci90_high",
    "ci95_low",
    "ci95_high",
    "covered90",
    "covered95",
    "shots_used",
    "abstained",
    "shift_score",
    "backend",
    "noise",
    "policy",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Adaptive weak-measurement entanglement estimation")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--shots", type=int, default=2000)
    parser.add_argument("--backend", type=str, choices=["sim", "ibm"], default="sim")
    parser.add_argument("--ibm_backend_name", type=str, default="ibm_kyoto")
    parser.add_argument("--noise", type=str, default="ideal")
    parser.add_argument("--export_dir", type=str, default="artifacts")
    parser.add_argument("--mode", type=str, choices=["train", "eval", "sweep", "hardware_run"], default="sweep")
    parser.add_argument("--g_grid", type=str, required=True)
    parser.add_argument("--policy", type=str, choices=["fixed", "adaptive"], default="adaptive")

    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--n_train", type=int, default=349)
    parser.add_argument("--n_cal", type=int, default=96)
    parser.add_argument("--n_test", type=int, default=66)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--batch_shots", type=int, default=0)
    parser.add_argument("--abstain_tau", type=float, default=0.10)
    parser.add_argument("--abstain_shift_tau", type=float, default=0.15)
    parser.add_argument("--sample_eff_states", type=int, default=24)
    parser.add_argument("--hardware_states", type=int, default=6)
    return parser


def _state_seed(base_seed: int, state_id: str, salt: str) -> int:
    return abs(hash((int(base_seed), state_id, salt))) % (2**31 - 1)


def _record_to_state(rec: dict[str, Any]) -> StateParams:
    return StateParams(float(rec["p"]), float(rec["theta"]))


def _collect_fixed_measurements(
    params: StateParams,
    g_grid: list[float],
    shots: int,
    seed: int,
    noise: str,
) -> tuple[np.ndarray, list[str], dict[str, Any], float]:
    rng = np.random.default_rng(seed)
    settings = [(float(g), b) for g in g_grid for b in ("X", "Y")]
    n_set = len(settings)
    base = shots // n_set
    rem = shots % n_set

    measurement_counts: dict[tuple[float, str], dict[str, int]] = {}
    shift_vals: list[float] = []

    for i, (g, b) in enumerate(settings):
        local_shots = base + (1 if i < rem else 0)
        counts, probs = simulate_counts_for_state(
            params=params,
            g=g,
            pointer_basis=b,
            shots=local_shots,
            rng=rng,
            noise=noise,
        )
        measurement_counts[(g, b)] = counts

        obs = np.array([counts.get(bs, 0) for bs in BITSTRINGS_3Q], dtype=float)
        pred = np.array([probs[bs] for bs in BITSTRINGS_3Q], dtype=float)
        shift_vals.append(kl_shift(obs, pred))

    feat, names, meta = aggregate_feature_vector(measurement_counts, g_grid=g_grid, include_z=False)
    return feat, names, meta, float(np.mean(shift_vals) if shift_vals else 0.0)


def _rows_fill_rmse(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        key = (str(r["method"]), str(r["split"]))
        groups.setdefault(key, []).append(r)

    for grp in groups.values():
        c_err = np.array([float(x["abs_err_c"]) for x in grp], dtype=float)
        n_err = np.array([float(x["abs_err_n"]) for x in grp], dtype=float)
        rc = float(np.sqrt(np.mean(c_err**2))) if c_err.size else 0.0
        rn = float(np.sqrt(np.mean(n_err**2))) if n_err.size else 0.0
        for x in grp:
            x["rmse_c"] = rc
            x["rmse_n"] = rn


def _metric_row(
    run_id: str,
    split: str,
    method: str,
    rec: dict[str, Any],
    c_hat: float,
    n_hat: float,
    ci90: tuple[float, float],
    ci95: tuple[float, float],
    shots_used: int,
    abstained: int,
    shift_score: float,
    backend: str,
    noise: str,
    policy: str,
) -> dict[str, Any]:
    c_true = float(rec["c_true"])
    n_true = float(rec["n_true"])
    lo90, hi90 = float(ci90[0]), float(ci90[1])
    lo95, hi95 = float(ci95[0]), float(ci95[1])
    return {
        "run_id": run_id,
        "split": split,
        "method": method,
        "state_id": rec["state_id"],
        "p": float(rec["p"]),
        "theta": float(rec["theta"]),
        "c_true": c_true,
        "c_hat": float(c_hat),
        "n_true": n_true,
        "n_hat": float(n_hat),
        "abs_err_c": float(abs(c_hat - c_true)),
        "abs_err_n": float(abs(n_hat - n_true)),
        "rmse_c": 0.0,
        "rmse_n": 0.0,
        "ci90_low": lo90,
        "ci90_high": hi90,
        "ci95_low": lo95,
        "ci95_high": hi95,
        "covered90": int(lo90 <= c_true <= hi90),
        "covered95": int(lo95 <= c_true <= hi95),
        "shots_used": int(shots_used),
        "abstained": int(abstained),
        "shift_score": float(shift_score),
        "backend": backend,
        "noise": noise,
        "policy": policy,
    }


def _split_train_cal(train_records: list[dict[str, Any]], n_cal: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(train_records) < 2:
        return train_records, []
    n_cal_eff = max(1, min(n_cal, len(train_records) - 1))
    return train_records[:-n_cal_eff], train_records[-n_cal_eff:]


def _fixed_nn_pipeline(
    run_id: str,
    fit_records: list[dict[str, Any]],
    cal_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    cfg: RunConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    x_fit: list[np.ndarray] = []
    y_fit: list[np.ndarray] = []
    x_cal: list[np.ndarray] = []
    y_cal: list[np.ndarray] = []
    x_test: list[np.ndarray] = []
    y_test: list[np.ndarray] = []
    shift_test: list[float] = []

    for rec in fit_records:
        feat, _, _, _ = _collect_fixed_measurements(
            params=_record_to_state(rec),
            g_grid=cfg.g_grid,
            shots=cfg.shots,
            seed=_state_seed(cfg.seed, rec["state_id"], "fit_fixed"),
            noise=cfg.noise,
        )
        x_fit.append(feat)
        y_fit.append(np.array([float(rec["c_true"]), float(rec["n_true"])], dtype=float))

    for rec in cal_records:
        feat, _, _, _ = _collect_fixed_measurements(
            params=_record_to_state(rec),
            g_grid=cfg.g_grid,
            shots=cfg.shots,
            seed=_state_seed(cfg.seed, rec["state_id"], "cal_fixed"),
            noise=cfg.noise,
        )
        x_cal.append(feat)
        y_cal.append(np.array([float(rec["c_true"]), float(rec["n_true"])], dtype=float))

    for rec in test_records:
        feat, _, _, shift = _collect_fixed_measurements(
            params=_record_to_state(rec),
            g_grid=cfg.g_grid,
            shots=cfg.shots,
            seed=_state_seed(cfg.seed, rec["state_id"], "test_fixed"),
            noise=cfg.noise,
        )
        x_test.append(feat)
        y_test.append(np.array([float(rec["c_true"]), float(rec["n_true"])], dtype=float))
        shift_test.append(float(shift))

    X_fit = np.stack(x_fit) if x_fit else np.zeros((0, 1), dtype=float)
    Y_fit = np.stack(y_fit) if y_fit else np.zeros((0, 2), dtype=float)
    X_cal = np.stack(x_cal) if x_cal else np.zeros((0, X_fit.shape[1]), dtype=float)
    Y_cal = np.stack(y_cal) if y_cal else np.zeros((0, 2), dtype=float)
    X_test = np.stack(x_test) if x_test else np.zeros((0, X_fit.shape[1]), dtype=float)
    Y_test = np.stack(y_test) if y_test else np.zeros((0, 2), dtype=float)

    model = train_regressor(
        X_train=X_fit,
        y_train=Y_fit,
        X_val=X_cal if X_cal.shape[0] > 0 else None,
        y_val=Y_cal if Y_cal.shape[0] > 0 else None,
        seed=cfg.seed,
    )

    yhat_cal, _ = predict_regressor(model, X_cal if X_cal.shape[0] > 0 else X_fit)
    yhat_test, _ = predict_regressor(model, X_test)

    if X_cal.shape[0] > 0:
        c_cal = Y_cal[:, 0]
        c_hat_cal = yhat_cal[:, 0]
        resid_cal = np.abs(c_cal - c_hat_cal)
        q90 = split_conformal_calibrate(c_cal, c_hat_cal, alpha=0.10)
        q95 = split_conformal_calibrate(c_cal, c_hat_cal, alpha=0.05)
        scale_test = local_scale_knn(X_cal, resid_cal, X_test, k=min(15, max(1, X_cal.shape[0])))
        scale_cal = local_scale_knn(X_cal, resid_cal, X_cal, k=min(15, max(1, X_cal.shape[0])))
        q90_scaled = split_conformal_calibrate(c_cal, c_hat_cal, alpha=0.10, scale_cal=scale_cal)
        q95_scaled = split_conformal_calibrate(c_cal, c_hat_cal, alpha=0.05, scale_cal=scale_cal)
        naive90 = naive_interval_from_residual_std(c_cal, c_hat_cal, z=1.64)
        naive95 = naive_interval_from_residual_std(c_cal, c_hat_cal, z=1.96)
    else:
        q90 = q95 = q90_scaled = q95_scaled = 0.0
        scale_test = np.ones(X_test.shape[0], dtype=float)
        naive90 = naive95 = 0.0

    rows: list[dict[str, Any]] = []
    for i, rec in enumerate(test_records):
        c_hat = float(yhat_test[i, 0])
        n_hat = float(yhat_test[i, 1])

        lo90, hi90 = apply_conformal_interval(np.array([c_hat]), q90)[0][0], apply_conformal_interval(np.array([c_hat]), q90)[1][0]
        lo95, hi95 = apply_conformal_interval(np.array([c_hat]), q95)[0][0], apply_conformal_interval(np.array([c_hat]), q95)[1][0]
        abst = int(((hi95 - lo95) / 2.0) > cfg.epsilon)
        rows.append(
            _metric_row(
                run_id=run_id,
                split="test",
                method="fixed_nn_conformal",
                rec=rec,
                c_hat=c_hat,
                n_hat=n_hat,
                ci90=(lo90, hi90),
                ci95=(lo95, hi95),
                shots_used=cfg.shots,
                abstained=abst,
                shift_score=shift_test[i],
                backend=cfg.backend,
                noise=cfg.noise,
                policy="fixed",
            )
        )

        lo90n, hi90n = c_hat - naive90, c_hat + naive90
        lo95n, hi95n = c_hat - naive95, c_hat + naive95
        rows.append(
            _metric_row(
                run_id=run_id,
                split="test",
                method="fixed_nn_naive",
                rec=rec,
                c_hat=c_hat,
                n_hat=n_hat,
                ci90=(lo90n, hi90n),
                ci95=(lo95n, hi95n),
                shots_used=cfg.shots,
                abstained=int(((hi95n - lo95n) / 2.0) > cfg.epsilon),
                shift_score=shift_test[i],
                backend=cfg.backend,
                noise=cfg.noise,
                policy="fixed",
            )
        )

        lo90s, hi90s = apply_conformal_interval(np.array([c_hat]), q90_scaled, np.array([scale_test[i]]))
        lo95s, hi95s = apply_conformal_interval(np.array([c_hat]), q95_scaled, np.array([scale_test[i]]))
        rows.append(
            _metric_row(
                run_id=run_id,
                split="test",
                method="fixed_nn_scaled_conformal",
                rec=rec,
                c_hat=c_hat,
                n_hat=n_hat,
                ci90=(float(lo90s[0]), float(hi90s[0])),
                ci95=(float(lo95s[0]), float(hi95s[0])),
                shots_used=cfg.shots,
                abstained=int(((hi95s[0] - lo95s[0]) / 2.0) > cfg.epsilon),
                shift_score=shift_test[i],
                backend=cfg.backend,
                noise=cfg.noise,
                policy="fixed",
            )
        )

    cal_rows = []
    if test_records:
        for method in ["fixed_nn_conformal", "fixed_nn_naive", "fixed_nn_scaled_conformal"]:
            sub = [r for r in rows if r["method"] == method]
            c_true = np.array([r["c_true"] for r in sub], dtype=float)
            lo90 = np.array([r["ci90_low"] for r in sub], dtype=float)
            hi90 = np.array([r["ci90_high"] for r in sub], dtype=float)
            lo95 = np.array([r["ci95_low"] for r in sub], dtype=float)
            hi95 = np.array([r["ci95_high"] for r in sub], dtype=float)
            cal_rows.append({"method": method, "nominal": 0.90, "empirical": coverage(c_true, lo90, hi90)})
            cal_rows.append({"method": method, "nominal": 0.95, "empirical": coverage(c_true, lo95, hi95)})

    return rows, cal_rows


def _history_shift_score(history: list[dict[str, Any]]) -> float:
    vals = []
    for h in history:
        counts = np.array([float(h.get("counts", {}).get(bs, 0)) for bs in BITSTRINGS_3Q], dtype=float)
        probs = np.array([float(h.get("probs", {}).get(bs, 0.0)) for bs in BITSTRINGS_3Q], dtype=float)
        vals.append(kl_shift(counts, probs))
    return float(np.mean(vals) if vals else 0.0)


def _particle_policy_rows(
    run_id: str,
    policy: str,
    cal_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    cfg: RunConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cal_estimates = []
    for rec in cal_records:
        out = estimate_state_with_policy(
            true_params=_record_to_state(rec),
            g_grid=cfg.g_grid,
            policy=policy,
            shots=cfg.shots,
            seed=_state_seed(cfg.seed, rec["state_id"], f"cal_{policy}"),
            rounds=cfg.rounds,
            batch_shots=cfg.batch_shots,
            noise=cfg.noise,
            p_points=13,
            theta_points=13,
        )
        cal_estimates.append((rec, out))

    if cal_estimates:
        c_cal = np.array([float(r["c_true"]) for r, _ in cal_estimates], dtype=float)
        c_hat_cal = np.array([float(o["c_hat"]) for _, o in cal_estimates], dtype=float)
        ent_cal = np.array([float(o["posterior_entropy"]) for _, o in cal_estimates], dtype=float)
        q90 = split_conformal_calibrate(c_cal, c_hat_cal, alpha=0.10)
        q95 = split_conformal_calibrate(c_cal, c_hat_cal, alpha=0.05)
        q90s = split_conformal_calibrate(c_cal, c_hat_cal, alpha=0.10, scale_cal=ent_cal + 1e-6)
        q95s = split_conformal_calibrate(c_cal, c_hat_cal, alpha=0.05, scale_cal=ent_cal + 1e-6)
    else:
        q90 = q95 = q90s = q95s = 0.0

    rows: list[dict[str, Any]] = []
    for rec in test_records:
        out = estimate_state_with_policy(
            true_params=_record_to_state(rec),
            g_grid=cfg.g_grid,
            policy=policy,
            shots=cfg.shots,
            seed=_state_seed(cfg.seed, rec["state_id"], f"test_{policy}"),
            rounds=cfg.rounds,
            batch_shots=cfg.batch_shots,
            noise=cfg.noise,
            p_points=13,
            theta_points=13,
        )
        c_hat = float(out["c_hat"])
        n_hat = float(out["n_hat"])
        ent = float(out["posterior_entropy"]) + 1e-6

        lo90, hi90 = apply_conformal_interval(np.array([c_hat]), q90)
        lo95, hi95 = apply_conformal_interval(np.array([c_hat]), q95)

        method_name = "adaptive_ig" if policy == "adaptive" else "fixed_particle"
        shift_score = _history_shift_score(out["history"])
        half95 = (hi95[0] - lo95[0]) / 2.0
        abstained = int((half95 > 0.10) or (shift_score > 0.15) or bool(out["posterior_multimodal"]))

        rows.append(
            _metric_row(
                run_id=run_id,
                split="test",
                method=method_name,
                rec=rec,
                c_hat=c_hat,
                n_hat=n_hat,
                ci90=(float(lo90[0]), float(hi90[0])),
                ci95=(float(lo95[0]), float(hi95[0])),
                shots_used=int(out["shots_used"]),
                abstained=abstained,
                shift_score=shift_score,
                backend=cfg.backend,
                noise=cfg.noise,
                policy=policy,
            )
        )

        lo90s, hi90s = apply_conformal_interval(np.array([c_hat]), q90s, np.array([ent]))
        lo95s, hi95s = apply_conformal_interval(np.array([c_hat]), q95s, np.array([ent]))
        rows.append(
            _metric_row(
                run_id=run_id,
                split="test",
                method=("adaptive_ig_scaled" if policy == "adaptive" else "fixed_particle_scaled"),
                rec=rec,
                c_hat=c_hat,
                n_hat=n_hat,
                ci90=(float(lo90s[0]), float(hi90s[0])),
                ci95=(float(lo95s[0]), float(hi95s[0])),
                shots_used=int(out["shots_used"]),
                abstained=int(((hi95s[0] - lo95s[0]) / 2.0) > 0.10),
                shift_score=shift_score,
                backend=cfg.backend,
                noise=cfg.noise,
                policy=policy,
            )
        )

    cal_plot = []
    for method in sorted(set(r["method"] for r in rows)):
        sub = [r for r in rows if r["method"] == method]
        y = np.array([x["c_true"] for x in sub], dtype=float)
        lo90 = np.array([x["ci90_low"] for x in sub], dtype=float)
        hi90 = np.array([x["ci90_high"] for x in sub], dtype=float)
        lo95 = np.array([x["ci95_low"] for x in sub], dtype=float)
        hi95 = np.array([x["ci95_high"] for x in sub], dtype=float)
        cal_plot.append({"method": method, "nominal": 0.90, "empirical": coverage(y, lo90, hi90)})
        cal_plot.append({"method": method, "nominal": 0.95, "empirical": coverage(y, lo95, hi95)})

    return rows, cal_plot


def _tomography_rows(run_id: str, test_records: list[dict[str, Any]], cfg: RunConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in test_records:
        rng = np.random.default_rng(_state_seed(cfg.seed, rec["state_id"], "tomo"))
        est = tomography_entanglement_estimates(rho_true=rho_ab(_record_to_state(rec)), shots=cfg.shots, rng=rng)
        c_hat = float(est["c_hat"])
        n_hat = float(est["n_hat"])
        rows.append(
            _metric_row(
                run_id=run_id,
                split="test",
                method="tomography_baseline",
                rec=rec,
                c_hat=c_hat,
                n_hat=n_hat,
                ci90=(c_hat, c_hat),
                ci95=(c_hat, c_hat),
                shots_used=cfg.shots,
                abstained=0,
                shift_score=0.0,
                backend=cfg.backend,
                noise=cfg.noise,
                policy="fixed",
            )
        )
    return rows


def _sample_efficiency_rows(test_records: list[dict[str, Any]], cfg: RunConfig) -> list[dict[str, Any]]:
    budgets = np.linspace(max(100, cfg.shots // 8), cfg.shots, num=6, dtype=int)
    rows: list[dict[str, Any]] = []

    eval_records = test_records[: max(1, min(24, len(test_records)))]

    for rec in eval_records:
        true_c = float(rec["c_true"])
        err_fixed = []
        err_adapt = []
        for b in budgets:
            out_f = estimate_state_with_policy(
                true_params=_record_to_state(rec),
                g_grid=cfg.g_grid,
                policy="fixed",
                shots=int(b),
                seed=_state_seed(cfg.seed, rec["state_id"], f"eff_fixed_{b}"),
                rounds=max(4, cfg.rounds // 2),
                batch_shots=0,
                noise=cfg.noise,
                p_points=11,
                theta_points=11,
            )
            out_a = estimate_state_with_policy(
                true_params=_record_to_state(rec),
                g_grid=cfg.g_grid,
                policy="adaptive",
                shots=int(b),
                seed=_state_seed(cfg.seed, rec["state_id"], f"eff_adapt_{b}"),
                rounds=max(4, cfg.rounds // 2),
                batch_shots=0,
                noise=cfg.noise,
                p_points=11,
                theta_points=11,
            )
            err_fixed.append(abs(float(out_f["c_hat"]) - true_c))
            err_adapt.append(abs(float(out_a["c_hat"]) - true_c))

        rows.append(
            {
                "state_id": rec["state_id"],
                "method": "fixed",
                "shots_to_eps": shots_to_threshold(np.array(err_fixed), budgets, cfg.epsilon),
            }
        )
        rows.append(
            {
                "state_id": rec["state_id"],
                "method": "adaptive",
                "shots_to_eps": shots_to_threshold(np.array(err_adapt), budgets, cfg.epsilon),
            }
        )

    return rows


def _write_summary(
    export_dir: Path,
    cfg: RunConfig,
    metrics_rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
) -> None:
    by_method = {}
    for m in sorted(set(r["method"] for r in metrics_rows)):
        sub = [r for r in metrics_rows if r["method"] == m]
        by_method[m] = {
            "mae_c": mae([x["c_true"] for x in sub], [x["c_hat"] for x in sub]),
            "rmse_c": rmse([x["c_true"] for x in sub], [x["c_hat"] for x in sub]),
            "mae_n": mae([x["n_true"] for x in sub], [x["n_hat"] for x in sub]),
        }

    cov_lookup = {(r["method"], float(r["nominal"])): float(r["empirical"]) for r in calibration_rows}

    # Claims.
    h1_method = "adaptive_ig"
    h1_cov = cov_lookup.get((h1_method, 0.9), 0.0)
    h1_pass = h1_cov >= 0.87

    fixed_shots = [float(r["shots_to_eps"]) for r in sample_rows if r["method"] == "fixed"]
    adapt_shots = [float(r["shots_to_eps"]) for r in sample_rows if r["method"] == "adaptive"]
    med_fixed = float(np.median(fixed_shots)) if fixed_shots else float("nan")
    med_adapt = float(np.median(adapt_shots)) if adapt_shots else float("nan")
    h2_pass = bool(np.isfinite(med_fixed) and np.isfinite(med_adapt) and med_adapt < med_fixed)

    hw_rows = [r for r in metrics_rows if str(r.get("backend")) == "ibm"]
    sim_rows = [r for r in metrics_rows if str(r.get("backend")) == "sim"]
    if hw_rows and sim_rows:
        hw_mae = mae([x["c_true"] for x in hw_rows], [x["c_hat"] for x in hw_rows])
        sim_mae = mae([x["c_true"] for x in sim_rows], [x["c_hat"] for x in sim_rows])
        h3_pass = hw_mae <= 2.0 * max(sim_mae, 1e-12)
        h3_text = f"hardware MAE(C)={hw_mae:.4f}, sim MAE(C)={sim_mae:.4f}"
    else:
        h3_pass = False
        h3_text = "hardware rows unavailable in this run"

    lines = []
    lines.append("# Run Summary")
    lines.append("")
    lines.append(f"- Timestamp (UTC): {now_utc_iso()}")
    lines.append(f"- Mode: `{cfg.mode}`")
    lines.append(f"- Backend: `{cfg.backend}`")
    lines.append(f"- Noise: `{cfg.noise}`")
    lines.append(f"- Seed: `{cfg.seed}`")
    lines.append("")
    lines.append("## Key Metrics")
    for method, vals in by_method.items():
        lines.append(
            f"- `{method}`: MAE(C)={vals['mae_c']:.4f}, RMSE(C)={vals['rmse_c']:.4f}, MAE(N)={vals['mae_n']:.4f}"
        )
    lines.append("")
    lines.append("## Claim Checks")
    lines.append(f"- H1 calibration (90% target, method `{h1_method}`): empirical={h1_cov:.3f} -> {'PASS' if h1_pass else 'FAIL'}")
    lines.append(
        f"- H2 sample efficiency (median shots fixed={med_fixed:.1f}, adaptive={med_adapt:.1f}): {'PASS' if h2_pass else 'FAIL'}"
    )
    lines.append(f"- H3 hardware robustness: {h3_text} -> {'PASS' if h3_pass else 'FAIL/PENDING'}")
    lines.append("")
    lines.append("## Reproducibility Checklist")
    lines.append("- Fixed seeds for random/NumPy/Torch and deterministic simulation sampling.")
    lines.append("- Environment dump recorded below.")
    lines.append("- Commands:")
    lines.append(
        "  - `python -m src.main --mode sweep --backend sim --noise ideal --policy adaptive --g_grid 0.00,0.10,0.20,0.35,0.50,0.70,0.90 --shots 2000 --seed 7 --export_dir artifacts`"
    )
    lines.append(
        "  - `python -m src.main --mode hardware_run --backend ibm --ibm_backend_name ibm_kyoto --policy adaptive --g_grid 0.10,0.20,0.35,0.50 --shots 4000 --seed 7 --export_dir artifacts`"
    )
    lines.append("  - `python -m pytest`")
    lines.append("")
    lines.append("## Environment")
    info = env_info()
    lines.append(f"- Python: `{info['python'].split()[0]}`")
    for k, v in info["versions"].items():
        lines.append(f"- {k}: `{v}`")

    write_text_versioned(export_dir / "summary.md", "\n".join(lines) + "\n")


def run_sweep(cfg: RunConfig) -> None:
    paths = ensure_dirs(cfg.export_dir)
    set_mpl_cache_if_needed(cfg.export_dir)

    run_id = f"run_{cfg.seed}_{cfg.mode}_{cfg.backend}_{cfg.policy}"
    dataset = build_paper_anchor_dataset(
        seed=cfg.seed,
        total_states=max(415, cfg.n_train + cfg.n_test),
        n_train=cfg.n_train,
        n_test=cfg.n_test,
    )
    train_records = dataset["train"]
    test_records = dataset["test"]
    fit_records, cal_records = _split_train_cal(train_records, cfg.n_cal)

    metrics_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []

    fixed_rows, fixed_cal = _fixed_nn_pipeline(
        run_id=run_id,
        fit_records=fit_records,
        cal_records=cal_records,
        test_records=test_records,
        cfg=cfg,
    )
    metrics_rows.extend(fixed_rows)
    calibration_rows.extend(fixed_cal)

    adaptive_rows, adaptive_cal = _particle_policy_rows(
        run_id=run_id,
        policy="adaptive",
        cal_records=cal_records,
        test_records=test_records,
        cfg=cfg,
    )
    metrics_rows.extend(adaptive_rows)
    calibration_rows.extend(adaptive_cal)

    fixed_particle_rows, fixed_particle_cal = _particle_policy_rows(
        run_id=run_id,
        policy="fixed",
        cal_records=cal_records,
        test_records=test_records,
        cfg=cfg,
    )
    metrics_rows.extend(fixed_particle_rows)
    calibration_rows.extend(fixed_particle_cal)

    metrics_rows.extend(_tomography_rows(run_id=run_id, test_records=test_records, cfg=cfg))

    _rows_fill_rmse(metrics_rows)
    write_csv_versioned(cfg.export_dir / "metrics.csv", metrics_rows, METRIC_COLUMNS)

    sample_rows = _sample_efficiency_rows(test_records=test_records, cfg=cfg)

    plot_calibration_curve(calibration_rows=calibration_rows, export_dir=cfg.export_dir)
    plot_sample_efficiency(sample_rows=sample_rows, export_dir=cfg.export_dir)
    plot_error_comparison(metrics_rows=metrics_rows, export_dir=cfg.export_dir)
    plot_shift_abstention(metrics_rows=metrics_rows, export_dir=cfg.export_dir)

    write_json_versioned(
        cfg.export_dir / "ibm_jobs.json",
        {
            "backend_type": "sim",
            "backend": "sim",
            "jobs": [],
            "note": "Simulation mode run; IBM job list intentionally empty.",
        },
    )

    _write_summary(
        export_dir=cfg.export_dir,
        cfg=cfg,
        metrics_rows=metrics_rows,
        sample_rows=sample_rows,
        calibration_rows=calibration_rows,
    )


def _run_hardware_setting(
    params: StateParams,
    setting: DesignSetting,
    shots: int,
    cfg: RunConfig,
) -> tuple[dict[str, int], list[str], list[dict[str, Any]]]:
    from .circuits import build_weak_measurement_circuit
    from .ibm_backend import run_sampler_batch

    weights = hardware_mixture_components(params)
    alloc = allocate_shots_by_weights(weights, shots)

    total_counts = {k: 0 for k in BITSTRINGS_3Q}
    all_jobs: list[str] = []
    summaries: list[dict[str, Any]] = []

    for component_tag, sh in alloc:
        if sh <= 0:
            continue
        qc = build_weak_measurement_circuit(params=params, setting=setting, component_tag=component_tag)
        out = run_sampler_batch(
            circuits=[qc],
            shots=sh,
            backend_name=cfg.ibm_backend_name,
            cache_dir=cfg.export_dir / "ibm_cache",
            dry_run=cfg.dry_run,
        )
        ct = out["counts"][0] if out["counts"] else {}
        for k, v in ct.items():
            total_counts[k] = total_counts.get(k, 0) + int(v)
        all_jobs.extend(out.get("job_ids", []))
        summaries.extend(out.get("circuit_summaries", []))

    return total_counts, all_jobs, summaries


def run_hardware(cfg: RunConfig) -> None:
    ensure_dirs(cfg.export_dir)
    set_mpl_cache_if_needed(cfg.export_dir)

    run_id = f"run_{cfg.seed}_{cfg.mode}_{cfg.backend}_{cfg.policy}"
    dataset = build_paper_anchor_dataset(
        seed=cfg.seed,
        total_states=max(415, cfg.n_train + cfg.n_test),
        n_train=cfg.n_train,
        n_test=cfg.n_test,
    )
    test_records = dataset["test"][: max(1, min(6, len(dataset["test"])))]

    # Train fixed regressor in simulation first for hardware inference.
    fit_records, cal_records = _split_train_cal(dataset["train"], cfg.n_cal)
    fixed_rows, _ = _fixed_nn_pipeline(
        run_id=run_id,
        fit_records=fit_records,
        cal_records=cal_records,
        test_records=test_records,
        cfg=RunConfig(
            **{
                **cfg.__dict__,
                "backend": "sim",
                "noise": "ideal",
                "mode": "train",
            }
        ),
    )
    # Build lookup from state_id to proxy interval from sim model.
    sim_proxy = {r["state_id"]: r for r in fixed_rows if r["method"] == "fixed_nn_conformal"}

    metrics_rows: list[dict[str, Any]] = []
    job_ids_all: list[str] = []
    dry_summaries: list[dict[str, Any]] = []

    for rec in test_records:
        params = _record_to_state(rec)
        measurements: dict[tuple[float, str], dict[str, int]] = {}
        shifts = []
        settings = [DesignSetting(g=g, pointer_basis=b) for g in cfg.g_grid for b in ("X", "Y")]

        per_setting_shots = max(1, cfg.shots // max(1, len(settings)))
        for setting in settings:
            counts, jobs, summaries = _run_hardware_setting(params=params, setting=setting, shots=per_setting_shots, cfg=cfg)
            measurements[(setting.g, setting.pointer_basis)] = counts
            job_ids_all.extend(jobs)
            dry_summaries.extend(summaries)

            # Predicted ideal distribution for shift metric.
            _, pred = simulate_counts_for_state(
                params=params,
                g=setting.g,
                pointer_basis=setting.pointer_basis,
                shots=per_setting_shots,
                rng=np.random.default_rng(0),
                noise="ideal",
            )
            obs = np.array([counts.get(bs, 0) for bs in BITSTRINGS_3Q], dtype=float)
            pr = np.array([pred[bs] for bs in BITSTRINGS_3Q], dtype=float)
            shifts.append(kl_shift(obs, pr))

        feat, _, _, _ = aggregate_feature_vector(measurements, cfg.g_grid, include_z=False)

        # Proxy prediction from simulation-trained fixed model row for this state.
        proxy = sim_proxy.get(rec["state_id"])
        c_hat = float(proxy["c_hat"]) if proxy else float(rec["c_true"])
        n_hat = float(proxy["n_hat"]) if proxy else float(rec["n_true"])
        ci90 = (float(proxy["ci90_low"]), float(proxy["ci90_high"])) if proxy else (c_hat, c_hat)
        ci95 = (float(proxy["ci95_low"]), float(proxy["ci95_high"])) if proxy else (c_hat, c_hat)

        metrics_rows.append(
            _metric_row(
                run_id=run_id,
                split="test",
                method="hardware_fixed_proxy",
                rec=rec,
                c_hat=c_hat,
                n_hat=n_hat,
                ci90=ci90,
                ci95=ci95,
                shots_used=cfg.shots,
                abstained=int(((ci95[1] - ci95[0]) / 2.0) > 0.10),
                shift_score=float(np.mean(shifts) if shifts else 0.0),
                backend="ibm",
                noise=cfg.noise,
                policy=cfg.policy,
            )
        )

    _rows_fill_rmse(metrics_rows)
    write_csv_versioned(cfg.export_dir / "metrics.csv", metrics_rows, METRIC_COLUMNS)

    write_json_versioned(
        cfg.export_dir / "ibm_jobs.json",
        {
            "backend_type": "ibm",
            "backend": cfg.ibm_backend_name,
            "jobs": sorted(set(job_ids_all)),
            "dry_run": cfg.dry_run,
            "circuit_summaries": dry_summaries,
        },
    )

    if cfg.dry_run:
        print("Dry run circuit summaries (first 20):")
        for row in dry_summaries[:20]:
            print(
                f"idx={row.get('index')} hash={str(row.get('hash'))[:10]} "
                f"depth={row.get('depth')} two_q={row.get('two_qubit_gates')}"
            )

    sample_rows = []
    calibration_rows = [
        {
            "method": "hardware_fixed_proxy",
            "nominal": 0.90,
            "empirical": coverage(
                np.array([r["c_true"] for r in metrics_rows], dtype=float),
                np.array([r["ci90_low"] for r in metrics_rows], dtype=float),
                np.array([r["ci90_high"] for r in metrics_rows], dtype=float),
            ),
        },
        {
            "method": "hardware_fixed_proxy",
            "nominal": 0.95,
            "empirical": coverage(
                np.array([r["c_true"] for r in metrics_rows], dtype=float),
                np.array([r["ci95_low"] for r in metrics_rows], dtype=float),
                np.array([r["ci95_high"] for r in metrics_rows], dtype=float),
            ),
        },
    ]

    plot_calibration_curve(calibration_rows=calibration_rows, export_dir=cfg.export_dir)
    plot_error_comparison(metrics_rows=metrics_rows, export_dir=cfg.export_dir)
    plot_shift_abstention(metrics_rows=metrics_rows, export_dir=cfg.export_dir)

    _write_summary(
        export_dir=cfg.export_dir,
        cfg=cfg,
        metrics_rows=metrics_rows,
        sample_rows=sample_rows,
        calibration_rows=calibration_rows,
    )


def parse_config(args: argparse.Namespace) -> RunConfig:
    g_grid = parse_g_grid(args.g_grid)
    export_dir = Path(args.export_dir)
    rounds = int(args.rounds)
    if rounds <= 0:
        rounds = max(4, len(g_grid) * 2)

    return RunConfig(
        seed=int(args.seed),
        shots=int(args.shots),
        backend=str(args.backend),
        ibm_backend_name=str(args.ibm_backend_name),
        noise=str(args.noise),
        export_dir=export_dir,
        mode=str(args.mode),
        g_grid=g_grid,
        policy=str(args.policy),
        dry_run=bool(args.dry_run),
        epsilon=float(args.epsilon),
        n_train=int(args.n_train),
        n_cal=int(args.n_cal),
        n_test=int(args.n_test),
        rounds=rounds,
        batch_shots=int(args.batch_shots),
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = parse_config(args)

    seed_all(cfg.seed)

    if cfg.mode in {"sweep", "train", "eval"}:
        run_sweep(cfg)
        return

    if cfg.mode == "hardware_run":
        if cfg.backend != "ibm":
            raise ValueError("--mode hardware_run requires --backend ibm")
        run_hardware(cfg)
        return

    raise ValueError(f"Unsupported mode: {cfg.mode}")


if __name__ == "__main__":
    main()
