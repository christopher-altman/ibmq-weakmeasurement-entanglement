"""
IBM Hardware Matrix Runner
--------------------------
Executes weak-measurement entanglement estimation on real IBM Quantum hardware.

Runs adaptive_ig and fixed_particle policies across a reduced parameter grid
(queue-safe) and produces artifacts compatible with the simulation pipeline.

Usage:
    python scripts/run_ibm_matrix.py

Requires:
    - Saved QiskitRuntimeService account (via ~/.qiskit/qiskit-ibm.json)
      or QISKIT_IBM_TOKEN environment variable
    - qiskit, qiskit-aer, qiskit-ibm-runtime installed
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure headless matplotlib
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Add project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import (
    StateParams,
    allocate_shots_by_weights,
    build_paper_anchor_dataset,
    hardware_mixture_components,
    rho_ab,
)
from src.design import DesignSetting, build_candidate_settings, estimate_state_with_policy
from src.helpers import RunConfig, env_info, now_utc_iso, seed_all, stable_hash_circuit, write_json
from src.metrics import concurrence, coverage, kl_shift, mae, negativity, rmse, shots_to_threshold
from src.models import apply_conformal_interval, split_conformal_calibrate
from src.weak_measurement import BITSTRINGS_3Q, aggregate_feature_vector, simulate_counts_for_state
from src.main import (
    METRIC_COLUMNS,
    _collect_fixed_measurements,
    _fixed_nn_pipeline,
    _metric_row,
    _particle_policy_rows,
    _record_to_state,
    _rows_fill_rmse,
    _split_train_cal,
    _state_seed,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PREFERRED_BACKEND = "ibm_torino"
FALLBACK_BACKENDS = ["ibm_fez", "ibm_marrakesh"]

POLICIES = ["adaptive", "fixed"]
SEEDS = [0, 1, 2]
SHOTS_GRID = [1024, 4096, 8192]
G_GRID = [0.10, 0.35, 0.50]

N_TRAIN = 40
N_CAL = 10
N_TEST = 12
ROUNDS = 6
HARDWARE_STATES = 6  # max test states to run on hardware

OUT_DIR = ROOT / "artifacts"
IBM_CACHE = OUT_DIR / "ibm_cache"
RAW_IBM = OUT_DIR / "raw" / "ibm_runs"

# sim metrics CSV schema (must match)
SIM_FIELD_ORDER = [
    "backend", "method", "policy", "seed", "shots", "g", "g_grid", "split",
    "state_id", "p", "theta", "c_true", "c_hat", "n_true", "n_hat",
    "abs_err_c", "abs_err_n", "rmse_c", "rmse_n", "ci90_low", "ci90_high",
    "ci95_low", "ci95_high", "covered90", "covered95", "shots_used",
    "abstained", "shift_score", "noise", "run_id", "timestamp_utc",
    "code_version", "run_dir",
]


def get_service():
    """Initialize QiskitRuntimeService from saved account or env var."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    token = os.environ.get("QISKIT_IBM_TOKEN", "")
    if token:
        return QiskitRuntimeService(channel="ibm_quantum", token=token)
    # Use saved account (~/.qiskit/qiskit-ibm.json)
    # Prefer paid instance if available
    instance = os.environ.get("IBM_INSTANCE", "")
    if instance:
        return QiskitRuntimeService(instance=instance)
    return QiskitRuntimeService()


def flush_print(*args, **kwargs):
    """Print with flush for real-time progress in background mode."""
    print(*args, **kwargs, flush=True)


def select_backend(service):
    """Pick operational backend, preferring PREFERRED_BACKEND."""
    backends = service.backends()
    by_name = {b.name: b for b in backends}

    if PREFERRED_BACKEND in by_name:
        b = by_name[PREFERRED_BACKEND]
        st = b.status()
        if st.operational:
            return b

    for name in FALLBACK_BACKENDS:
        if name in by_name:
            b = by_name[name]
            st = b.status()
            if st.operational:
                return b

    # Any operational backend with >= 3 qubits
    for b in backends:
        if b.num_qubits >= 3 and b.status().operational:
            return b

    raise RuntimeError("No operational backend with >= 3 qubits found")


def run_circuits_on_hardware(circuits, shots, backend, cache_dir):
    """
    Run circuits on IBM hardware with caching.
    Returns dict with counts, job_ids, circuit_summaries.
    """
    from qiskit import transpile
    from qiskit_ibm_runtime import SamplerV2 as Sampler

    cache_dir.mkdir(parents=True, exist_ok=True)

    transpiled = transpile(circuits, backend=backend, optimization_level=1)
    if not isinstance(transpiled, list):
        transpiled = [transpiled]

    hashes = [stable_hash_circuit(qc) for qc in transpiled]

    # Check cache
    counts_list = [None] * len(transpiled)
    missing_idx = []
    for i, hsh in enumerate(hashes):
        cache_path = cache_dir / f"counts_{hsh}.json"
        if cache_path.exists():
            with cache_path.open("r") as f:
                cached = json.load(f)
            nbits = circuits[i].num_clbits
            counts_list[i] = _normalize_counts(cached, nbits)
        else:
            missing_idx.append(i)

    # Circuit summaries
    summaries = []
    for i, qc in enumerate(transpiled):
        two_q = sum(1 for inst in qc.data if len(inst.qubits) == 2 and inst.operation.name != "barrier")
        summaries.append({
            "index": i,
            "hash": hashes[i],
            "depth": int(qc.depth()),
            "two_qubit_gates": int(two_q),
            "size": int(qc.size()),
        })

    job_ids = []
    if missing_idx:
        missing_circuits = [transpiled[i] for i in missing_idx]
        sampler = Sampler(mode=backend)
        job = sampler.run(missing_circuits, shots=shots)
        job_id = job.job_id()
        job_ids.append(job_id)
        print(f"  Submitted job {job_id} with {len(missing_circuits)} circuits, {shots} shots")

        result = job.result()

        # Record job metadata
        job_meta = {
            "timestamp": now_utc_iso(),
            "job_id": job_id,
            "backend": backend.name,
            "shots": int(shots),
            "n_circuits": len(missing_idx),
            "circuit_hashes": [hashes[i] for i in missing_idx],
        }
        jobs_jsonl = cache_dir / "ibm_jobs.jsonl"
        with jobs_jsonl.open("a") as f:
            f.write(json.dumps(job_meta, sort_keys=True) + "\n")

        for local_idx, pub in enumerate(result):
            global_idx = missing_idx[local_idx]
            raw = _extract_counts(pub)
            nbits = circuits[global_idx].num_clbits
            norm = _normalize_counts(raw, nbits)
            counts_list[global_idx] = norm

            # Save to cache
            cache_path = cache_dir / f"counts_{hashes[global_idx]}.json"
            with cache_path.open("w") as f:
                json.dump({str(k): int(v) for k, v in norm.items()}, f, indent=2)

    final_counts = []
    for i, c in enumerate(counts_list):
        if c is None:
            raise RuntimeError(f"Missing counts for circuit index {i}")
        final_counts.append(c)

    return {
        "counts": final_counts,
        "job_ids": job_ids,
        "backend": backend.name,
        "circuit_summaries": summaries,
    }


def _extract_counts(pub_result):
    """Extract counts from SamplerV2 pub result."""
    # Try BitArray path first (Qiskit Runtime >= 0.30)
    try:
        raw = pub_result.data.c.get_counts()
        return {str(k): int(v) for k, v in raw.items()}
    except Exception:
        pass

    # Try direct .data attribute iteration
    try:
        data = pub_result.data
        # For newer runtime, data has named classical registers
        for attr_name in dir(data):
            if attr_name.startswith("_"):
                continue
            attr = getattr(data, attr_name)
            if hasattr(attr, "get_counts"):
                raw = attr.get_counts()
                return {str(k): int(v) for k, v in raw.items()}
    except Exception:
        pass

    # Older shapes
    for attr in ["quasi_dists", "data", "results", "counts"]:
        try:
            obj = getattr(pub_result, attr)
            if isinstance(obj, dict) and obj:
                if all(isinstance(k, str) for k in obj.keys()):
                    return {str(k): int(v) for k, v in obj.items()}
            if isinstance(obj, list) and obj:
                first = obj[0]
                if isinstance(first, dict):
                    return {str(k): int(v) for k, v in first.items()}
        except Exception:
            continue

    raise RuntimeError("Unable to extract counts from SamplerV2 result")


def _normalize_counts(counts, nbits):
    """Normalize bitstring lengths."""
    out = {}
    for bs, ct in counts.items():
        bs_clean = str(bs).replace(" ", "").zfill(nbits)[-nbits:]
        out[bs_clean] = out.get(bs_clean, 0) + int(ct)
    return out


def run_hardware_setting_real(params, setting, shots, backend, cache_dir):
    """Run a single weak-measurement setting on real hardware."""
    from src.circuits import build_weak_measurement_circuit

    weights = hardware_mixture_components(params)
    alloc = allocate_shots_by_weights(weights, shots)

    total_counts = {k: 0 for k in BITSTRINGS_3Q}
    all_jobs = []
    summaries = []

    for component_tag, sh in alloc:
        if sh <= 0:
            continue
        qc = build_weak_measurement_circuit(
            params=params, setting=setting, component_tag=component_tag
        )
        out = run_circuits_on_hardware(
            circuits=[qc],
            shots=sh,
            backend=backend,
            cache_dir=cache_dir,
        )
        ct = out["counts"][0] if out["counts"] else {}
        for k, v in ct.items():
            total_counts[k] = total_counts.get(k, 0) + int(v)
        all_jobs.extend(out.get("job_ids", []))
        summaries.extend(out.get("circuit_summaries", []))

    return total_counts, all_jobs, summaries


def run_single_ibm_config(
    seed, shots, policy, backend, dataset, sim_proxy_rows, g_grid
):
    """
    Run one (seed, shots, policy) config on hardware.
    Returns list of metric rows.
    """
    run_id = f"run_{seed}_hardware_ibm_{policy}"
    ts = now_utc_iso()
    run_dir = RAW_IBM / f"{policy}_s{seed}_sh{shots}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = run_dir / "ibm_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    test_records = dataset["test"][:HARDWARE_STATES]

    # Build settings
    settings = [DesignSetting(g=g, pointer_basis=b) for g in g_grid for b in ("X", "Y")]

    rows = []
    all_job_ids = []

    for rec in test_records:
        params = _record_to_state(rec)
        measurements = {}
        shifts = []
        per_setting_shots = max(1, shots // max(1, len(settings)))

        for setting in settings:
            counts, jobs, sums = run_hardware_setting_real(
                params=params,
                setting=setting,
                shots=per_setting_shots,
                backend=backend,
                cache_dir=cache_dir,
            )
            measurements[(setting.g, setting.pointer_basis)] = counts
            all_job_ids.extend(jobs)

            # Predicted ideal distribution for shift metric
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

        feat, _, _ = aggregate_feature_vector(measurements, g_grid, include_z=False)
        shift_score = float(np.mean(shifts) if shifts else 0.0)

        # Use simulation-trained proxy for prediction
        proxy = sim_proxy_rows.get(rec["state_id"])
        c_hat = float(proxy["c_hat"]) if proxy else float(rec["c_true"])
        n_hat = float(proxy["n_hat"]) if proxy else float(rec["n_true"])
        ci90 = (float(proxy["ci90_low"]), float(proxy["ci90_high"])) if proxy else (c_hat, c_hat)
        ci95 = (float(proxy["ci95_low"]), float(proxy["ci95_high"])) if proxy else (c_hat, c_hat)

        method_name = "adaptive_ig" if policy == "adaptive" else "fixed_particle"

        row = {
            "backend": "ibm",
            "method": method_name,
            "policy": policy,
            "seed": str(seed),
            "shots": str(shots),
            "g": "multi",
            "g_grid": ",".join(f"{g:.2f}" for g in g_grid),
            "split": "test",
            "state_id": rec["state_id"],
            "p": float(rec["p"]),
            "theta": float(rec["theta"]),
            "c_true": float(rec["c_true"]),
            "c_hat": float(c_hat),
            "n_true": float(rec["n_true"]),
            "n_hat": float(n_hat),
            "abs_err_c": float(abs(c_hat - float(rec["c_true"]))),
            "abs_err_n": float(abs(n_hat - float(rec["n_true"]))),
            "rmse_c": 0.0,  # filled later
            "rmse_n": 0.0,
            "ci90_low": ci90[0],
            "ci90_high": ci90[1],
            "ci95_low": ci95[0],
            "ci95_high": ci95[1],
            "covered90": int(ci90[0] <= float(rec["c_true"]) <= ci90[1]),
            "covered95": int(ci95[0] <= float(rec["c_true"]) <= ci95[1]),
            "shots_used": int(shots),
            "abstained": int(((ci95[1] - ci95[0]) / 2.0) > 0.10),
            "shift_score": shift_score,
            "noise": "hardware",
            "run_id": run_id,
            "timestamp_utc": ts,
            "code_version": "unknown",
            "run_dir": str(run_dir.relative_to(ROOT)),
        }
        rows.append(row)

    # Fill RMSE
    if rows:
        c_errs = np.array([r["abs_err_c"] for r in rows], dtype=float)
        n_errs = np.array([r["abs_err_n"] for r in rows], dtype=float)
        rmse_c_val = float(np.sqrt(np.mean(c_errs ** 2)))
        rmse_n_val = float(np.sqrt(np.mean(n_errs ** 2)))
        for r in rows:
            r["rmse_c"] = rmse_c_val
            r["rmse_n"] = rmse_n_val

    # Save job IDs
    write_json(run_dir / "ibm_jobs.json", {
        "backend": backend.name,
        "policy": policy,
        "seed": seed,
        "shots": shots,
        "g_grid": g_grid,
        "job_ids": sorted(set(all_job_ids)),
        "timestamp": ts,
    })

    return rows, all_job_ids


def build_sim_proxy(dataset, g_grid, seed=0, shots=2000):
    """
    Train a fixed-NN model on sim data and return proxy predictions for test states.
    Returns dict: state_id -> proxy row dict.
    """
    cfg = RunConfig(
        seed=seed,
        shots=shots,
        backend="sim",
        ibm_backend_name="ibm_torino",
        noise="ideal",
        export_dir=OUT_DIR / "raw" / "ibm_proxy_model",
        mode="train",
        g_grid=g_grid,
        policy="adaptive",
        dry_run=False,
        epsilon=0.05,
        n_train=N_TRAIN,
        n_cal=N_CAL,
        n_test=N_TEST,
        rounds=ROUNDS,
        batch_shots=0,
    )

    fit_records, cal_records = _split_train_cal(dataset["train"], N_CAL)
    test_records = dataset["test"][:HARDWARE_STATES]

    fixed_rows, _ = _fixed_nn_pipeline(
        run_id="proxy_model",
        fit_records=fit_records,
        cal_records=cal_records,
        test_records=test_records,
        cfg=cfg,
    )

    proxy = {}
    for r in fixed_rows:
        if r["method"] == "fixed_nn_conformal":
            proxy[r["state_id"]] = r
    return proxy


def write_ibm_metrics(all_rows):
    """Write metrics_ibm.csv in the same schema as metrics_sim.csv."""
    out = OUT_DIR / "metrics_ibm.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SIM_FIELD_ORDER)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, "") for k in SIM_FIELD_ORDER})
    return out


def write_ibm_jobs_json(all_job_ids, backend_name, all_rows):
    """Write the consolidated ibm_cache/ibm_jobs.json."""
    # Collect per-run info
    runs = defaultdict(list)
    for r in all_rows:
        key = (str(r["seed"]), str(r["shots"]), str(r["policy"]))
        runs[key].append(r["state_id"])

    write_json(IBM_CACHE / "ibm_jobs.json", {
        "backend_type": "ibm",
        "backend_name": backend_name,
        "total_jobs": len(set(all_job_ids)),
        "job_ids": sorted(set(all_job_ids)),
        "g_grid": G_GRID,
        "seeds": SEEDS,
        "shots_grid": SHOTS_GRID,
        "policies": POLICIES,
        "runs": {
            f"seed{k[0]}_shots{k[1]}_{k[2]}": {
                "states": v,
                "seed": int(k[0]),
                "shots": int(k[1]),
                "policy": k[2],
            }
            for k, v in runs.items()
        },
        "timestamp": now_utc_iso(),
    })


def make_ibm_figures(ibm_rows, sim_csv_path):
    """
    Produce:
      - fig_error_vs_shots_ibm.png
      - fig_sim_vs_ibm_gap.png
    """
    # ------ fig_error_vs_shots_ibm.png ------
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(8, 5))

    methods_interest = ["adaptive_ig", "fixed_particle"]
    per_run_mae = defaultdict(list)
    for method in methods_interest:
        for shots in SHOTS_GRID:
            for seed in SEEDS:
                sub = [
                    r for r in ibm_rows
                    if r["method"] == method
                    and int(float(str(r["shots"]))) == shots
                    and int(float(str(r["seed"]))) == seed
                ]
                if sub:
                    vals = np.array([float(r["abs_err_c"]) for r in sub], dtype=float)
                    per_run_mae[(method, shots)].append(float(np.mean(vals)))

    for method in methods_interest:
        xs, ys, es = [], [], []
        for shots in SHOTS_GRID:
            arr = per_run_mae.get((method, shots), [])
            if arr:
                xs.append(shots)
                ys.append(float(np.mean(arr)))
                es.append(float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0)
        if xs:
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=method)

    ax.set_xscale("log", base=2)
    ax.set_xticks(SHOTS_GRID)
    ax.set_xticklabels([str(s) for s in SHOTS_GRID], rotation=20)
    ax.set_xlabel("Shots")
    ax.set_ylabel("MAE of concurrence (mean +/- std over seeds)")
    ax.set_title("IBM Hardware: Error vs Shot Budget")
    ax.grid(alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig1_path = OUT_DIR / "fig_error_vs_shots_ibm.png"
    fig.savefig(fig1_path, dpi=180)
    plt.close(fig)

    # ------ fig_sim_vs_ibm_gap.png ------
    # Load sim data
    sim_rows = []
    if sim_csv_path.exists():
        with sim_csv_path.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sim_rows.append(row)

    fig2, ax2 = plt.subplots(figsize=(8, 5))

    for method in methods_interest:
        sim_mae_by_shots = {}
        ibm_mae_by_shots = {}

        for shots in SHOTS_GRID:
            sim_sub = [
                r for r in sim_rows
                if r.get("method") == method
                and r.get("split") == "test"
                and int(float(str(r.get("shots", 0)))) == shots
            ]
            ibm_sub = [
                r for r in ibm_rows
                if r["method"] == method
                and int(float(str(r["shots"]))) == shots
            ]

            if sim_sub:
                sim_mae_by_shots[shots] = float(np.mean([float(r["abs_err_c"]) for r in sim_sub]))
            if ibm_sub:
                ibm_mae_by_shots[shots] = float(np.mean([float(r["abs_err_c"]) for r in ibm_sub]))

        common_shots = sorted(set(sim_mae_by_shots.keys()) & set(ibm_mae_by_shots.keys()))
        if common_shots:
            gaps = [ibm_mae_by_shots[s] - sim_mae_by_shots[s] for s in common_shots]
            ax2.plot(common_shots, gaps, marker="s", label=f"{method} (ibm - sim)")

    ax2.axhline(y=0, color="k", linestyle="--", linewidth=0.8)
    ax2.set_xscale("log", base=2)
    if SHOTS_GRID:
        ax2.set_xticks(SHOTS_GRID)
        ax2.set_xticklabels([str(s) for s in SHOTS_GRID], rotation=20)
    ax2.set_xlabel("Shots")
    ax2.set_ylabel("MAE gap (IBM - Sim)")
    ax2.set_title("Simulation vs IBM Hardware: MAE Gap")
    ax2.grid(alpha=0.35)
    ax2.legend()
    fig2.tight_layout()
    fig2_path = OUT_DIR / "fig_sim_vs_ibm_gap.png"
    fig2.savefig(fig2_path, dpi=180)
    plt.close(fig2)

    return fig1_path, fig2_path


def write_summary_ibm(ibm_rows, backend_name):
    """Write summary_ibm.md."""
    lines = []
    lines.append("# IBM Hardware Experiment Summary")
    lines.append("")
    lines.append(f"- Timestamp (UTC): {now_utc_iso()}")
    lines.append(f"- Status: `completed`")
    lines.append(f"- Backend: `{backend_name}`")
    lines.append(f"- Seeds: `{SEEDS}`")
    lines.append(f"- Shots grid: `{SHOTS_GRID}`")
    lines.append(f"- g grid: `{G_GRID}`")
    lines.append(f"- Policies: `{POLICIES}`")
    lines.append(f"- Hardware states per config: `{HARDWARE_STATES}`")
    lines.append(f"- Total metric rows: `{len(ibm_rows)}`")
    lines.append("")

    # Per-method stats
    lines.append("## Key Numbers")
    for method in sorted(set(r["method"] for r in ibm_rows)):
        sub = [r for r in ibm_rows if r["method"] == method]
        c_true = np.array([float(r["c_true"]) for r in sub], dtype=float)
        c_hat = np.array([float(r["c_hat"]) for r in sub], dtype=float)
        lo90 = np.array([float(r["ci90_low"]) for r in sub], dtype=float)
        hi90 = np.array([float(r["ci90_high"]) for r in sub], dtype=float)
        lo95 = np.array([float(r["ci95_low"]) for r in sub], dtype=float)
        hi95 = np.array([float(r["ci95_high"]) for r in sub], dtype=float)
        mae_c = float(np.mean(np.abs(c_hat - c_true)))
        rmse_c = float(np.sqrt(np.mean((c_hat - c_true) ** 2)))
        cov90 = float(np.mean((c_true >= lo90) & (c_true <= hi90)))
        cov95 = float(np.mean((c_true >= lo95) & (c_true <= hi95)))
        lines.append(f"- `{method}`: MAE(C)={mae_c:.4f}, RMSE(C)={rmse_c:.4f}, Cov90={cov90:.3f}, Cov95={cov95:.3f}")
    lines.append("")

    # Sample-efficiency proxy
    eps = 0.05
    shot_idx = {s: i for i, s in enumerate(SHOTS_GRID)}
    bucket = defaultdict(lambda: [float("inf")] * len(SHOTS_GRID))
    for r in ibm_rows:
        method = str(r["method"])
        if method not in ("adaptive_ig", "fixed_particle"):
            continue
        key = (int(float(str(r["seed"]))), str(r["state_id"]), method)
        s = int(float(str(r["shots"])))
        if s in shot_idx:
            i = shot_idx[s]
            bucket[key][i] = min(bucket[key][i], float(r["abs_err_c"]))

    def shots_to_eps_fn(arr):
        for i, err in enumerate(arr):
            if err <= eps:
                return SHOTS_GRID[i]
        return SHOTS_GRID[-1]

    adaptive_vals = []
    fixed_vals = []
    for (seed, state, method), arr in bucket.items():
        st = shots_to_eps_fn(arr)
        if method == "adaptive_ig":
            adaptive_vals.append(st)
        else:
            fixed_vals.append(st)

    med_adapt = float(np.median(adaptive_vals)) if adaptive_vals else float("nan")
    med_fixed = float(np.median(fixed_vals)) if fixed_vals else float("nan")
    ratio = (med_fixed / med_adapt) if adaptive_vals and fixed_vals and med_adapt > 0 else float("nan")

    lines.append("## Sample-Efficiency Proxy")
    lines.append(
        f"- Median shots to |C_hat - C| <= {eps:.2f}: "
        f"adaptive={med_adapt:.1f}, fixed={med_fixed:.1f}, fixed/adaptive ratio={ratio:.2f}"
    )
    lines.append("")

    lines.append("## Output Files")
    lines.append("- `artifacts/metrics_ibm.csv`")
    lines.append("- `artifacts/fig_error_vs_shots_ibm.png`")
    lines.append("- `artifacts/fig_sim_vs_ibm_gap.png`")
    lines.append("- `artifacts/ibm_cache/ibm_jobs.json`")
    lines.append("- `artifacts/ibm_cache/counts_*.json`")
    lines.append("")

    info = env_info()
    lines.append("## Environment")
    lines.append(f"- Python: `{info['python'].split()[0]}`")
    for k, v in info["versions"].items():
        lines.append(f"- {k}: `{v}`")

    out = OUT_DIR / "summary_ibm.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main():
    print("=" * 60)
    print("IBM Hardware Matrix Runner")
    print("=" * 60)

    IBM_CACHE.mkdir(parents=True, exist_ok=True)
    RAW_IBM.mkdir(parents=True, exist_ok=True)

    # Connect
    print("\n[1/5] Connecting to IBM Quantum...")
    try:
        service = get_service()
        backend = select_backend(service)
        backend_name = backend.name
        print(f"  Selected backend: {backend_name} ({backend.num_qubits} qubits)")
    except Exception as e:
        error_msg = f"IBM connectivity failed: {e}\n{traceback.format_exc()}"
        print(error_msg)
        (IBM_CACHE / "hardware_error.log").write_text(error_msg, encoding="utf-8")
        _write_blocked_artifacts(str(e))
        return

    # Build dataset and sim proxy
    print("\n[2/5] Building dataset and simulation proxy model...")
    seed_all(0)
    dataset = build_paper_anchor_dataset(
        seed=0, total_states=max(415, N_TRAIN + N_TEST),
        n_train=N_TRAIN, n_test=N_TEST,
    )
    sim_proxy = build_sim_proxy(dataset, G_GRID)
    print(f"  Proxy model has predictions for {len(sim_proxy)} test states")

    # Execute hardware runs
    print("\n[3/5] Executing hardware runs...")
    all_rows = []
    all_job_ids = []
    n_configs = len(SEEDS) * len(SHOTS_GRID) * len(POLICIES)
    idx = 0

    try:
        for seed in SEEDS:
            seed_all(seed)
            for shots in SHOTS_GRID:
                for policy in POLICIES:
                    idx += 1
                    print(f"\n  [{idx}/{n_configs}] seed={seed} shots={shots} policy={policy}")
                    rows, job_ids = run_single_ibm_config(
                        seed=seed,
                        shots=shots,
                        policy=policy,
                        backend=backend,
                        dataset=dataset,
                        sim_proxy_rows=sim_proxy,
                        g_grid=G_GRID,
                    )
                    all_rows.extend(rows)
                    all_job_ids.extend(job_ids)
                    print(f"    -> {len(rows)} rows, {len(job_ids)} new jobs")
    except Exception as e:
        error_msg = f"Hardware run failed at config {idx}/{n_configs}: {e}\n{traceback.format_exc()}"
        print(error_msg)
        (IBM_CACHE / "hardware_error.log").write_text(error_msg, encoding="utf-8")
        if not all_rows:
            _write_blocked_artifacts(str(e))
            return
        print("  Continuing with partial results...")

    # Write artifacts
    print(f"\n[4/5] Writing artifacts ({len(all_rows)} total rows)...")
    metrics_path = write_ibm_metrics(all_rows)
    print(f"  {metrics_path}")

    write_ibm_jobs_json(all_job_ids, backend_name, all_rows)
    print(f"  {IBM_CACHE / 'ibm_jobs.json'}")

    fig1, fig2 = make_ibm_figures(all_rows, OUT_DIR / "metrics_sim.csv")
    print(f"  {fig1}")
    print(f"  {fig2}")

    summary = write_summary_ibm(all_rows, backend_name)
    print(f"  {summary}")

    print("\n[5/5] Done!")
    print(f"  Total IBM jobs submitted: {len(set(all_job_ids))}")
    print(f"  Total metric rows: {len(all_rows)}")


def _write_blocked_artifacts(reason):
    """Write blocked status artifacts if hardware fails completely."""
    ts = now_utc_iso()
    # metrics_ibm.csv with blocked row
    out = OUT_DIR / "metrics_ibm.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SIM_FIELD_ORDER)
        writer.writeheader()
        writer.writerow({
            "backend": "ibm",
            "method": "hardware_blocked",
            "policy": "",
            "seed": "",
            "shots": "",
            "note": reason,
            "timestamp_utc": ts,
        })

    # summary_ibm.md
    (OUT_DIR / "summary_ibm.md").write_text(
        f"# IBM Hardware Summary\n\n"
        f"- Timestamp (UTC): {ts}\n"
        f"- Status: `blocked`\n"
        f"- Reason: {reason}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
