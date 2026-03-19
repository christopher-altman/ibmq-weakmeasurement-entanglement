from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .artifact_history import save_figure_versioned


def _base_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (7.5, 4.8),
            "axes.grid": True,
            "grid.alpha": 0.3,
            "savefig.dpi": 160,
        }
    )


def plot_calibration_curve(calibration_rows: list[dict[str, Any]], export_dir: Path) -> Path:
    _base_style()
    fig, ax = plt.subplots()

    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in calibration_rows:
        grouped[str(row["method"])].append((float(row["nominal"]), float(row["empirical"])))

    for method, pts in grouped.items():
        pts = sorted(pts, key=lambda x: x[0])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=method)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, label="ideal")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Conformal Calibration")
    ax.legend(loc="lower right")

    path = export_dir / "fig_calibration.png"
    fig.tight_layout()
    save_figure_versioned(fig, path)
    plt.close(fig)
    return path


def plot_sample_efficiency(sample_rows: list[dict[str, Any]], export_dir: Path) -> Path:
    _base_style()
    fig, ax = plt.subplots()

    grouped: dict[str, list[float]] = defaultdict(list)
    for row in sample_rows:
        grouped[str(row["method"])].append(float(row["shots_to_eps"]))

    methods = sorted(grouped.keys())
    vals = [grouped[m] for m in methods]
    ax.boxplot(vals, tick_labels=methods, showmeans=True)
    ax.set_ylabel("Shots to reach |Ĉ-C| ≤ ε")
    ax.set_title("Sample Efficiency (Lower is Better)")

    path = export_dir / "fig_sample_efficiency.png"
    fig.tight_layout()
    save_figure_versioned(fig, path)
    plt.close(fig)
    return path


def plot_error_comparison(metrics_rows: list[dict[str, Any]], export_dir: Path) -> Path:
    _base_style()
    fig, ax = plt.subplots()

    methods = sorted(set(str(r["method"]) for r in metrics_rows if str(r.get("split", "")) == "test"))
    mae_vals = []
    rmse_vals = []
    for m in methods:
        subset = [r for r in metrics_rows if str(r["method"]) == m and str(r.get("split", "")) == "test"]
        if not subset:
            mae_vals.append(0.0)
            rmse_vals.append(0.0)
            continue
        errs = np.array([float(r["abs_err_c"]) for r in subset], dtype=float)
        mae_vals.append(float(np.mean(errs)))
        rmse_vals.append(float(np.sqrt(np.mean(errs**2))))

    x = np.arange(len(methods))
    w = 0.36
    ax.bar(x - w / 2, mae_vals, width=w, label="MAE(C)")
    ax.bar(x + w / 2, rmse_vals, width=w, label="RMSE(C)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("Error")
    ax.set_title("Concurrence Error Comparison")
    ax.legend()

    path = export_dir / "fig_error_comparison.png"
    fig.tight_layout()
    save_figure_versioned(fig, path)
    plt.close(fig)
    return path


def plot_shift_abstention(metrics_rows: list[dict[str, Any]], export_dir: Path) -> Path:
    _base_style()
    fig, ax = plt.subplots()

    shift = np.array([float(r.get("shift_score", 0.0)) for r in metrics_rows], dtype=float)
    abst = np.array([float(r.get("abstained", 0.0)) for r in metrics_rows], dtype=float)

    ax.scatter(shift, abst, alpha=0.7)
    ax.set_xlabel("Shift score (KL)")
    ax.set_ylabel("Abstained (0/1)")
    ax.set_title("Domain Shift vs Abstention")

    path = export_dir / "fig_shift_abstention.png"
    fig.tight_layout()
    save_figure_versioned(fig, path)
    plt.close(fig)
    return path
