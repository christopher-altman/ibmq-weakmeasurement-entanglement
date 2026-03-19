from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / 'artifacts' / 'raw' / 'sim_sweeps'
OUT_DIR = ROOT / 'artifacts'

SHOTS_GRID = [128, 256, 512, 1024, 2048, 4096, 8192]
SEEDS = [0, 1, 2, 3, 4]
G_GRID = [0.00, 0.10, 0.20, 0.35, 0.50, 0.70, 0.90]


FIELD_ORDER = [
    'backend','method','policy','seed','shots','g','g_grid','split','state_id','p','theta','c_true','c_hat','n_true','n_hat',
    'abs_err_c','abs_err_n','rmse_c','rmse_n','ci90_low','ci90_high','ci95_low','ci95_high','covered90','covered95','shots_used',
    'abstained','shift_score','noise','run_id','timestamp_utc','code_version','run_dir'
]


def load_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        for shots in SHOTS_GRID:
            run_dir = RAW_ROOT / f'seed_{seed}_shots_{shots}'
            metrics_path = run_dir / 'metrics.csv'
            if not metrics_path.exists():
                raise FileNotFoundError(metrics_path)
            ts = datetime.fromtimestamp(metrics_path.stat().st_mtime, tz=timezone.utc).isoformat()
            with metrics_path.open('r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    out = dict(row)
                    out['backend'] = 'sim'
                    out['seed'] = str(seed)
                    out['shots'] = str(shots)
                    out['g'] = 'multi'
                    out['g_grid'] = ','.join(f'{g:.2f}' for g in G_GRID)
                    out['timestamp_utc'] = ts
                    out['code_version'] = 'unknown'
                    out['run_dir'] = str(run_dir.relative_to(ROOT))
                    rows.append(out)
    return rows


def write_metrics(rows: list[dict[str, object]]) -> Path:
    out = OUT_DIR / 'metrics_sim.csv'
    with out.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELD_ORDER)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in FIELD_ORDER})
    return out


def make_figures(rows: list[dict[str, object]]) -> tuple[Path, Path]:
    rows_test = [r for r in rows if r.get('split') == 'test']

    methods_interest = ['adaptive_ig', 'fixed_particle', 'fixed_nn_conformal', 'tomography_baseline']
    per_run_mae = defaultdict(list)

    for method in methods_interest:
        for shots in SHOTS_GRID:
            for seed in SEEDS:
                sub = [
                    r for r in rows_test
                    if r['method'] == method and int(float(str(r['shots']))) == shots and int(float(str(r['seed']))) == seed
                ]
                if not sub:
                    continue
                vals = np.array([float(str(r['abs_err_c'])) for r in sub], dtype=float)
                per_run_mae[(method, shots)].append(float(np.mean(vals)))

    plt.style.use('default')
    plt.figure(figsize=(8.2, 5.1))
    for method in methods_interest:
        xs, ys, es = [], [], []
        for shots in SHOTS_GRID:
            arr = per_run_mae.get((method, shots), [])
            if not arr:
                continue
            xs.append(shots)
            ys.append(float(np.mean(arr)))
            es.append(float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0)
        if xs:
            plt.errorbar(xs, ys, yerr=es, marker='o', capsize=3, label=method)

    plt.xscale('log', base=2)
    plt.xticks(SHOTS_GRID, [str(s) for s in SHOTS_GRID], rotation=20)
    plt.xlabel('Shots')
    plt.ylabel('MAE of concurrence (mean ± std over seeds)')
    plt.title('Simulation: Error vs Shot Budget')
    plt.grid(alpha=0.35)
    plt.legend()
    plt.tight_layout()
    fig1 = OUT_DIR / 'fig_error_vs_shots_sim.png'
    plt.savefig(fig1, dpi=180)
    plt.close()

    cal_methods = ['adaptive_ig', 'fixed_particle', 'fixed_nn_conformal', 'fixed_nn_naive']
    plt.figure(figsize=(6.8, 5.0))
    for method in cal_methods:
        sub = [r for r in rows_test if r['method'] == method]
        if not sub:
            continue
        y = np.array([float(str(r['c_true'])) for r in sub], dtype=float)
        lo90 = np.array([float(str(r['ci90_low'])) for r in sub], dtype=float)
        hi90 = np.array([float(str(r['ci90_high'])) for r in sub], dtype=float)
        lo95 = np.array([float(str(r['ci95_low'])) for r in sub], dtype=float)
        hi95 = np.array([float(str(r['ci95_high'])) for r in sub], dtype=float)
        cov90 = float(np.mean((y >= lo90) & (y <= hi90)))
        cov95 = float(np.mean((y >= lo95) & (y <= hi95)))
        plt.plot([0.90, 0.95], [cov90, cov95], marker='o', label=method)

    plt.plot([0.88, 0.97], [0.88, 0.97], 'k--', linewidth=1.0, label='ideal')
    plt.xlim(0.88, 0.97)
    plt.ylim(0.88, 0.97)
    plt.xlabel('Nominal coverage')
    plt.ylabel('Empirical coverage')
    plt.title('Simulation calibration')
    plt.grid(alpha=0.35)
    plt.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    fig2 = OUT_DIR / 'fig_calibration_sim.png'
    plt.savefig(fig2, dpi=180)
    plt.close()

    return fig1, fig2


def write_summary(rows: list[dict[str, object]]) -> Path:
    rows_test = [r for r in rows if r.get('split') == 'test']
    method_stats = {}
    for method in sorted(set(r['method'] for r in rows_test)):
        sub = [r for r in rows_test if r['method'] == method]
        if not sub:
            continue
        c_true = np.array([float(str(r['c_true'])) for r in sub], dtype=float)
        c_hat = np.array([float(str(r['c_hat'])) for r in sub], dtype=float)
        lo90 = np.array([float(str(r['ci90_low'])) for r in sub], dtype=float)
        hi90 = np.array([float(str(r['ci90_high'])) for r in sub], dtype=float)
        method_stats[method] = {
            'mae_c': float(np.mean(np.abs(c_hat - c_true))),
            'rmse_c': float(np.sqrt(np.mean((c_hat - c_true) ** 2))),
            'cov90': float(np.mean((c_true >= lo90) & (c_true <= hi90))),
        }

    eps = 0.05
    shot_idx = {s: i for i, s in enumerate(SHOTS_GRID)}
    bucket = defaultdict(lambda: [float('inf')] * len(SHOTS_GRID))
    for r in rows_test:
        method = str(r['method'])
        if method not in ('adaptive_ig', 'fixed_particle'):
            continue
        key = (int(float(str(r['seed']))), str(r['state_id']), method)
        s = int(float(str(r['shots'])))
        i = shot_idx[s]
        bucket[key][i] = min(bucket[key][i], float(str(r['abs_err_c'])))

    def shots_to_eps(arr: list[float]) -> int:
        for i, err in enumerate(arr):
            if err <= eps:
                return SHOTS_GRID[i]
        return SHOTS_GRID[-1]

    adaptive_vals = []
    fixed_vals = []
    for (seed, state, method), arr in bucket.items():
        st = shots_to_eps(arr)
        if method == 'adaptive_ig':
            adaptive_vals.append(st)
        else:
            fixed_vals.append(st)

    med_adapt = float(np.median(adaptive_vals)) if adaptive_vals else float('nan')
    med_fixed = float(np.median(fixed_vals)) if fixed_vals else float('nan')
    ratio = (med_fixed / med_adapt) if adaptive_vals and fixed_vals and med_adapt > 0 else float('nan')

    out = OUT_DIR / 'summary_sim.md'
    lines = []
    lines.append('# Simulation Experiment Summary')
    lines.append('')
    lines.append(f'- Timestamp (UTC): {datetime.now(timezone.utc).isoformat()}')
    lines.append('- Backend: `sim`')
    lines.append(f'- Seeds: `{SEEDS}`')
    lines.append(f'- Shots grid: `{SHOTS_GRID}`')
    lines.append(f'- g grid: `{G_GRID}`')
    lines.append('- Sweep config: `n_train=40, n_cal=10, n_test=12, rounds=6`')
    lines.append(f'- Total runs: `{len(SEEDS) * len(SHOTS_GRID)}`')
    lines.append(f'- Total metric rows: `{len(rows)}`')
    lines.append('')
    lines.append('## Key numbers')
    for m in ['adaptive_ig', 'fixed_particle', 'fixed_nn_conformal', 'tomography_baseline']:
        if m in method_stats:
            s = method_stats[m]
            lines.append(f"- `{m}`: MAE(C)={s['mae_c']:.4f}, RMSE(C)={s['rmse_c']:.4f}, Cov90={s['cov90']:.3f}")
    lines.append('')
    lines.append('## Sample-efficiency proxy')
    lines.append(f'- Median shots to |Ĉ-C| <= {eps:.2f}: adaptive={med_adapt:.1f}, fixed={med_fixed:.1f}, fixed/adaptive ratio={ratio:.2f}')
    lines.append('')
    lines.append('## Output files')
    lines.append('- `artifacts/metrics_sim.csv`')
    lines.append('- `artifacts/fig_error_vs_shots_sim.png`')
    lines.append('- `artifacts/fig_calibration_sim.png`')
    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return out


def main() -> None:
    rows = load_rows()
    metrics = write_metrics(rows)
    fig1, fig2 = make_figures(rows)
    summary = write_summary(rows)
    print(metrics)
    print(fig1)
    print(fig2)
    print(summary)


if __name__ == '__main__':
    main()
