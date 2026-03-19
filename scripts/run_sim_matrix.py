from __future__ import annotations

import os
from pathlib import Path

from src.helpers import RunConfig
from src.main import run_sweep


os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mplconfig')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp')

ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / 'artifacts' / 'raw' / 'sim_sweeps'
RAW_ROOT.mkdir(parents=True, exist_ok=True)

shots_grid = [128, 256, 512, 1024, 2048, 4096, 8192]
seeds = [0, 1, 2, 3, 4]
g_grid = [0.00, 0.10, 0.20, 0.35, 0.50, 0.70, 0.90]

n_total = len(seeds) * len(shots_grid)
idx = 0
for seed in seeds:
    for shots in shots_grid:
        idx += 1
        out_dir = RAW_ROOT / f'seed_{seed}_shots_{shots}'
        cfg = RunConfig(
            seed=seed,
            shots=shots,
            backend='sim',
            ibm_backend_name='ibm_kyoto',
            noise='ideal',
            export_dir=out_dir,
            mode='sweep',
            g_grid=g_grid,
            policy='adaptive',
            dry_run=False,
            epsilon=0.05,
            n_train=40,
            n_cal=10,
            n_test=12,
            rounds=6,
            batch_shots=0,
        )
        print(f'[{idx}/{n_total}] seed={seed} shots={shots} -> {out_dir}')
        run_sweep(cfg)

print('DONE')
