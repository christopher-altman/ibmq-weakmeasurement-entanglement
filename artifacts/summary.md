# Run Summary

- Timestamp (UTC): 2026-02-15T16:05:47.944707+00:00
- Mode: `sweep`
- Backend: `sim`
- Noise: `ideal`
- Seed: `11`

## Key Metrics
- `adaptive_ig`: MAE(C)=0.0108, RMSE(C)=0.0146, MAE(N)=0.0020
- `adaptive_ig_scaled`: MAE(C)=0.0108, RMSE(C)=0.0146, MAE(N)=0.0020
- `fixed_nn_conformal`: MAE(C)=0.1378, RMSE(C)=0.1406, MAE(N)=0.0572
- `fixed_nn_naive`: MAE(C)=0.1378, RMSE(C)=0.1406, MAE(N)=0.0572
- `fixed_nn_scaled_conformal`: MAE(C)=0.1378, RMSE(C)=0.1406, MAE(N)=0.0572
- `fixed_particle`: MAE(C)=0.1086, RMSE(C)=0.1513, MAE(N)=0.0392
- `fixed_particle_scaled`: MAE(C)=0.1086, RMSE(C)=0.1513, MAE(N)=0.0392
- `tomography_baseline`: MAE(C)=0.0611, RMSE(C)=0.0663, MAE(N)=0.0102

## Claim Checks
- H1 calibration (90% target, method `adaptive_ig`): empirical=1.000 -> PASS
- H2 sample efficiency (median shots fixed=112.0, adaptive=100.0): PASS
- H3 hardware robustness: hardware rows unavailable in this run -> FAIL/PENDING

## Reproducibility Checklist
- Fixed seeds for random/NumPy/Torch and deterministic simulation sampling.
- Environment dump recorded below.
- Commands:
  - `python -m src.main --mode sweep --backend sim --noise ideal --policy adaptive --g_grid 0.00,0.10,0.20,0.35,0.50,0.70,0.90 --shots 2000 --seed 7 --export_dir artifacts`
  - `python -m src.main --mode hardware_run --backend ibm --ibm_backend_name ibm_kyoto --policy adaptive --g_grid 0.10,0.20,0.35,0.50 --shots 4000 --seed 7 --export_dir artifacts`
  - `python -m pytest`

## Environment
- Python: `3.14.2`
- numpy: `1.26.4`
- scipy: `1.17.0`
- matplotlib: `3.10.7`
- torch: `missing`
- pandas: `missing`
- qiskit: `missing`
- qiskit_aer: `missing`
- qiskit_ibm_runtime: `missing`
