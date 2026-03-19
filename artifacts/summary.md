# Run Summary

- Timestamp (UTC): 2026-03-19T15:17:49.243549+00:00
- Mode: `sweep`
- Backend: `sim`
- Noise: `ideal`
- Seed: `7`

## Key Metrics
- `adaptive_ig`: MAE(C)=0.0208, RMSE(C)=0.0324, MAE(N)=0.0100
- `adaptive_ig_scaled`: MAE(C)=0.0208, RMSE(C)=0.0324, MAE(N)=0.0100
- `fixed_nn_conformal`: MAE(C)=0.0671, RMSE(C)=0.0835, MAE(N)=0.0404
- `fixed_nn_naive`: MAE(C)=0.0671, RMSE(C)=0.0835, MAE(N)=0.0404
- `fixed_nn_scaled_conformal`: MAE(C)=0.0671, RMSE(C)=0.0835, MAE(N)=0.0404
- `fixed_particle`: MAE(C)=0.0383, RMSE(C)=0.0693, MAE(N)=0.0184
- `fixed_particle_scaled`: MAE(C)=0.0383, RMSE(C)=0.0693, MAE(N)=0.0184
- `tomography_baseline`: MAE(C)=0.0100, RMSE(C)=0.0168, MAE(N)=0.0038

## Claim Checks
- H1 calibration (90% target, method `adaptive_ig`): empirical=0.879 -> PASS
- H2 sample efficiency (median shots fixed=775.0, adaptive=250.0): PASS
- H3 hardware robustness: hardware rows unavailable in this run -> FAIL/PENDING

## Reproducibility Checklist
- Fixed seeds for random/NumPy/Torch and deterministic simulation sampling.
- Environment dump recorded below.
- Commands:
  - `python -m src.main --mode sweep --backend sim --noise ideal --policy adaptive --g_grid 0.00,0.10,0.20,0.35,0.50,0.70,0.90 --shots 2000 --seed 7 --export_dir artifacts`
  - `python -m src.main --mode hardware_run --backend ibm --ibm_backend_name ibm_kyoto --policy adaptive --g_grid 0.10,0.20,0.35,0.50 --shots 4000 --seed 7 --export_dir artifacts`
  - `python -m pytest`

## Environment
- Python: `3.10.11`
- numpy: `1.26.4`
- scipy: `1.14.1`
- matplotlib: `3.10.7`
- torch: `2.9.1`
- pandas: `1.5.3`
- qiskit: `2.3.1`
- qiskit_aer: `0.17.2`
- qiskit_ibm_runtime: `0.45.1`
