# Simulation Experiment Summary

- Timestamp (UTC): 2026-02-16T08:46:30.955707+00:00
- Backend: `sim`
- Seeds: `[0, 1, 2, 3, 4]`
- Shots grid: `[128, 256, 512, 1024, 2048, 4096, 8192]`
- g grid: `[0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9]`
- Sweep config: `n_train=40, n_cal=10, n_test=12, rounds=6`
- Total runs: `35`
- Total metric rows: `3360`

## Key numbers
- `adaptive_ig`: MAE(C)=0.0259, RMSE(C)=0.0493, Cov90=0.976
- `fixed_particle`: MAE(C)=0.0974, RMSE(C)=0.1489, Cov90=0.936
- `fixed_nn_conformal`: MAE(C)=0.2748, RMSE(C)=0.3769, Cov90=0.924
- `tomography_baseline`: MAE(C)=0.0168, RMSE(C)=0.0309, Cov90=0.371

## Sample-efficiency proxy
- Median shots to |Ĉ-C| <= 0.05: adaptive=128.0, fixed=512.0, fixed/adaptive ratio=4.00

## Output files
- `artifacts/metrics_sim.csv`
- `artifacts/fig_error_vs_shots_sim.png`
- `artifacts/fig_calibration_sim.png`
