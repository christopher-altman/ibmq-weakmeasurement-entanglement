# IBM Hardware Experiment Summary

- Timestamp (UTC): 2026-02-19T21:45:13.287544+00:00
- Status: `completed`
- Backend: `ibm_torino`
- Seeds: `[0, 1, 2]`
- Shots grid: `[1024, 4096, 8192]`
- g grid: `[0.1, 0.35, 0.5]`
- Policies: `['adaptive', 'fixed']`
- Completed configs: `18`
- Hardware states per config: `6`
- Total metric rows: `108`
- Total IBM jobs: `2052`

## Key Numbers
- `adaptive_ig`: MAE(C)=0.1325, RMSE(C)=0.1778, Cov90=0.833, Cov95=0.833
- `fixed_particle`: MAE(C)=0.1325, RMSE(C)=0.1778, Cov90=0.833, Cov95=0.833

## Sample-Efficiency Proxy
- Median shots to |C_hat - C| <= 0.05: adaptive=8192.0, fixed=8192.0, fixed/adaptive ratio=1.00

## Output Files
- `artifacts/metrics_ibm.csv`
- `artifacts/fig_error_vs_shots_ibm.png`
- `artifacts/fig_sim_vs_ibm_gap.png`
- `artifacts/ibm_cache/ibm_jobs.json`
- `artifacts/ibm_cache/counts_*.json`

## Environment
- Python: `3.11.14`
- numpy: `2.4.2`
- scipy: `1.17.0`
- matplotlib: `3.10.8`
- torch: `2.10.0`
- pandas: `3.0.0`
- qiskit: `2.3.0`
- qiskit_aer: `0.17.2`
- qiskit_ibm_runtime: `0.45.1`
