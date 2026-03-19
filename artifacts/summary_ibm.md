# IBM Hardware Experiment Summary

- Timestamp (UTC): 2026-03-19T15:20:58.739223+00:00
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
- `adaptive_ig`: MAE(C)=0.0991, RMSE(C)=0.1096, Cov90=1.000, Cov95=1.000
- `fixed_particle`: MAE(C)=0.0991, RMSE(C)=0.1096, Cov90=1.000, Cov95=1.000

## Sample-Efficiency Proxy
- Median shots to |C_hat - C| <= 0.05: adaptive=8192.0, fixed=8192.0, fixed/adaptive ratio=1.00

## Output Files
- `artifacts/metrics_ibm.csv`
- `artifacts/fig_error_vs_shots_ibm.png`
- `artifacts/fig_sim_vs_ibm_gap.png`
- `artifacts/run_manifest.json`
- `artifacts/claims_map.json`

## Provenance Note
- Raw IBM job IDs and per-circuit count caches are preserved in local provenance during artifact generation and are not part of the published repository snapshot.

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
