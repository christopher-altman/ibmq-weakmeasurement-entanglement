from __future__ import annotations

from pathlib import Path

from src.main import build_parser, main


def test_cli_parser_required_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(["--g_grid", "0.0,0.1", "--mode", "sweep"])
    assert args.g_grid == "0.0,0.1"
    assert args.mode == "sweep"


def test_small_sweep_smoke(tmp_path: Path) -> None:
    outdir = tmp_path / "artifacts"
    main(
        [
            "--mode",
            "sweep",
            "--backend",
            "sim",
            "--noise",
            "ideal",
            "--policy",
            "adaptive",
            "--g_grid",
            "0.00,0.20,0.50",
            "--shots",
            "240",
            "--seed",
            "11",
            "--n_train",
            "10",
            "--n_cal",
            "4",
            "--n_test",
            "4",
            "--rounds",
            "6",
            "--export_dir",
            str(outdir),
        ]
    )

    assert (outdir / "metrics.csv").exists()
    assert (outdir / "summary.md").exists()
    assert (outdir / "ibm_jobs.json").exists()
    assert (outdir / "fig_calibration.png").exists()
    assert (outdir / "fig_sample_efficiency.png").exists()
    assert (outdir / "fig_error_comparison.png").exists()
