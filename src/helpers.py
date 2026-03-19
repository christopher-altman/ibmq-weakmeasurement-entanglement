from __future__ import annotations

import csv
import hashlib
import importlib
import io
import json
import os
import platform
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class RunConfig:
    seed: int
    shots: int
    backend: str
    ibm_backend_name: str
    noise: str
    export_dir: Path
    mode: str
    g_grid: list[float]
    policy: str
    dry_run: bool
    epsilon: float
    n_train: int
    n_cal: int
    n_test: int
    rounds: int
    batch_shots: int


@dataclass(frozen=True)
class SweepResult:
    state_id: str
    method: str
    c_true: float
    c_hat: float
    n_true: float
    n_hat: float
    ci_low: float
    ci_high: float
    covered_90: int
    covered_95: int
    shots_used: int
    abstained: int
    shift_score: float


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    except Exception:
        pass


def ensure_dirs(export_dir: Path) -> dict[str, Path]:
    export_dir.mkdir(parents=True, exist_ok=True)
    ibm_cache = export_dir / "ibm_cache"
    ibm_cache.mkdir(parents=True, exist_ok=True)
    model_dir = export_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    return {
        "export": export_dir,
        "ibm_cache": ibm_cache,
        "model_dir": model_dir,
    }


def set_mpl_cache_if_needed(export_dir: Path) -> None:
    if "MPLCONFIGDIR" not in os.environ:
        mpl_dir = export_dir / ".mplconfig"
        mpl_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_dir)


def stable_hash_circuit(qc: Any) -> str:
    """
    Return stable SHA256 hash for a Qiskit circuit.
    Uses QPY serialization if available; falls back to qasm/str.
    """
    payload: bytes
    try:
        from qiskit import qpy

        buff = io.BytesIO()
        qpy.dump(qc, buff)
        payload = buff.getvalue()
    except Exception:
        try:
            payload = qc.qasm().encode("utf-8")
        except Exception:
            payload = str(qc).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def safe_div(num: float, den: float, eps: float = 1e-12) -> float:
    return float(num) / float(den + eps)


def safe_expectation_from_counts(
    counts: dict[str, int], bit_index: int, mask: dict[str, bool] | None = None, eps: float = 1e-12
) -> tuple[float, int]:
    """
    E[(-1)^bit] under optional mask over bitstrings.
    Returns (expectation, denominator_count).
    """
    total = 0
    signed = 0
    for bitstring, ct in counts.items():
        if mask is not None and not mask.get(bitstring, False):
            continue
        bit = int(bitstring[bit_index])
        total += int(ct)
        signed += int(ct) * (1 if bit == 0 else -1)
    if total == 0:
        return 0.0, 0
    return safe_div(signed, total, eps=eps), total


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def save_dataclass_json(path: Path, value: Any) -> None:
    write_json(path, asdict(value))


def parse_g_grid(raw: str) -> list[float]:
    values = [x.strip() for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("--g_grid cannot be empty.")
    g_vals = [float(v) for v in values]
    for g in g_vals:
        if g < 0 or g > (np.pi / 2 + 1e-9):
            raise ValueError(f"g value out of range [0, pi/2]: {g}")
    return g_vals


def env_info() -> dict[str, Any]:
    mods = [
        "numpy",
        "scipy",
        "matplotlib",
        "torch",
        "pandas",
        "qiskit",
        "qiskit_aer",
        "qiskit_ibm_runtime",
    ]
    versions: dict[str, str] = {}
    for mod_name in mods:
        try:
            mod = importlib.import_module(mod_name)
            versions[mod_name] = str(getattr(mod, "__version__", "unknown"))
        except Exception:
            versions[mod_name] = "missing"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "versions": versions,
    }


def to_serializable_records(items: Iterable[SweepResult]) -> list[dict[str, Any]]:
    return [asdict(x) for x in items]
