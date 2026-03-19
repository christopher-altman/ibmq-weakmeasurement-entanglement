from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
HISTORY_ROOT = ROOT / ".internal" / "artifact_history"


def _artifact_relative_path(path: Path) -> Path:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT)
    except ValueError:
        return Path(resolved.name)


def _history_dir_for(path: Path) -> Path:
    rel = _artifact_relative_path(path)
    return HISTORY_ROOT / rel.parent


def _split_name(path: Path) -> tuple[str, str]:
    suffix = "".join(path.suffixes)
    if suffix:
        stem = path.name[: -len(suffix)]
    else:
        stem = path.name
    return stem, suffix


def _timestamp_label(dt: datetime | None = None) -> str:
    current = dt or datetime.now(timezone.utc)
    return current.strftime("%Y%m%dT%H%M%S_%fZ")


def _history_path(path: Path, *, label: str, timestamp: str) -> Path:
    stem, suffix = _split_name(path)
    return _history_dir_for(path) / f"{stem}__{label}__{timestamp}{suffix}"


def prepare_versioned_artifact(path: Path) -> Path:
    if path.exists():
        history_dir = _history_dir_for(path)
        history_dir.mkdir(parents=True, exist_ok=True)
        replaced_at = _timestamp_label(datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc))
        archived = _history_path(path, label="replaced", timestamp=replaced_at)
        path.replace(archived)
    return path


def snapshot_artifact(path: Path) -> Path:
    history_dir = _history_dir_for(path)
    history_dir.mkdir(parents=True, exist_ok=True)
    snapshot = _history_path(path, label="snapshot", timestamp=_timestamp_label())
    shutil.copy2(path, snapshot)
    return snapshot


def write_text_versioned(path: Path, text: str, *, encoding: str = "utf-8") -> Path:
    prepare_versioned_artifact(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding=encoding)
    snapshot_artifact(path)
    return path


def write_json_versioned(path: Path, obj: Any) -> Path:
    prepare_versioned_artifact(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    snapshot_artifact(path)
    return path


def write_csv_versioned(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
    prepare_versioned_artifact(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    snapshot_artifact(path)
    return path


def save_figure_versioned(fig: Any, path: Path, **savefig_kwargs: Any) -> Path:
    prepare_versioned_artifact(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, **savefig_kwargs)
    snapshot_artifact(path)
    return path
