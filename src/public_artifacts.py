from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifact_history import write_json_versioned
from .helpers import load_json, now_utc_iso


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts"
INTERNAL_PROVENANCE_DIR = ROOT / ".internal" / "provenance"

PUBLIC_MANIFEST_PATH = OUT_DIR / "run_manifest.json"
PUBLIC_CLAIMS_PATH = OUT_DIR / "claims_map.json"

INTERNAL_RUN_MANIFEST_PATH = INTERNAL_PROVENANCE_DIR / "run_manifest.json"
INTERNAL_IBM_JOBS_PATH = INTERNAL_PROVENANCE_DIR / "ibm_jobs.json"


def update_public_manifest(section_name: str, payload: dict[str, Any]) -> Path:
    manifest = load_json(PUBLIC_MANIFEST_PATH, default={})
    if not isinstance(manifest, dict) or manifest.get("artifact_schema") != "public-v1":
        manifest = {}
    published = manifest.get("published_artifacts", {})
    if not isinstance(published, dict):
        published = {}
    published[section_name] = payload
    manifest["artifact_schema"] = "public-v1"
    manifest["last_updated_utc"] = now_utc_iso()
    manifest["published_artifacts"] = published
    write_json_versioned(PUBLIC_MANIFEST_PATH, manifest)
    return PUBLIC_MANIFEST_PATH


def update_public_claims(section_name: str, claims: list[dict[str, Any]]) -> Path:
    claim_map = load_json(PUBLIC_CLAIMS_PATH, default={})
    if not isinstance(claim_map, dict) or claim_map.get("schema") != "public-claims-v1":
        claim_map = {}
    sections = claim_map.get("sections", {})
    if not isinstance(sections, dict):
        sections = {}
    sections[section_name] = claims
    claim_map["schema"] = "public-claims-v1"
    claim_map["last_updated_utc"] = now_utc_iso()
    claim_map["sections"] = sections
    write_json_versioned(PUBLIC_CLAIMS_PATH, claim_map)
    return PUBLIC_CLAIMS_PATH


def update_internal_manifest(section_name: str, payload: dict[str, Any]) -> Path:
    manifest = load_json(INTERNAL_RUN_MANIFEST_PATH, default={})
    if not isinstance(manifest, dict) or manifest.get("manifest_schema") != "internal-v1":
        manifest = {}
    sections = manifest.get("sections", {})
    if not isinstance(sections, dict):
        sections = {}
    sections[section_name] = payload
    manifest["manifest_schema"] = "internal-v1"
    manifest["last_updated_utc"] = now_utc_iso()
    manifest["sections"] = sections
    write_json_versioned(INTERNAL_RUN_MANIFEST_PATH, manifest)
    return INTERNAL_RUN_MANIFEST_PATH
