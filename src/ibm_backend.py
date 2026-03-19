from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .helpers import append_jsonl, load_json, now_utc_iso, stable_hash_circuit, write_json


def init_service(token_env: str = "QISKIT_IBM_TOKEN") -> Any:
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("qiskit-ibm-runtime is required for IBM hardware mode") from exc

    token = os.environ.get(token_env)
    if token:
        return QiskitRuntimeService(channel="ibm_quantum", token=token)
    return QiskitRuntimeService(channel="ibm_quantum")


def get_backend(service: Any, name: str) -> Any:
    try:
        return service.backend(name)
    except Exception as exc:
        raise RuntimeError(f"Unable to access backend '{name}'.") from exc


def _counts_cache_path(cache_dir: Path, circuit_hash: str) -> Path:
    return cache_dir / f"counts_{circuit_hash}.json"


def _load_cached_counts(cache_dir: Path, circuit_hash: str) -> dict[str, int] | None:
    path = _counts_cache_path(cache_dir, circuit_hash)
    if not path.exists():
        return None
    data = load_json(path, default={})
    return {str(k): int(v) for k, v in data.items()}


def _save_cached_counts(cache_dir: Path, circuit_hash: str, counts: dict[str, int]) -> None:
    path = _counts_cache_path(cache_dir, circuit_hash)
    write_json(path, {str(k): int(v) for k, v in counts.items()})


def _extract_counts_pub(pub_result: Any) -> dict[str, int]:
    # Common Runtime shape: pub_result.data.c.get_counts()
    try:
        raw = pub_result.data.c.get_counts()
        return {str(k): int(v) for k, v in raw.items()}
    except Exception:
        pass

    # Older/alternative shapes.
    for attr in ["quasi_dists", "data", "results", "counts"]:
        try:
            obj = getattr(pub_result, attr)
            if isinstance(obj, dict) and obj:
                # If already bitstring dict.
                if all(isinstance(k, str) for k in obj.keys()):
                    return {str(k): int(v) for k, v in obj.items()}
            if isinstance(obj, list) and obj:
                first = obj[0]
                if isinstance(first, dict):
                    if all(isinstance(k, str) for k in first.keys()):
                        return {str(k): int(v) for k, v in first.items()}
        except Exception:
            continue

    # Last resort: try to parse json repr.
    try:
        maybe = json.loads(str(pub_result))
        if isinstance(maybe, dict) and "counts" in maybe:
            return {str(k): int(v) for k, v in maybe["counts"].items()}
    except Exception:
        pass
    raise RuntimeError("Unable to extract counts from SamplerV2 result payload")


def _normalize_counts_bitlen(counts: dict[str, int], nbits: int) -> dict[str, int]:
    out: dict[str, int] = {}
    for bitstring, ct in counts.items():
        bs = str(bitstring).replace(" ", "")
        bs = bs.zfill(nbits)[-nbits:]
        out[bs] = out.get(bs, 0) + int(ct)
    return out


def run_sampler_batch(
    circuits: list[Any],
    shots: int,
    backend_name: str,
    cache_dir: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Run batch on IBM Runtime SamplerV2 with circuit-hash cache.
    """
    try:
        from qiskit import transpile
        from qiskit_ibm_runtime import SamplerV2 as Sampler
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Qiskit Runtime stack unavailable for hardware execution") from exc

    service = init_service()
    backend = get_backend(service, backend_name)

    transpiled = transpile(circuits, backend=backend, optimization_level=1)
    if not isinstance(transpiled, list):
        transpiled = [transpiled]

    hashes = [stable_hash_circuit(qc) for qc in transpiled]

    counts_list: list[dict[str, int] | None] = [None] * len(transpiled)
    missing_idx: list[int] = []
    for i, hsh in enumerate(hashes):
        cached = _load_cached_counts(cache_dir, hsh)
        if cached is not None:
            nbits = circuits[i].num_clbits
            counts_list[i] = _normalize_counts_bitlen(cached, nbits)
        else:
            missing_idx.append(i)

    dry_summary: list[dict[str, Any]] = []
    for i, qc in enumerate(transpiled):
        two_qubit = sum(1 for inst in qc.data if len(inst.qubits) == 2 and inst.operation.name != "barrier")
        dry_summary.append(
            {
                "index": i,
                "hash": hashes[i],
                "depth": int(qc.depth()),
                "two_qubit_gates": int(two_qubit),
                "size": int(qc.size()),
            }
        )

    if dry_run:
        return {
            "counts": [c if c is not None else {} for c in counts_list],
            "job_ids": [],
            "backend": backend_name,
            "dry_run": True,
            "circuit_summaries": dry_summary,
        }

    job_ids: list[str] = []
    if missing_idx:
        missing_circuits = [transpiled[i] for i in missing_idx]
        sampler = Sampler(mode=backend)
        job = sampler.run(missing_circuits, shots=shots)
        job_id = job.job_id()
        job_ids.append(job_id)
        result = job.result()

        append_jsonl(
            cache_dir / "ibm_jobs.jsonl",
            {
                "timestamp": now_utc_iso(),
                "job_id": job_id,
                "backend": backend_name,
                "shots": int(shots),
                "n_circuits": len(missing_idx),
                "circuit_hashes": [hashes[i] for i in missing_idx],
            },
        )

        for local_idx, pub in enumerate(result):
            global_idx = missing_idx[local_idx]
            raw = _extract_counts_pub(pub)
            nbits = circuits[global_idx].num_clbits
            norm = _normalize_counts_bitlen(raw, nbits)
            counts_list[global_idx] = norm
            _save_cached_counts(cache_dir, hashes[global_idx], norm)

    # Any unresolved element means extraction failed.
    final_counts: list[dict[str, int]] = []
    for i, c in enumerate(counts_list):
        if c is None:
            raise RuntimeError(f"Missing counts for circuit index {i}")
        final_counts.append(c)

    return {
        "counts": final_counts,
        "job_ids": job_ids,
        "backend": backend_name,
        "dry_run": False,
        "circuit_summaries": dry_summary,
    }


def build_aer_noise_model(noise: str, ibm_backend_name: str | None = None) -> tuple[Any | None, str]:
    mode = noise.lower().strip()
    if mode == "ideal":
        return None, "ideal"

    try:
        from qiskit_aer.noise import NoiseModel, depolarizing_error
    except Exception:
        return None, "fallback_no_aer"

    if mode == "depolarizing":
        nm = NoiseModel()
        nm.add_all_qubit_quantum_error(depolarizing_error(0.002, 1), ["id", "rz", "sx", "x", "h"])
        nm.add_all_qubit_quantum_error(depolarizing_error(0.01, 2), ["cx", "ecr"])
        return nm, "depolarizing"

    if mode == "from_backend":
        try:
            service = init_service()
            if not ibm_backend_name:
                raise RuntimeError("ibm_backend_name required for from_backend")
            backend = get_backend(service, ibm_backend_name)
            nm = NoiseModel.from_backend(backend)
            return nm, f"from_backend:{ibm_backend_name}"
        except Exception:
            # Clean fallback.
            nm = NoiseModel()
            nm.add_all_qubit_quantum_error(depolarizing_error(0.003, 1), ["id", "rz", "sx", "x", "h"])
            nm.add_all_qubit_quantum_error(depolarizing_error(0.015, 2), ["cx", "ecr"])
            return nm, "from_backend_fallback_depolarizing"

    raise ValueError(f"Unknown noise mode: {noise}")


def run_aer_batch(
    circuits: list[Any],
    shots: int,
    seed: int,
    noise: str,
    ibm_backend_name: str | None = None,
) -> dict[str, Any] | None:
    try:
        from qiskit import transpile
        from qiskit_aer import AerSimulator
    except Exception:
        return None

    noise_model, noise_tag = build_aer_noise_model(noise=noise, ibm_backend_name=ibm_backend_name)
    if noise_model is None:
        sim = AerSimulator(seed_simulator=seed)
    else:
        sim = AerSimulator(noise_model=noise_model, seed_simulator=seed)

    tqcs = transpile(circuits, sim, optimization_level=0, seed_transpiler=seed)
    if not isinstance(tqcs, list):
        tqcs = [tqcs]
    job = sim.run(tqcs, shots=shots, seed_simulator=seed)
    res = job.result()

    counts_list: list[dict[str, int]] = []
    for i, qc in enumerate(tqcs):
        raw = res.get_counts(i)
        counts_list.append(_normalize_counts_bitlen({str(k): int(v) for k, v in raw.items()}, qc.num_clbits))

    return {
        "counts": counts_list,
        "noise": noise_tag,
        "backend": "aer",
    }
