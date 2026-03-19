from __future__ import annotations

from typing import Any

import numpy as np

from .data import StateParams

A_Q = 0
B_Q = 1
P_Q = 2


def _require_qiskit() -> Any:
    try:
        from qiskit import QuantumCircuit

        return QuantumCircuit
    except Exception as exc:  # pragma: no cover - depends on env
        raise RuntimeError("Qiskit is required for circuit construction") from exc


def _prepare_component_3q(qc: Any, params: StateParams, component_tag: str) -> None:
    theta = float(params.theta)
    if component_tag == "entangled":
        qc.ry(2.0 * theta, A_Q)
        qc.cx(A_Q, B_Q)
        return

    # Product components.
    if component_tag not in {"a0b0", "a0b1", "a1b0", "a1b1"}:
        raise ValueError(f"Unknown component_tag: {component_tag}")
    a_state = int(component_tag[1])
    b_state = int(component_tag[3])
    if a_state == 1:
        qc.x(A_Q)
    if b_state == 1:
        qc.x(B_Q)


def _apply_weak_interaction_xb_yp(qc: Any, g: float) -> None:
    # Implements exp(-i g/2 X_B ⊗ Y_P) via local basis changes and ZZ interaction.
    qc.h(B_Q)
    qc.rx(-np.pi / 2.0, P_Q)
    qc.cx(B_Q, P_Q)
    qc.rz(float(g), P_Q)
    qc.cx(B_Q, P_Q)
    qc.h(B_Q)
    qc.rx(np.pi / 2.0, P_Q)


def _apply_pointer_basis_rotation(qc: Any, pointer_basis: str) -> None:
    basis = pointer_basis.upper()
    if basis == "Z":
        return
    if basis == "X":
        qc.h(P_Q)
        return
    if basis == "Y":
        qc.sdg(P_Q)
        qc.h(P_Q)
        return
    raise ValueError(f"Unknown pointer basis: {pointer_basis}")


def build_weak_measurement_circuit(params: StateParams, setting: Any, component_tag: str) -> Any:
    QuantumCircuit = _require_qiskit()
    qc = QuantumCircuit(3, 3)

    _prepare_component_3q(qc, params=params, component_tag=component_tag)
    _apply_weak_interaction_xb_yp(qc, g=float(setting.g))

    # Local compression measurement basis on A: H then Z measurement.
    qc.h(A_Q)

    # Optional B postselection basis rotation.
    if abs(float(setting.b_basis_angle)) > 1e-12:
        qc.ry(float(setting.b_basis_angle), B_Q)

    _apply_pointer_basis_rotation(qc, setting.pointer_basis)

    qc.measure(A_Q, 0)
    qc.measure(B_Q, 1)
    qc.measure(P_Q, 2)

    qc.metadata = {
        "component_tag": component_tag,
        "g": float(setting.g),
        "pointer_basis": str(setting.pointer_basis),
        "b_basis_angle": float(setting.b_basis_angle),
    }
    return qc


def _prepare_component_2q(qc: Any, params: StateParams, component_tag: str) -> None:
    theta = float(params.theta)
    if component_tag == "entangled":
        qc.ry(2.0 * theta, 0)
        qc.cx(0, 1)
        return
    if component_tag not in {"a0b0", "a0b1", "a1b0", "a1b1"}:
        raise ValueError(f"Unknown component_tag: {component_tag}")
    a_state = int(component_tag[1])
    b_state = int(component_tag[3])
    if a_state == 1:
        qc.x(0)
    if b_state == 1:
        qc.x(1)


def _apply_measure_basis_rotation(qc: Any, qubit: int, basis: str) -> None:
    b = basis.upper()
    if b == "Z":
        return
    if b == "X":
        qc.h(qubit)
        return
    if b == "Y":
        qc.sdg(qubit)
        qc.h(qubit)
        return
    raise ValueError(f"Unknown Pauli basis {basis}")


def build_tomography_circuit(params: StateParams, basisA: str, basisB: str, component_tag: str) -> Any:
    QuantumCircuit = _require_qiskit()
    qc = QuantumCircuit(2, 2)
    _prepare_component_2q(qc, params=params, component_tag=component_tag)
    _apply_measure_basis_rotation(qc, qubit=0, basis=basisA)
    _apply_measure_basis_rotation(qc, qubit=1, basis=basisB)
    qc.measure(0, 0)
    qc.measure(1, 1)
    qc.metadata = {
        "component_tag": component_tag,
        "basisA": basisA,
        "basisB": basisB,
    }
    return qc


def circuit_stats(qc: Any) -> dict[str, int]:
    two_qubit = sum(1 for inst in qc.data if len(inst.qubits) == 2 and inst.operation.name not in {"barrier"})
    return {
        "depth": int(qc.depth()),
        "size": int(qc.size()),
        "two_qubit_gates": int(two_qubit),
    }
