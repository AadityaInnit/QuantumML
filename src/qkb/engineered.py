"""Disclosed circuit-teacher task and a projected kernel from local Pauli features.

This is a small classically simulated representation experiment, not a classical
hardness construction or a literal reproduction of Huang et al.'s dataset.
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp

TEACHER_SEED = 271828
N_QUBITS = 6
REPS = 2
FREQUENCY = 2.0


def make_circuit(entangling=True):
    """Two data-encoding layers interleaved with fixed rotations and a CNOT ring."""
    rng = np.random.default_rng(TEACHER_SEED)
    offsets = rng.uniform(-np.pi, np.pi, size=(REPS, N_QUBITS, 2))
    x = ParameterVector("x", N_QUBITS)
    qc = QuantumCircuit(N_QUBITS)
    for layer in range(REPS):
        for q in range(N_QUBITS):
            qc.ry(FREQUENCY * x[q] + offsets[layer, q, 0], q)
            qc.rz(x[(q + layer + 1) % N_QUBITS] + offsets[layer, q, 1], q)
        if entangling:
            for q in range(N_QUBITS - 1):
                qc.cx(q, q + 1)
            qc.cx(N_QUBITS - 1, 0)
    return qc


def observables():
    """Local X,Y,Z on each qubit, ordered q0:X,Y,Z; q1:X,Y,Z; ... ."""
    result = []
    for q in range(N_QUBITS):
        for axis in "XYZ":
            label = ["I"] * N_QUBITS
            label[N_QUBITS - 1 - q] = axis
            result.append(SparsePauliOp("".join(label)))
    return result


def projected_features(X, entangling=True):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != N_QUBITS or not np.isfinite(X).all():
        raise ValueError(f"Expected finite (n, {N_QUBITS}) inputs")
    circuit = make_circuit(entangling)
    ops = observables()
    features = np.empty((len(X), len(ops)))
    for i, x in enumerate(X):
        state = Statevector(circuit.assign_parameters(x))
        features[i] = [state.expectation_value(op).real for op in ops]
    return features


def teacher_weights():
    weights = np.random.default_rng(TEACHER_SEED + 1).normal(size=3 * N_QUBITS)
    return weights / np.linalg.norm(weights)


def labels(features, threshold):
    return (np.asarray(features) @ teacher_weights() >= threshold).astype(int)


def draw_inputs(seed, n, stream):
    """Disjoint seeded streams for pilot/calibration/train/test; iid uniform inputs."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(stream), TEACHER_SEED]))
    return rng.uniform(-np.pi, np.pi, size=(n, N_QUBITS))


def fourier_features(X):
    """Classical domain-informed competitor: coordinate Fourier terms AND pairs.

    Includes frequencies 1..4, encompassing input encoding frequencies. These
    features are deterministic and fit no parameters on train or test data.
    """
    X = np.asarray(X)
    pieces = [np.sin(k * X) for k in range(1, 5)] + [np.cos(k * X) for k in range(1, 5)]
    F = np.concatenate(pieces, axis=1)
    # Pairwise products include cross-coordinate interactions and harmonics.
    i, j = np.triu_indices(F.shape[1], k=0)
    return np.column_stack([F, F[:, i] * F[:, j]])
