"""Fidelity kernels from statevectors and compute-uncompute simulation."""
import numpy as np
from qiskit.circuit.library import zz_feature_map, z_feature_map
from qiskit.quantum_info import Statevector


def feature_map(kind, n_qubits, reps=2):
    if kind == "Z":
        return z_feature_map(n_qubits, reps=reps)
    if kind in ("ZZ", "ZZfull"):
        return zz_feature_map(n_qubits, reps=reps, entanglement="linear" if kind == "ZZ" else "full")
    raise ValueError(kind)


def _states(fm, X, bandwidth):
    return np.array([Statevector(fm.assign_parameters(bandwidth * np.asarray(x))).data for x in X])


def exact_kernel(fm, XA, XB=None, bandwidth=1.0):
    A = _states(fm, XA, bandwidth)
    B = A if XB is None else _states(fm, XB, bandwidth)
    return np.clip(np.abs(A.conj() @ B.T) ** 2, 0, 1)


def shot_kernel(K_exact, shots, rng, symmetric=False):
    """Binomial compute-uncompute sampling. Shape does not identify a Gram matrix."""
    if shots <= 0:
        raise ValueError("shots must be positive")
    K = rng.binomial(shots, np.clip(K_exact, 0, 1)) / shots
    if symmetric:
        if K.shape[0] != K.shape[1]:
            raise ValueError("Symmetric kernel must be square")
        K = np.triu(K) + np.triu(K, 1).T
        np.fill_diagonal(K, 1.)
    return K


def noisy_kernel(fm, XA, XB=None, bandwidth=1.0, backend=None, shots=1024, seed=0):
    """Raw noisy kernel, measured diagonal included; no PSD projection/mitigation.

    A barrier keeps U(a) and U(b)^dagger separated during optimization. For a
    training Gram matrix one circuit per unordered pair is mirrored explicitly.
    Aer truncates idle physical qubits after transpilation to the fake backend.
    """
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    sim = AerSimulator.from_backend(backend, method="density_matrix") if backend is not None else AerSimulator(method="density_matrix")
    sim.set_options(max_parallel_threads=1)
    sym = XB is None
    XB_ = XA if sym else XB
    K = np.zeros((len(XA), len(XB_)))
    pairs = [(i, j) for i in range(len(XA)) for j in range(len(XB_)) if not sym or j >= i]
    # Batches bound peak memory and preserve deterministic circuit sampling seeds.
    for start in range(0, len(pairs), 128):
        subset = pairs[start:start+128]
        circuits = []
        for i, j in subset:
            qc = fm.assign_parameters(bandwidth * np.asarray(XA[i]))
            qc.barrier()
            qc = qc.compose(fm.assign_parameters(bandwidth * np.asarray(XB_[j])).inverse())
            qc.measure_all()
            circuits.append(qc)
        tc = transpile(circuits, sim, optimization_level=1, seed_transpiler=seed, num_processes=1)
        result = sim.run(tc, shots=shots, seed_simulator=seed + start).result()
        for k, (i, j) in enumerate(subset):
            K[i, j] = result.get_counts(k).get("0" * fm.num_qubits, 0) / shots
            if sym:
                K[j, i] = K[i, j]
    return K
