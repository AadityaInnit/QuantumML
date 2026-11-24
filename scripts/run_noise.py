"""Small descriptive noise study: exact -> finite shots -> frozen fake IBM noise."""
import argparse
import json
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from qkb.data import load_raw, preprocessor
from qkb.bench import prepare_inner, inner_folds, tune_quantum, C_GRID
from qkb.qkernel import feature_map, exact_kernel, shot_kernel, noisy_kernel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-train", type=int, default=40)
    p.add_argument("--n-test", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise-shots", type=int, default=1024)
    p.add_argument("--shot-repeats", type=int, default=5)
    a = p.parse_args()
    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
    backend = FakeSherbrooke()
    X, Xt, y, yt, ids, idst = load_raw(seed=a.seed)
    if not (12 <= a.n_train < len(y) and 2 <= a.n_test < len(yt)):
        p.error("Use 12 <= n-train < 112 and 2 <= n-test < 48")
    tr, _ = train_test_split(np.arange(len(y)), train_size=a.n_train, stratify=y, random_state=a.seed)
    te, _ = train_test_split(np.arange(len(yt)), train_size=a.n_test, stratify=yt, random_state=a.seed)
    X, Xt, y, yt = X[tr], Xt[te], y[tr], yt[te]
    prep = preprocessor(4, a.seed)
    Xtr, Xte = prep.fit_transform(X), prep.transform(Xt)
    prepared = prepare_inner(X, y, 4, inner_folds(X, y, a.seed), a.seed)
    configs = [("Z", 3, .25), ("ZZ", 1, .25), ("ZZ", 2, .25)]
    rows, selected_c = [], {}
    out = Path("results/noise")
    out.mkdir(parents=True, exist_ok=True)
    (out / "fake_backend_properties.json").write_text(json.dumps(backend.properties().to_dict(), indent=2, default=str))
    (out / "fake_backend_configuration.json").write_text(json.dumps(backend.configuration().to_dict(), indent=2, default=str))
    for config in configs:
        kind, reps, bw = config
        _, best_c, _ = tune_quantum(prepared, [config], 4)
        c = C_GRID[best_c[0]]
        selected_c[f"{kind}-{reps}"] = c
        fm = feature_map(kind, 4, reps)
        Etr, Ete = exact_kernel(fm, Xtr, bandwidth=bw), exact_kernel(fm, Xte, Xtr, bandwidth=bw)
        off = ~np.eye(len(Xtr), dtype=bool)
        def record(Ktr, Kte, level, repeat=0, shots=0):
            pred = SVC(kernel="precomputed", C=c).fit(Ktr, y).predict(Kte)
            rows.append(dict(fmap=kind, reps=reps, bandwidth=bw, C=c, level=level, repeat=repeat, shots=shots,
                err=np.linalg.norm(Ktr-Etr)/np.linalg.norm(Etr),
                offdiag_err=np.linalg.norm((Ktr-Etr)[off])/np.linalg.norm(Etr[off]),
                cross_err=np.linalg.norm(Kte-Ete)/np.linalg.norm(Ete),
                min_eigenvalue=float(np.linalg.eigvalsh(Ktr)[0]), diagonal_mean=float(np.diag(Ktr).mean()),
                test_acc=accuracy_score(yt, pred)))
        record(Etr, Ete, "exact")
        for shots in [32, 128, 512, 2048]:
            for repeat in range(a.shot_repeats):
                rng = np.random.default_rng(a.seed + shots + 10000 * repeat)
                record(shot_kernel(Etr, shots, rng, symmetric=True), shot_kernel(Ete, shots, rng),
                       "finite shots", repeat, shots)
        t0 = time.perf_counter()
        Ktr = noisy_kernel(fm, Xtr, bandwidth=bw, backend=backend, shots=a.noise_shots, seed=a.seed)
        Kte = noisy_kernel(fm, Xte, Xtr, bandwidth=bw, backend=backend, shots=a.noise_shots, seed=a.seed+100000)
        record(Ktr, Kte, "fake_sherbrooke noise", shots=a.noise_shots)
        np.savez_compressed(out / f"{kind}_reps{reps}_kernels.npz", exact_train=Etr, exact_test=Ete,
                            noisy_train=Ktr, noisy_test=Kte)
        pd.DataFrame(rows).to_csv(out / "noise_study.csv", index=False)
        print(f"{kind} reps={reps}: {time.perf_counter()-t0:.1f}s; {rows[-1]}", flush=True)
    (out / "summary.json").write_text(json.dumps(dict(n_train=len(y), n_test=len(yt), seed=a.seed,
        backend=backend.name, noise_shots=a.noise_shots, shot_repeats=a.shot_repeats,
        train_ids=ids[tr].tolist(), test_ids=idst[te].tolist(), C_by_config=selected_c,
        caveat="Fixed C from exact train-only CV; raw indefinite noisy Gram matrices; single fake-backend realization; descriptive accuracy only."), indent=2))


if __name__ == "__main__":
    main()
