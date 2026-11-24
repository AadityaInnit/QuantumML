"""Train-only off-diagonal statistics as qubit count varies; no accuracy claim."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from qkb.data import load_raw, preprocessor
from qkb.qkernel import exact_kernel, feature_map
from qkb.diagnostics import concentration, target_alignment


def main():
    X, _, y, _, _, _ = load_raw()
    rows = []
    for nq in [2, 4, 6, 8]:
        A = preprocessor(nq).fit_transform(X)
        for kind in ["Z", "ZZ", "ZZfull"]:
            for bw in [.25, 1.]:
                K = exact_kernel(feature_map(kind, nq, 2), A, bandwidth=bw)
                m, v = concentration(K)
                rows.append(dict(qubits=nq, fmap=kind, reps=2, bandwidth=bw, offdiag_mean=m,
                                 offdiag_var=v, alignment=target_alignment(K, y)))
    out = Path("results/concentration")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "qubit_sweep.csv", index=False)


if __name__ == "__main__":
    main()
