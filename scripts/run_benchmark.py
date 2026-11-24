"""Nested benchmark and exploratory fixed-map sweep. Run from repository root."""
import argparse
import json
import os
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics.pairwise import rbf_kernel
from qkb.data import load_raw, preprocessor
from qkb.bench import (C_GRID, CONFIGS, classical_models, inner_folds, fit_classical,
                       prepare_inner, tune_quantum, quantum_predict, metrics)
from qkb.qkernel import feature_map, exact_kernel
from qkb.diagnostics import concentration, target_alignment, geometric_difference, model_complexity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["breast_cancer", "circles"], default="breast_cancer")
    p.add_argument("--n-features", type=int, default=4)
    p.add_argument("--n-samples", type=int, default=160)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--splits", type=int, default=5)
    p.add_argument("--repeats", type=int, default=3)
    a = p.parse_args()
    nq = 2 if a.dataset == "circles" else a.n_features
    out = Path(f"results/{a.dataset}_{nq}q")
    out.mkdir(parents=True, exist_ok=True)
    X, Xt, y, yt, train_ids, test_ids = load_raw(a.dataset, a.n_samples, seed=a.seed)
    configs = [c for c in CONFIGS if nq >= 3 or c[0] != "ZZfull"]
    folds = list(RepeatedStratifiedKFold(n_splits=a.splits, n_repeats=a.repeats,
                                        random_state=a.seed).split(X, y))
    models = classical_models(a.seed)
    records, sweep_records, selections = [], [], []
    t0 = time.perf_counter()
    for fold, (tr, va) in enumerate(folds):
        inner = inner_folds(X[tr], y[tr], a.seed)
        for name, (est, grid) in models.items():
            fit = fit_classical(X[tr], y[tr], est, grid, nq, inner, a.seed)
            records.append(dict(model=name, fold=fold, **metrics(y[va], fit.predict(X[va]))))
        prepared = prepare_inner(X[tr], y[tr], nq, inner, a.seed)
        best_i, best_c, scores = tune_quantum(prepared, configs, nq)
        prep = preprocessor(nq, a.seed)
        Xtr, Xva = prep.fit_transform(X[tr]), prep.transform(X[va])
        # Fixed-config outer scores are EXPLORATORY; final model selection uses inner CV.
        for i, config in enumerate(configs):
            pred = quantum_predict(config, C_GRID[best_c[i]], Xtr, y[tr], Xva)
            m = metrics(y[va], pred)
            sweep_records.append(dict(fold=fold, fmap=config[0], reps=config[1], bandwidth=config[2], **m))
            if i == best_i:
                records.append(dict(model="QKernel-SVM (nested selection)", fold=fold, **m))
        selections.append(dict(fold=fold, config=configs[best_i], C=C_GRID[best_c[best_i]]))
        print(f"{a.dataset} fold {fold+1}/{len(folds)}: {configs[best_i]}, elapsed {time.perf_counter()-t0:.1f}s", flush=True)
    fold_df = pd.DataFrame(records)
    fold_df.to_csv(out / "fold_scores.csv", index=False)
    pd.DataFrame(sweep_records).to_csv(out / "sweep_fold_scores.csv", index=False)
    # All hyperparameters for the final holdout fits are selected using TRAIN ONLY.
    final_inner = inner_folds(X, y, a.seed)
    final_fits, rows = {}, []
    for name, (est, grid) in models.items():
        fit = fit_classical(X, y, est, grid, nq, final_inner, a.seed)
        final_fits[name] = fit
        cv = fold_df[fold_df.model == name]
        rows.append(dict(model=name, family="classical", cv_acc=cv.acc.mean(), cv_std=cv.acc.std(ddof=0),
                         cv_f1=cv.f1.mean(), test_acc=metrics(yt, fit.predict(Xt))["acc"]))
    prepared = prepare_inner(X, y, nq, final_inner, a.seed)
    best_i, best_c, scores = tune_quantum(prepared, configs, nq)
    config, c = configs[best_i], C_GRID[best_c[best_i]]
    prep = preprocessor(nq, a.seed)
    Xtr, Xte = prep.fit_transform(X), prep.transform(Xt)
    qpred = quantum_predict(config, c, Xtr, y, Xte)
    qcv = fold_df[fold_df.model.str.startswith("QKernel")]
    rows.append(dict(model="QKernel-SVM (nested selection)", family="quantum", cv_acc=qcv.acc.mean(),
                     cv_std=qcv.acc.std(ddof=0), cv_f1=qcv.f1.mean(), test_acc=metrics(yt, qpred)["acc"]))
    pd.DataFrame(rows).to_csv(out / "benchmark.csv", index=False)
    # Train-only diagnostics, using the SAME tuned RBF gamma as the RBF baseline.
    rbf = final_fits["SVM-RBF"].best_estimator_.named_steps["model"]
    gamma = float(rbf._gamma)
    Kc = rbf_kernel(Xtr, gamma=gamma)
    sweep, sensitivity = [], []
    swdf = pd.DataFrame(sweep_records)
    for i, (kind, reps, bw) in enumerate(configs):
        K = exact_kernel(feature_map(kind, nq, reps), Xtr, bandwidth=bw)
        vals = swdf[(swdf.fmap == kind) & (swdf.reps == reps) & (swdf.bandwidth == bw)]
        m, v = concentration(K)
        sweep.append(dict(fmap=kind, reps=reps, bandwidth=bw, cv_acc=vals.acc.mean(), cv_std=vals.acc.std(ddof=0),
                          inner_selection_acc=scores[i, best_c[i]], offdiag_mean=m, offdiag_var=v,
                          alignment=target_alignment(K, y), geo_diff_ridge_0_001=geometric_difference(K, Kc),
                          ridge_complexity=model_complexity(K, y)))
        for reg in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]:
            sensitivity.append(dict(fmap=kind, reps=reps, bandwidth=bw, reg=reg,
                                    geometric_difference=geometric_difference(K, Kc, reg)))
    pd.DataFrame(sweep).to_csv(out / "qkernel_sweep.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(out / "geometric_sensitivity.csv", index=False)
    paired = []
    for name in models:
        base = fold_df[fold_df.model == name].sort_values("fold").acc.to_numpy()
        delta = qcv.sort_values("fold").acc.to_numpy() - base
        paired.append(dict(comparator=name, mean_delta=delta.mean(), std_delta=delta.std(ddof=0),
                           quantum_wins=int((delta > 1e-12).sum()), ties=int((abs(delta) <= 1e-12).sum()),
                           classical_wins=int((delta < -1e-12).sum()),
                           note="Descriptive paired folds; repeated CV folds are dependent, no independent-fold p-value."))
    pd.DataFrame(paired).to_csv(out / "paired_comparisons.csv", index=False)
    summary = dict(dataset=a.dataset, n_features=nq, n_train=len(y), n_test=len(yt), seed=a.seed,
                   outer_splits=a.splits, outer_repeats=a.repeats, inner_splits=3,
                   best_quantum=dict(fmap=config[0], reps=config[1], bandwidth=config[2], C=c),
                   qtest=metrics(yt, qpred)["acc"], rbf_gamma=gamma, fold_selections=selections,
                   train_ids=train_ids.tolist(), test_ids=test_ids.tolist(),
                   outer_folds=[dict(train=tr.tolist(), validation=va.tolist()) for tr, va in folds],
                   elapsed_seconds=time.perf_counter()-t0)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
