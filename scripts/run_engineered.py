"""Pilot -> freeze -> fresh evaluation of a disclosed quantum circuit teacher.

Run each stage explicitly. Only freeze/pilot can access calibration/pilot data;
evaluation verifies the frozen protocol and code hashes before drawing test data.
"""
import argparse
import hashlib
import itertools
import json
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel
from qkb.engineered import (TEACHER_SEED, N_QUBITS, REPS, FREQUENCY, projected_features,
                            teacher_weights, labels, draw_inputs, fourier_features)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/engineered"
PROTOCOL = ROOT / "experiments/engineered_protocol.json"
C_GRID = [.1, 1., 10., 100.]
GAMMA_GRID = [.1, 1., 10., 100.]
RAW_MODELS = ["RBF SVM", "MLP", "Random forest", "Gradient boosting", "Fourier+pairwise SVM"]
QMODEL = "Projected quantum kernel"
SAME_MODEL = "Classical same-feature RBF"
LINEAR_MODEL = "Classical same-feature logistic"
ABLATION = "Nonentangling projection RBF"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    return {str(p.relative_to(ROOT)): sha(p) for p in
            [ROOT / "src/qkb/engineered.py", Path(__file__).resolve()]}


def model_specs(seed):
    return {
        "RBF SVM": (SVC(), {"C": C_GRID, "gamma": [.01, .1, 1., 10.]}),
        "MLP": (MLPClassifier(max_iter=2000, max_fun=50000, random_state=seed, early_stopping=False),
                {"hidden_layer_sizes": [(32,), (64, 32)], "alpha": [.0001, .01], "solver": ["adam", "lbfgs"]}),
        "Random forest": (RandomForestClassifier(n_estimators=150, random_state=seed, n_jobs=1),
                          {"max_depth": [4, None], "min_samples_leaf": [1, 3]}),
        "Gradient boosting": (HistGradientBoostingClassifier(random_state=seed, max_iter=150),
                              {"max_leaf_nodes": [7, 15], "l2_regularization": [0., 1.], "min_samples_leaf": [5]}),
        "Fourier+pairwise SVM": (SVC(), {"C": C_GRID, "gamma": [.0001, .001, .01, .1]}),
        SAME_MODEL: (SVC(), {"C": C_GRID, "gamma": GAMMA_GRID}),
        LINEAR_MODEL: (LogisticRegression(max_iter=3000), {"C": C_GRID}),
        ABLATION: (SVC(), {"C": C_GRID, "gamma": GAMMA_GRID}),
    }


def fit_grid(name, X, y, folds, seed):
    est, grid = model_specs(seed)[name]
    # Projected RBF comparisons use the actual local-Pauli geometry unchanged.
    # Raw baselines fit scaling inside the inner folds.
    if name in RAW_MODELS:
        est = Pipeline([("scaler", StandardScaler()), ("model", est)])
        grid = {"model__" + k: v for k, v in grid.items()}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = GridSearchCV(est, grid, cv=folds, scoring="accuracy", n_jobs=1).fit(X, y)
    fit.recorded_warnings_ = [dict(category=type(w.message).__name__, message=str(w.message)) for w in caught]
    return fit


def fit_projected(F, y, folds):
    """Fixed deterministic features; only C and gamma are selected in inner CV."""
    kernels = {g: rbf_kernel(F, gamma=g) for g in GAMMA_GRID}
    trials = []
    for c, gamma in itertools.product(C_GRID, GAMMA_GRID):
        K = kernels[gamma]
        scores = []
        for tr, va in folds:
            pred = SVC(kernel="precomputed", C=c).fit(K[np.ix_(tr, tr)], y[tr]).predict(K[np.ix_(va, tr)])
            scores.append(accuracy_score(y[va], pred))
        trials.append((float(np.mean(scores)), c, gamma))
    # Same parameter order and first-maximum tie break as GridSearchCV.
    score, c, gamma = max(trials, key=lambda row: row[0])
    fit = SVC(kernel="precomputed", C=c).fit(kernels[gamma], y)
    return fit, dict(C=c, gamma=gamma), score


def evaluate_dataset(seed, sizes, n_test, threshold, output):
    """Test predictions happen only after ALL fits for each training size finish."""
    start = time.perf_counter()
    Xpool = draw_inputs(seed, max(sizes), 10)
    Xtest = draw_inputs(seed, n_test, 20)
    Fpool = projected_features(Xpool)
    Ftest = projected_features(Xtest)
    Apool = projected_features(Xpool, entangling=False)
    Atest = projected_features(Xtest, entangling=False)
    ypool, ytest = labels(Fpool, threshold), labels(Ftest, threshold)
    # Deterministic Fourier competitor sees the encoding's frequency range.
    Ppool, Ptest = fourier_features(Xpool), fourier_features(Xtest)
    output.mkdir(parents=True, exist_ok=True)
    # Save exact data, features and labels for auditing the whole task.
    np.savez_compressed(output / f"dataset_seed{seed}.npz", X_train_pool=Xpool, X_test=Xtest,
                        F_train_pool=Fpool, F_test=Ftest, A_train_pool=Apool, A_test=Atest,
                        y_train_pool=ypool, y_test=ytest)
    rows, predictions, warning_records = [], {}, []
    for n in sizes:
        X, F, A, P, y = Xpool[:n], Fpool[:n], Apool[:n], Ppool[:n], ypool[:n]
        if np.bincount(y, minlength=2).min() < 3:
            raise ValueError("Fewer than three examples in a class; report failure, do not redraw")
        folds = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(X, y))
        qfit, params, qcv = fit_projected(F, y, folds)
        fits = {}
        for name in model_specs(seed):
            train = P if name == "Fourier+pairwise SVM" else (F if name in [SAME_MODEL, LINEAR_MODEL] else (A if name == ABLATION else X))
            fits[name] = fit_grid(name, train, y, folds, seed)
        # Classical comparator is chosen using inner CV scores only, never test accuracy.
        chosen = max(RAW_MODELS, key=lambda name: fits[name].best_score_)
        pred = qfit.predict(rbf_kernel(Ftest, F, gamma=params['gamma']))
        predictions[f"n{n}_quantum"] = pred
        qrow = dict(seed=seed, n_train=n, n_test=n_test, model=QMODEL, test_acc=accuracy_score(ytest, pred),
                    inner_cv_acc=qcv, best_params=json.dumps(params, sort_keys=True),
                    positive_train=int(y.sum()), positive_test=int(ytest.sum()), selected_raw_baseline=chosen, fit_warning_count=0)
        rows.append(qrow)
        for name, fit in fits.items():
            test = Ptest if name == "Fourier+pairwise SVM" else (Ftest if name in [SAME_MODEL, LINEAR_MODEL] else (Atest if name == ABLATION else Xtest))
            p = fit.predict(test)
            predictions[f"n{n}_{name}"] = p
            rows.append(dict(qrow, model=name, test_acc=accuracy_score(ytest, p), inner_cv_acc=fit.best_score_,
                             best_params=json.dumps(fit.best_params_, sort_keys=True), fit_warning_count=len(fit.recorded_warnings_)))
            warning_records.extend(dict(n_train=n, model=name, **w) for w in fit.recorded_warnings_)
            if name == SAME_MODEL:
                if params != fit.best_params_ or not np.array_equal(pred, p):
                    raise AssertionError("Precomputed projected kernel and direct same-feature RBF must agree")
        selected_row = next(r for r in rows[-len(fits):] if r['model'] == chosen)
        rows.append(dict(selected_row, model="Classical raw-input selection"))
        print(f"seed={seed} n={n}: quantum={qrow['test_acc']:.3f}; train-selected {chosen}={selected_row['test_acc']:.3f}; elapsed={time.perf_counter()-start:.1f}s", flush=True)
    np.savez_compressed(output / f"predictions_seed{seed}.npz", y_test=ytest, **predictions)
    (output / f"warnings_seed{seed}.json").write_text(json.dumps(warning_records, indent=2))
    return rows


def aggregate(out, rows):
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "scores.csv", index=False)
    summary, pairs = [], []
    for (n, name), group in frame.groupby(['n_train', 'model']):
        count, mean = len(group), group.test_acc.mean()
        sd = group.test_acc.std(ddof=1) if count > 1 else float('nan')
        radius = float(t.ppf(.975, count-1) * sd / np.sqrt(count)) if count > 1 else float('nan')
        summary.append(dict(n_train=n, model=name, runs=count, mean_acc=mean, sd_acc=sd,
                            ci95_low=mean-radius, ci95_high=mean+radius))
    for n, group in frame.groupby('n_train'):
        pivot = group.pivot(index='seed', columns='model', values='test_acc')
        for name in pivot.columns:
            if name == QMODEL:
                continue
            delta = pivot[QMODEL] - pivot[name]
            count, mean = len(delta), delta.mean()
            sd = delta.std(ddof=1) if count > 1 else float('nan')
            radius = float(t.ppf(.975, count-1)*sd/np.sqrt(count)) if count > 1 else float('nan')
            pairs.append(dict(n_train=n, comparator=name, runs=count, mean_delta=mean, sd_delta=sd,
                              ci95_low=mean-radius, ci95_high=mean+radius,
                              wins=int((delta>1e-12).sum()), ties=int((abs(delta)<=1e-12).sum()),
                              losses=int((delta < -1e-12).sum())))
    pd.DataFrame(summary).to_csv(out / 'summary.csv', index=False)
    pd.DataFrame(pairs).to_csv(out / 'paired_comparisons.csv', index=False)
    return frame


def freeze():
    if PROTOCOL.exists():
        raise FileExistsError("Protocol already frozen. Reuse it; do not silently replace after viewing results.")
    calibration = draw_inputs(314159, 1024, 0)
    F = projected_features(calibration)
    threshold = float(np.median(F @ teacher_weights()))
    grids = {name: grid for name, (_, grid) in model_specs(0).items()}
    spec = dict(task="fixed_entangling_circuit_teacher_v1", frozen_at_utc=utc_now(),
                teacher_seed=TEACHER_SEED, qubits=N_QUBITS, reps=REPS, input_frequency=FREQUENCY,
                teacher_weights=teacher_weights().tolist(), label_threshold=threshold,
                input_distribution="iid Uniform(-pi, pi)^6", label_noise=0.,
                calibration=dict(seed=314159, stream=0, n=1024, use="fixed global median threshold only"),
                evaluation_seeds=list(range(1000,1010)), training_sizes=[32,64,128,256], n_test=512,
                train_stream=10, test_stream=20, inner_folds=3, selection_metric="accuracy",
                baseline_grids=grids, projected_grid=dict(C=C_GRID, gamma=GAMMA_GRID),
                source_sha256=source_hashes(),
                primary_comparison=dict(n_train=128, comparator="Classical raw-input selection",
                  reason="Predeclared small-data target; comparator selected by training-only inner CV"),
                uncertainty="95% Student-t intervals over 10 independently generated datasets, conditional on one fixed teacher; other sizes/comparators secondary and unadjusted",
                disclosure="Task engineered to favor the chosen representation. Exact classical statevector simulation; no computational advantage. Same-feature classical model must match the projected kernel.")
    PROTOCOL.parent.mkdir(exist_ok=True)
    PROTOCOL.write_text(json.dumps(spec, indent=2, sort_keys=True) + '\n')
    print(f"Frozen {PROTOCOL.name}, SHA256={sha(PROTOCOL)}", flush=True)


def verify_protocol():
    spec = json.loads(PROTOCOL.read_text())
    if spec['source_sha256'] != source_hashes():
        raise ValueError("Scientific code differs from frozen protocol. Preserve results and define a new experiment.")
    return spec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['pilot','freeze','evaluate'], required=True)
    p.add_argument('--overwrite', action='store_true', help='Explicitly rerun the unchanged frozen evaluation')
    a = p.parse_args()
    if a.stage == 'freeze':
        freeze()
        return
    if a.stage == 'pilot':
        # Separate pilot data are never part of the final evaluation.
        # Calibrate the teacher threshold independently, as in the final protocol.
        F = projected_features(draw_inputs(314159, 1024, 0))
        threshold = float(np.median(F @ teacher_weights()))
        output = OUT / 'pilot'
        rows = evaluate_dataset(101, [128], 256, threshold, output)
        aggregate(output, rows)
        return
    spec = verify_protocol()
    if (OUT / 'scores.csv').exists() and not a.overwrite:
        raise FileExistsError("Evaluation exists. Use --overwrite only to reproduce the unchanged frozen protocol.")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'execution.json').write_text(json.dumps(dict(started_at_utc=utc_now(), protocol_sha256=sha(PROTOCOL),
        source_sha256=source_hashes(), python=sys.version, platform=platform.platform()), indent=2))
    t0, rows = time.perf_counter(), []
    for seed in spec['evaluation_seeds']:
        rows.extend(evaluate_dataset(seed, spec['training_sizes'], spec['n_test'], spec['label_threshold'], OUT))
        aggregate(OUT, rows)
    record = json.loads((OUT / 'execution.json').read_text())
    record.update(completed_at_utc=utc_now(), elapsed_seconds=time.perf_counter()-t0,
                  datasets=len(spec['evaluation_seeds']), score_rows=len(rows))
    (OUT / 'execution.json').write_text(json.dumps(record, indent=2))
    print(pd.read_csv(OUT / 'summary.csv').to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
