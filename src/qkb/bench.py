"""Nested selection using shared outer/inner folds and fold-local preprocessing."""
import itertools
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from qkb.data import preprocessor
from qkb.qkernel import feature_map, exact_kernel

C_GRID = [0.1, 1, 10, 100]
CONFIGS = list(itertools.product(["Z", "ZZ", "ZZfull"], [1, 2, 3], [0.25, 0.5, 1., 2.]))


def classical_models(seed=0):
    return {
        "SVM-RBF": (SVC(kernel="rbf"), {"C": C_GRID, "gamma": [0.01, 0.1, 0.3, 1, "scale"]}),
        "SVM-linear": (SVC(kernel="linear"), {"C": C_GRID}),
        "LogReg": (LogisticRegression(max_iter=2000), {"C": C_GRID}),
        "RandomForest": (RandomForestClassifier(random_state=seed), {"n_estimators": [100], "max_depth": [3, None]}),
        "MLP": (MLPClassifier(max_iter=2000, random_state=seed), {"hidden_layer_sizes": [(8,), (16, 8)], "alpha": [1e-4, 1e-2]}),
    }


def inner_folds(X, y, seed=0):
    return list(StratifiedKFold(3, shuffle=True, random_state=seed).split(X, y))


def fit_classical(X, y, est, grid, n_features, folds, seed=0):
    pipe = Pipeline([("prep", preprocessor(n_features, seed)), ("model", est)])
    return GridSearchCV(pipe, {f"model__{k}": v for k, v in grid.items()}, cv=folds,
                        scoring="accuracy", n_jobs=1).fit(X, y)


def prepare_inner(X, y, n_features, folds, seed=0):
    prepared = []
    for tr, va in folds:
        prep = preprocessor(n_features, seed)
        prepared.append((prep.fit_transform(X[tr]), prep.transform(X[va]), y[tr], y[va]))
    return prepared


def tune_quantum(prepared, configs, n_features):
    """Tune map, depth, bandwidth AND C only on supplied inner folds."""
    scores = np.zeros((len(configs), len(C_GRID)))
    for i, (kind, reps, bw) in enumerate(configs):
        fm = feature_map(kind, n_features, reps)
        for Xtr, Xva, ytr, yva in prepared:
            Ktr = exact_kernel(fm, Xtr, bandwidth=bw)
            Kva = exact_kernel(fm, Xva, Xtr, bandwidth=bw)
            for j, c in enumerate(C_GRID):
                pred = SVC(kernel="precomputed", C=c).fit(Ktr, ytr).predict(Kva)
                scores[i, j] += accuracy_score(yva, pred) / len(prepared)
    # Deterministic tie break: declared config order then ascending C.
    best_c = scores.argmax(axis=1)
    best_i = int(scores[np.arange(len(configs)), best_c].argmax())
    return best_i, best_c, scores


def quantum_predict(config, c, Xtr, ytr, Xte):
    kind, reps, bw = config
    fm = feature_map(kind, Xtr.shape[1], reps)
    Ktr = exact_kernel(fm, Xtr, bandwidth=bw)
    Kte = exact_kernel(fm, Xte, Xtr, bandwidth=bw)
    return SVC(kernel="precomputed", C=c).fit(Ktr, ytr).predict(Kte)


def metrics(y, pred):
    return dict(acc=float(accuracy_score(y, pred)), f1=float(f1_score(y, pred)))
