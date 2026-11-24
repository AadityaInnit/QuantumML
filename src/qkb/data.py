"""Raw fixed holdout splits and fold-local feature preprocessing."""
import numpy as np
from sklearn.datasets import load_breast_cancer, make_circles
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline


def load_raw(name="breast_cancer", n_samples=160, test_size=0.3, seed=0):
    if name == "breast_cancer":
        X, y = load_breast_cancer(return_X_y=True)
    elif name == "circles":
        X, y = make_circles(n_samples=1000, noise=0.1, factor=0.4, random_state=seed)
    else:
        raise ValueError(name)
    if n_samples < 20 or n_samples % 2 or n_samples // 2 > np.bincount(y).min():
        raise ValueError("n_samples must be even, >=20, and fit both classes")
    rng = np.random.RandomState(seed)
    idx = np.hstack([rng.choice(np.where(y == c)[0], n_samples // 2, replace=False) for c in (0, 1)])
    ids_train, ids_test = train_test_split(idx, test_size=test_size, stratify=y[idx], random_state=seed)
    return X[ids_train], X[ids_test], y[ids_train], y[ids_test], ids_train, ids_test


def preprocessor(n_features=4, seed=0):
    # clip=False deliberately preserves out-of-training-range values at inference.
    return Pipeline([("standardize", StandardScaler()),
                     ("pca", PCA(n_components=n_features, random_state=seed)),
                     ("angles", MinMaxScaler(feature_range=(0, np.pi)))])


def load(name="breast_cancer", n_features=4, n_samples=160, test_size=0.3, seed=0):
    """Convenience holdout transform. Use load_raw for nested CV."""
    Xtr, Xte, ytr, yte, _, _ = load_raw(name, n_samples, test_size, seed)
    prep = preprocessor(2 if name == "circles" else n_features, seed)
    return prep.fit_transform(Xtr), prep.transform(Xte), ytr, yte
