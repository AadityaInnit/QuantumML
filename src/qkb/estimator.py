"""scikit-learn-compatible quantum-kernel classifier.

Follows the BaseEstimator/ClassifierMixin convention used by benchmark suites such as
XanaduAI/qml-benchmarks, so it drops into GridSearchCV, cross_val_score and Pipelines.
Inputs are expected to be rotation angles (e.g. MinMax-scaled to [0, pi]).
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC

from .qkernel import feature_map, exact_kernel


class QuantumKernelSVC(BaseEstimator, ClassifierMixin):
    def __init__(self, fmap="Z", reps=1, bandwidth=0.25, C=1.0):
        self.fmap = fmap
        self.reps = reps
        self.bandwidth = bandwidth
        self.C = C

    def _fm(self, n_features):
        return feature_map(self.fmap, n_features, self.reps)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.X_train_ = X
        self.fm_ = self._fm(X.shape[1])
        K = exact_kernel(self.fm_, X, bandwidth=self.bandwidth)
        self.svc_ = SVC(kernel="precomputed", C=self.C).fit(K, y)
        self.classes_ = self.svc_.classes_
        return self

    def _cross(self, X):
        return exact_kernel(self.fm_, np.asarray(X, dtype=float), self.X_train_, bandwidth=self.bandwidth)

    def predict(self, X):
        return self.svc_.predict(self._cross(X))

    def decision_function(self, X):
        return self.svc_.decision_function(self._cross(X))
