"""Training-set diagnostics; geometric difference follows Huang et al. Eq. (5)."""
import numpy as np
from scipy.linalg import eigh


def normalized_psd(K):
    K = np.asarray(K, dtype=float)
    if K.ndim != 2 or K.shape[0] != K.shape[1] or not np.isfinite(K).all():
        raise ValueError("Kernel must be a finite square matrix")
    if not np.allclose(K, K.T, atol=1e-9):
        raise ValueError("Kernel must be symmetric")
    K = (K + K.T) / 2
    if np.trace(K) <= 0:
        raise ValueError("Kernel trace must be positive")
    K = K * len(K) / np.trace(K)
    vals, vecs = eigh(K)
    if vals[0] < -1e-8 * max(1, vals[-1]):
        raise ValueError("Kernel must be positive semidefinite")
    return (vecs * np.maximum(vals, 0)) @ vecs.T


def concentration(K):
    """Off-diagonal mean and population variance; low variance alone is not proof."""
    values = np.asarray(K)[~np.eye(len(K), dtype=bool)]
    if not values.size:
        raise ValueError("Need at least two samples")
    return float(values.mean()), float(values.var())


def target_alignment(K, y):
    """Centered Frobenius alignment, with BOTH kernel and label kernel centered."""
    y = 2 * np.asarray(y, dtype=float) - 1
    y = y - y.mean()
    K = np.asarray(K)
    Kc = K - K.mean(0)[None, :] - K.mean(1)[:, None] + K.mean()
    Yc = np.outer(y, y)
    den = np.linalg.norm(Kc) * np.linalg.norm(Yc)
    return float(np.sum(Kc * Yc) / den) if den > 1e-14 else 0.0


def geometric_difference(K_q, K_c, reg=1e-3):
    """Return g_lambda(C||Q) = sqrt(lambda_max(Q, C + lambda I)).

    Both input Gram matrices are first normalized to trace n. At lambda=0
    this is Huang et al. (2021), Eq. (5), for positive-definite C. Positive
    lambda gives an explicitly ridge-stabilized diagnostic, not their full
    regularized prediction-error bound. Singular C requires positive ridge.
    There is NO universal sqrt(n) ceiling: with ridge, g <= sqrt(n/lambda).
    Large g for one RBF comparator is not evidence of quantum advantage.
    """
    if not np.isfinite(reg) or reg < 0:
        raise ValueError("reg must be finite and nonnegative")
    Q, C = normalized_psd(K_q), normalized_psd(K_c)
    if Q.shape != C.shape:
        raise ValueError("Kernel shapes must match")
    if reg == 0 and np.linalg.eigvalsh(C)[0] <= 1e-12:
        raise ValueError("Singular comparator: use a positive ridge")
    largest = eigh(Q, C + reg * np.eye(len(C)), subset_by_index=[len(C)-1, len(C)-1], eigvals_only=True)[0]
    return float(np.sqrt(max(largest, 0)))


def model_complexity(K, y, reg=1e-3):
    """Ridge inverse quadratic form y.T (K + lambda I)^-1 y.

    This is a descriptive surrogate, not the full regularized bound in the paper.
    """
    yy = 2 * np.asarray(y) - 1.0
    Kn = normalized_psd(K)
    return float(yy @ np.linalg.solve(Kn + reg * np.eye(len(Kn)), yy))
