"""Regenerate all figures from CSVs; no model fitting or heldout-based selection."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

COLORS = {"Z": "#117C83", "ZZ": "#CE6E38", "ZZfull": "#7355A1"}
plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 180, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(Path("figures") / f"{name}.png", bbox_inches="tight")
    fig.savefig(Path("figures") / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def main():
    Path("figures").mkdir(exist_ok=True)
    root = Path("results/breast_cancer_4q")
    b = pd.read_csv(root / "benchmark.csv")
    labels = ["RBF SVM", "Linear SVM", "Logistic", "Random forest", "MLP", "Quantum SVM"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(b))
    ax.errorbar(x-.08, b.cv_acc, yerr=b.cv_std, fmt='o', capsize=4, label="Nested CV mean ± fold SD", color="#117C83")
    ax.scatter(x+.08, b.test_acc, marker="D", label="Descriptive holdout accuracy (48 points)", color="#CE6E38")
    ax.set(xticks=x, xticklabels=labels, ylim=(.65, 1.02), ylabel="Accuracy", title="Shared folds, preprocessing and hyperparameters fit inside CV")
    ax.legend(fontsize=8, loc="lower left")
    save(fig, "benchmark")
    sw = pd.read_csv(root / "qkernel_sweep.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), sharey=True)
    for ax, (kind, color) in zip(axes, COLORS.items()):
        a = sw[sw.fmap == kind]
        for reps in [1, 2, 3]:
            z = a[a.reps == reps].sort_values("bandwidth")
            ax.plot(z.bandwidth, z.cv_acc, marker="o", label=f"reps {reps}", alpha=.45+.15*reps, color=color, linestyle=["-", "--", ":"][reps-1])
        ax.set(title=kind, xlabel="Bandwidth (log scale)", xscale="log", ylim=(.45, 1.02))
        ax.set_xticks([.25, .5, 1, 2], [".25", ".5", "1", "2"])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Exploratory fixed-map outer CV accuracy")
    fig.suptitle("Depth and bandwidth matter; these curves do not estimate the selected model", fontsize=11)
    save(fig, "depth_bandwidth")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for kind, color in COLORS.items():
        z = sw[sw.fmap == kind]
        axes[0].scatter(z.offdiag_var, z.cv_acc, label=kind, color=color)
        axes[1].scatter(z.alignment, z.cv_acc, label=kind, color=color)
    axes[0].set(xlabel="Off-diagonal kernel variance", ylabel="Exploratory fixed-map CV accuracy")
    axes[1].set(xlabel="Centered kernel–target alignment (training set)")
    axes[1].legend()
    fig.suptitle("Descriptive associations across 36 configurations, not causal evidence", fontsize=11)
    save(fig, "diagnostics")
    sensitivity = pd.read_csv(root / "geometric_sensitivity.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    for kind, color in COLORS.items():
        z = sensitivity[(sensitivity.fmap == kind) & (sensitivity.reps == 2) & (sensitivity.bandwidth == .25)]
        ax.loglog(z.reg, z.geometric_difference, marker="o", color=color, label=f"{kind}, reps=2, bw=.25")
    ax.set(xlabel="Ridge λ after trace-n normalization", ylabel="gλ(C ∥ Q)", title="Geometry depends strongly on ridge and the chosen RBF comparator")
    ax.legend(fontsize=8)
    save(fig, "geometric_sensitivity")
    concentration = pd.read_csv("results/concentration/qubit_sweep.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, bw in zip(axes, [.25, 1.]):
        for kind, color in COLORS.items():
            z = concentration[(concentration.fmap == kind) & (concentration.bandwidth == bw)]
            ax.semilogy(z.qubits, z.offdiag_var, 'o-', label=kind, color=color)
        ax.set(title=f"Bandwidth {bw}, reps=2", xlabel="Qubits / PCA components", xticks=[2, 4, 6, 8])
    axes[0].set_ylabel("Off-diagonal variance (training data)")
    axes[1].legend()
    fig.suptitle("Qubit sweep also changes the PCA representation; association only", fontsize=11)
    save(fig, "concentration_qubits")
    noise = pd.read_csv("results/noise/noise_study.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for (kind, reps), group in noise.groupby(["fmap", "reps"]):
        z = group[group.level == "finite shots"].groupby("shots").err.agg(["mean", "std"]).reset_index()
        axes[0].errorbar(z.shots, z['mean'], yerr=z['std'], marker='o', capsize=3, label=f"{kind} reps={reps}")
    axes[0].set(xscale="log", xlabel="Shots", ylabel="Relative Frobenius error (training Gram)", title="Finite shots: mean ± SD over 5 draws")
    axes[0].legend(fontsize=8)
    z = noise[noise.level == "fake_sherbrooke noise"]
    ix = np.arange(len(z))
    axes[1].bar(ix-.17, z.err, .34, label="Training Gram")
    axes[1].bar(ix+.17, z.cross_err, .34, label="Test/train cross kernel")
    axes[1].set(xticks=ix, xticklabels=[f"{r.fmap}, reps={r.reps}" for r in z.itertuples()],
                ylabel="Relative Frobenius error", title="Fake Sherbrooke: one realization, 1024 shots")
    axes[1].legend(fontsize=8)
    save(fig, "noise_degradation")
    control = pd.read_csv("results/circles_2q/benchmark.csv")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.errorbar(np.arange(len(control)), control.cv_acc, yerr=control.cv_std, fmt='o', capsize=4)
    ax.set(xticks=np.arange(len(control)), xticklabels=labels, ylim=(.3, 1.05), ylabel="Nested CV accuracy ± fold SD",
           title="Circles: a classical nonlinear control, not an engineered quantum task")
    save(fig, "circles_control")


if __name__ == "__main__":
    main()
