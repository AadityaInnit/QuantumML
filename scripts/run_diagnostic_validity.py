"""Do cheap, training-only kernel diagnostics predict CV accuracy?

Reads the saved sweeps (no new quantum runs). Because bandwidth drives both the
diagnostics and accuracy, we report pooled Spearman AND within-bandwidth Spearman.
Exploratory: 36 (breast cancer) / 24 (circles) correlated configs, one seed.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr

DIAG = ["alignment", "offdiag_var", "geo_diff_ridge_0_001", "ridge_complexity"]
rows = []
for d in ["breast_cancer_4q", "circles_2q"]:
    s = pd.read_csv(f"results/{d}/qkernel_sweep.csv")
    for c in DIAG:
        pooled = spearmanr(s[c], s.cv_acc)[0]
        within = [spearmanr(g[c], g.cv_acc)[0] for _, g in s.groupby("bandwidth") if g[c].nunique() > 2]
        rows.append(dict(dataset=d, diagnostic=c, n_configs=len(s), pooled_spearman=pooled,
                         within_bandwidth_mean_spearman=float(np.nanmean(within))))
out = pd.DataFrame(rows).round(3)
out.to_csv("results/diagnostic_validity.csv", index=False)
print(out.to_string(index=False))
