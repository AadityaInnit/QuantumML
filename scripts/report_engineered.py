"""Validate and report the completed frozen experiment without fitting or selection."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/engineered'
FIG = ROOT / 'figures'
Q = 'Projected quantum kernel'
SELECTED = 'Classical raw-input selection'
SAME = 'Classical same-feature RBF'
LOG = 'Classical same-feature logistic'
ABLATION = 'Nonentangling projection RBF'
RAW = ['RBF SVM', 'MLP', 'Random forest', 'Gradient boosting', 'Fourier+pairwise SVM']
ORDER = [Q, SELECTED, *RAW, SAME, LOG, ABLATION]
LABELS = {Q:'Projected kernel', SELECTED:'Train-selected classical', SAME:'Same-feature classical RBF',
          LOG:'Same-feature logistic', ABLATION:'Nonentangling features', 'Fourier+pairwise SVM':'Fourier + pairwise SVM'}
COLORS = {Q:'#007E87', SELECTED:'#20262C', SAME:'#38444C', LOG:'#B38312', ABLATION:'#9E559A',
          'RBF SVM':'#3479AC', 'MLP':'#CE6653', 'Random forest':'#829837', 'Gradient boosting':'#936D58', 'Fourier+pairwise SVM':'#BF7B29'}
plt.rcParams.update({'figure.dpi':140, 'savefig.dpi':180, 'font.size':10,
                     'axes.spines.top':False, 'axes.spines.right':False})


def save(fig, name):
    fig.tight_layout()
    for ext in ['png','svg']:
        fig.savefig(FIG/f'{name}.{ext}', bbox_inches='tight')
    plt.close(fig)


def validate():
    protocol_path = ROOT/'experiments/engineered_protocol.json'
    spec = json.loads(protocol_path.read_text())
    execution = json.loads((OUT/'execution.json').read_text())
    assert execution['protocol_sha256'] == hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    for relative, expected in spec['source_sha256'].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() == expected
    assert execution.get('completed_at_utc'), 'Evaluation is incomplete'
    assert execution['started_at_utc'] > spec['frozen_at_utc']
    scores = pd.read_csv(OUT/'scores.csv')
    assert len(scores) == len(spec['evaluation_seeds'])*len(spec['training_sizes'])*len(ORDER)
    assert sorted(scores.seed.unique()) == spec['evaluation_seeds']
    assert sorted(scores.n_train.unique()) == spec['training_sizes']
    assert not scores.duplicated(['seed','n_train','model']).any()
    for seed in spec['evaluation_seeds']:
        dataset = np.load(OUT/f'dataset_seed{seed}.npz')
        pred = np.load(OUT/f'predictions_seed{seed}.npz')
        y = dataset['y_test']
        np.testing.assert_array_equal(y, pred['y_test'])
        assert set(map(tuple,dataset['X_train_pool'])).isdisjoint(set(map(tuple,dataset['X_test'])))
        for n in spec['training_sizes']:
            group = scores[(scores.seed==seed)&(scores.n_train==n)].set_index('model')
            for name in ORDER:
                chosen = group.loc[name,'selected_raw_baseline'] if name==SELECTED else name
                key = f'n{n}_quantum' if chosen==Q else f'n{n}_{chosen}'
                np.testing.assert_allclose(group.loc[name,'test_acc'], (pred[key]==y).mean())
            np.testing.assert_array_equal(pred[f'n{n}_quantum'], pred[f'n{n}_{SAME}'])
            raw = group.loc[RAW]
            # Serialized baseline selection is a training-CV argmax, never a test argmax.
            assert group.loc[Q,'selected_raw_baseline'] == raw.inner_cv_acc.idxmax()
    return spec, execution, scores


def main():
    FIG.mkdir(exist_ok=True)
    spec, execution, scores = validate()
    summary = pd.read_csv(OUT/'summary.csv')
    pairs = pd.read_csv(OUT/'paired_comparisons.csv')
    n = spec['primary_comparison']['n_train']
    primary = summary[summary.n_train==n].set_index('model')
    delta = pairs[(pairs.n_train==n)&(pairs.comparator==SELECTED)].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(12,4.8), sharey=True)
    for ax, names in zip(axes, [[Q,SELECTED,*RAW], [Q,SAME,LOG,ABLATION,SELECTED]]):
        for name in names:
            g = summary[summary.model==name].sort_values('n_train')
            style = '--' if name in [SAME,SELECTED] else '-'
            ax.plot(g.n_train,g.mean_acc,style,marker='o',label=LABELS.get(name,name),color=COLORS[name],
                    lw=2.4 if name==Q else 1.3, markersize=6 if name==Q else 4)
            if name in [Q,SELECTED,LOG,ABLATION]:
                ax.fill_between(g.n_train,g.ci95_low,g.ci95_high,alpha=.12,color=COLORS[name])
        ax.set(xlabel='Training examples (same dataset prefixes)',xscale='log',ylim=(.40,1.02))
        ax.set_xticks(spec['training_sizes'], [str(k) for k in spec['training_sizes']])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.legend(fontsize=8,loc='center right')
    axes[0].set(ylabel='Fresh-test accuracy, mean across 10 independent datasets',title='Against specified raw-input baselines')
    axes[1].set_title('Controls identify a representation benefit')
    fig.suptitle('Disclosed circuit-teacher task • shading: 95% intervals over datasets • one fixed teacher',fontsize=11)
    save(fig,'engineered_learning_curves')
    fig, axes = plt.subplots(1,2,figsize=(12,5.5))
    order = [Q,SELECTED,*RAW,SAME,LOG,ABLATION]
    vals = primary.loc[order]
    for i,(name,r) in enumerate(vals.iterrows()):
        axes[0].errorbar(r.mean_acc,i,xerr=[[r.mean_acc-r.ci95_low],[r.ci95_high-r.mean_acc]],
                         fmt='o',color=COLORS[name],capsize=3)
    axes[0].set(yticks=np.arange(len(order)),yticklabels=[LABELS.get(k,k) for k in order],
                xlabel='Mean fresh-test accuracy ± 95% interval',xlim=(.4,1.02),title=f'Predeclared training size: {n}')
    axes[0].invert_yaxis()
    pivot=scores[scores.n_train==n].pivot(index='seed',columns='model',values='test_acc')
    d=pivot[Q]-pivot[SELECTED]
    axes[1].scatter(np.arange(len(d)),d,color=COLORS[Q],s=40)
    axes[1].axhline(0,color='#555555',lw=1)
    axes[1].axhline(delta.mean_delta,color=COLORS[Q],linestyle='--',label='Mean paired gap')
    axes[1].axhspan(delta.ci95_low,delta.ci95_high,color=COLORS[Q],alpha=.15,label='95% interval over datasets')
    axes[1].set(xticks=np.arange(len(d)),xticklabels=[str(k) for k in d.index],
                xlabel='Independent dataset seed',ylabel='Projected − train-selected classical accuracy',
                title='Paired gap on identical fresh test examples')
    axes[1].tick_params(axis='x',rotation=45)
    axes[1].legend(fontsize=8,loc='lower right')
    save(fig,'engineered_primary_comparison')
    table='\n'.join(f'| {name} | {primary.loc[name,"mean_acc"]:.3f} | [{primary.loc[name,"ci95_low"]:.3f}, {primary.loc[name,"ci95_high"]:.3f}] |' for name in order)
    warning_counts=scores.groupby('model').fit_warning_count.sum().astype(int).to_dict()
    warning_text=', '.join(f'{name}: {count}' for name,count in warning_counts.items() if count) or 'none'
    best_raw=float(primary.loc[RAW,'mean_acc'].max())
    outcome='positive' if delta.ci95_low>0 else 'inconclusive'
    (OUT/'validation.json').write_text(json.dumps(dict(protocol_and_source_hashes_verified=True,
        frozen_before_execution=True,datasets_verified=10,training_sizes_verified=4,score_rows_verified=len(scores),
        every_prediction_score_recomputed=True,same_feature_predictions_identical=True,
        classical_selection_verified_training_only=True,primary_outcome=outcome,fit_warning_counts=warning_counts),indent=2))
    report=f'''# Engineered task: a reproducible representation win

This is a disclosed synthetic task engineered to favor a fixed quantum-circuit
representation. It establishes no computational quantum advantage. All circuit
features are obtained by exact **classical** statevector simulation.

## Predeclared primary result

At **{n} training examples**, the projected kernel reaches
**{primary.loc[Q,'mean_acc']:.3f}** mean accuracy on fresh test data. The classical
baseline selected exclusively by training CV reaches **{primary.loc[SELECTED,'mean_acc']:.3f}**.
The paired gap is **{delta.mean_delta:+.3f}**, with a 95% interval
**[{delta.ci95_low:+.3f}, {delta.ci95_high:+.3f}]** across ten independently
generated datasets ({delta.wins} wins, {delta.ties} ties, {delta.losses} losses).
The largest mean among the five individual raw-input baselines is {best_raw:.3f};
that maximum is descriptive and was not used for the primary comparator selection.

| Model | Mean accuracy, n={n} | 95% interval over datasets |
|---|---:|---:|
{table}

![Primary comparison](figures/engineered_primary_comparison.png)

The **classical same-feature RBF model makes identical predictions** to the
projected quantum kernel in every evaluated case. The classical same-feature
logistic model sees the exact observables used by the linear teacher and is an
informed control, not an independent raw-input competitor. These controls show
that the win is due to a favorable representation; the learning algorithm after
feature extraction is classical.

## Frozen task and evaluation

- Six input coordinates are drawn iid uniformly from [-π, π]. The fixed circuit
  has six qubits and two encoding/entangling layers. Each layer applies
  Ry(2x_q + fixed offset), Rz(x_neighbor + fixed offset), a CNOT chain and its
  closing CNOT. Offset seed is 271828; teacher-weight seed is 271829.
- The feature vector consists of local X, Y and Z expectation values: 18 real
  numbers. Labels threshold a fixed weighted sum of those features. There is
  no label noise or post-hoc rejection of difficult examples. A separate
  1024-point calibration draw fixes the median threshold before evaluation.
- The projected kernel is exp(-γ||F(x)-F(x')||²). Local Pauli features encode the
  single-qubit reduced density matrices; the squared feature distance equals
  twice the summed squared Frobenius distance between those matrices, with that
  factor absorbed by γ. The implementation is inspired by projected kernels in
  [Huang et al. (2021)](https://www.nature.com/articles/s41467-021-22539-9),
  **not a replication of their engineered dataset or quantum-hardness result**.
- Ten fresh dataset seeds (1000–1009), each with training prefixes of
  32/64/128/256 and a separate fixed test set of 512. Every baseline shares
  these samples. Training and test use distinct deterministic RNG streams.
- Hyperparameters are selected by identical shuffled stratified 3-fold inner
  CV. Standardization for raw-input baselines is fit inside each training fold.
  Deterministic circuit and Fourier features fit no parameters on any data.
- Raw-input competitors: RBF SVM; a small MLP with Adam/L-BFGS and two architectures;
  random forest; histogram gradient boosting; and an SVM over coordinate
  Fourier frequencies 1–4 plus all pairwise products (1224 features).
- The nonentangling ablation retains encoding/offsets but removes CNOT gates.
  Same-feature RBF and logistic controls use the 18 exact entangling features.
- The **primary comparison was frozen at n=128** against a classical baseline
  chosen using inner CV, not test accuracy. Other training sizes and individual
  comparator intervals are secondary, unadjusted exploratory results.
- 95% intervals use Student-t uncertainty over the ten independent datasets,
  conditional on this one teacher. Sizes share training prefixes and test sets
  within a seed and must not be treated as independent additional runs.

![Learning curves and controls](figures/engineered_learning_curves.png)

## Provenance and limits

`experiments/engineered_protocol.json` was frozen at
`{spec['frozen_at_utc']}`. Evaluation started at
`{execution['started_at_utc']}`. Source and protocol hashes are checked before
running and independently checked by the reporting script. The execution record
contains the final completion time; the run took {execution['elapsed_seconds']:.1f}s.

The separate pilot uses seed 101 with 128/256 train/test examples. An initial
pilot exposed MLP convergence warnings; before freezing, its grid was expanded
to Adam plus L-BFGS and a larger iteration budget. Teacher, input distribution,
feature map and labels did not change. Initial and final pilot outputs are
preserved in `results/engineered/pilot_initial/` and `pilot/`. No evaluation
seed was used for those decisions. This is a local protocol freeze with saved
hashes, not a third-party timestamped preregistration.

All tuning warnings are saved by dataset. Across evaluation fits the recorded
counts are: {warning_text}. Counts include warnings from inner tuning fits;
they are not necessarily failures of the final selected estimator. Counts for
the classical-selection row reuse its chosen fit, rather than additional fits.
Finite grids
and optimizer budgets cannot establish superiority over all classical learners.

The task is intentionally favorable to its teacher representation. A classical
method that computes the same features can match it, and this repo does exactly
that. The disclosed teacher itself is also an exact classical labeling oracle at
this scale; the benchmark restricts learners to their specified inputs and training
examples. Only one teacher, six qubits, exact simulation and one input distribution
were studied. We measure no hardware/runtime advantage and make no claim about
ordinary real-world data. The earlier breast-cancer benchmark remains a useful
counterexample to assuming the representation will help everywhere.

## Reproduce and inspect

From the repo root in the environment from `requirements.txt`:

```bash
# The supplied protocol is already frozen. Do not refreeze after seeing results.
python scripts/run_engineered.py --stage evaluate --overwrite
python scripts/report_engineered.py
python scripts/build_notebooks.py --execute
PYTHONPATH=src python -m unittest discover -s tests -v
```

`--overwrite` reproduces the **unchanged** frozen protocol; scientific source
edits are rejected. To study another task, preserve this protocol/results and
create a separately versioned experiment rather than replacing the freeze.

- `scores.csv`: every model/dataset/training size and training-selected parameters.
- `summary.csv`: accuracy means and intervals; `paired_comparisons.csv`: paired gaps.
- `dataset_seed*.npz`: raw inputs, exact features, labels and ablation features.
- `predictions_seed*.npz`: predictions for independent score recomputation.
- `warnings_seed*.json`: recorded tuning warnings, without silently discarding runs.
- `execution.json` and `validation.json`: timing, hashes and integrity checks.
- `notebooks/04_engineered.ipynb`: executed report with paired plots and controls.

**Defensible claim:** On this disclosed circuit-teacher task, projected circuit
features improve prediction over the specified tuned raw-input baselines at
small training sizes. Classical learning with those same features matches or
exceeds the projected-kernel model. This is a representation result, not a
quantum computational speedup.
'''
    (ROOT/'ENGINEERED_RESULTS.md').write_text(report)
    print(f'Primary n={n}: {delta.mean_delta:+.4f}, CI [{delta.ci95_low:+.4f},{delta.ci95_high:+.4f}] ({outcome})')


if __name__=='__main__':
    main()
