# Label-Efficient Comparison of Shared Logistic Models with Finite-Sample Guarantees

**Label-Efficient Comparison (LEC)** acquires training and evaluation labels in batches and stops once they support a reliable model choice.

[Algorithm 1](docs/ALGORITHM.md) · [Population extension: Algorithm 2](docs/POPULATION.md) · [Results](docs/RESULTS.md) · [Reproduction](docs/REPRODUCING.md) · [中文算法说明](docs/ALGORITHM_ZH.md)

Model selection can consume labels beyond those needed for the final fit. In a multi-label task, each object also has several attributes that can be annotated separately. LEC compares two shared logistic training rules **before acquiring all labels in a prescribed cross-validation (CV) plan**. Public features determine a weighted sum of label products and an approximation error bound. Each draw queries at most two labels, and a shared cache avoids repeat purchases.

Algorithm 1 has three stages: **label-free preparation, sequential label queries, and selection with final training**. Its [14-step pseudocode](docs/ALGORITHM.md#queries-and-decisions) and [LaTeX source](docs/ALGORITHM1.tex) follow the manuscript.

The main method starts at a zero anchor and needs **no pilot labels**. It uses a Clopper--Pearson (CP) exact binomial interval at each prespecified inspection, with error probability `delta/J` per look. Adding the approximation error bound gives simultaneous coverage of the exact CV score difference with probability at least `1-delta` across the prescribed inspections. The interval supports a choice when its upper endpoint is at most `g`, or its lower endpoint is at least `-g`. Unresolved trials complete exact CV through the same cache before final training.

## Two comparison targets

| Target | Procedure | Entry point |
|---|---|---|
| Fixed CV plan | Public zero anchor, feature-based paired bound, exact binomial intervals, and exact-CV fallback | **Algorithm 1**, implemented by `whichlabels.compare` |
| Expected test risk over random training sets | Independent pilot and calibration, prediction-assisted queries, bounded-variable intervals, and paired-refit fallback | **Algorithm 2**, described in [POPULATION.md](docs/POPULATION.md); experiment scripts are in the companion archive |

The paper also derives necessary label-information bounds for fixed feature designs. Sensitivity, dispersion, and approximation conditions yield matching inverse-square dependence on tolerance, with a logarithmic confidence factor. The sequential algorithm uses `log(2J/delta)` in place of `log(1/delta)`, where `J` is the number of inspections. See [the theory map and results](docs/RESULTS.md).

## Manuscript results

The current manuscript studies simulations, repeated random splits, full validation coverage at a fixed training size, and population program risk.

- **Repeated splits:** mean total labels fall by **58.1%–73.8%** for images and **64.6%** for text relative to sequential CV with active attribute evaluation.
- **Full validation coverage:** every object enters validation once. Mean total labels fall by **70.5%–83.2%** relative to full annotation and **69.6%–80.9%** relative to active CV.
- **Complete accounting:** counts include distinct labels acquired for preparation, comparison, fallback, and final training. The full-coverage LEC runs return acceptable choices on **238/240** paths at `delta = 0.05`.

[RESULTS.md](docs/RESULTS.md) gives the configurations, complete counts, decision outcomes, and corresponding frozen summaries. These are the paper's recorded experiments; the small demo below illustrates the API.

Documentation follows the 50-page manuscript revised on **12 November 2026**. The executable package remains **0.2.1**; this documentation update does not change its sampling or decision rules.

## Installation

Python 3.10 or later is required. From this directory:

```bash
python -m venv .venv
# Activate .venv using your shell, then:
python -m pip install -e .
```

The main package depends only on NumPy and SciPy. To run the archived paper scripts and all regression checks:

```bash
python -m pip install -e ".[reproduce]"
```

## A complete example

```bash
python -m whichlabels.demo --output demo_output
python -m unittest discover -s tests -v
```

The example generates correlated binary attributes and compares:

1. **LEC** (recorded as selective moments in the demo output): the manuscript's Algorithm 1.
2. **Active CV:** complete training labels, selective evaluation attributes, valid sequential intervals, label reuse, and early stopping.

It writes `results.json`, `summary.csv`, and `progress.json`. Results include the supporting interval, distinct label counts by stage, fallback, final fitted coefficients, and measured computation time. A separate full-label audit runs **after** both decisions. The example is an executable illustration, not additional evidence for the paper's reported percentages.

## Use your own label provider

```python
from whichlabels import prepare_comparison, compare, LabelCache

# a, b: arrays of shape (objects, attributes, coefficients).
# plan: fixed list of (training_object_ids, evaluation_object_ids).
# Each feature row has Euclidean norm at most one.
prepared = prepare_comparison(a, b, plan, penalty=0.5)

def acquire(object_ids, attribute_ids):
    # Return one binary label for every requested object-attribute pair.
    return annotation_service(object_ids, attribute_ids)

cache = LabelCache(a.shape[0], a.shape[1], acquire)
result = compare(
    prepared, cache, final_indices,
    tolerance=0.005, delta=0.05, seed=17,
)
print(result['choice'], result['interval'], result['total_labels'])
```

`annotation_service`, the feature arrays, the fixed plan, and `final_indices` are supplied by the caller. The provider is called only for distinct, unbought labels. For a labeled dataset replay, `ArrayLabelProvider(y)` supplies the same interface.

The final subset has the same training size as the CV splits and must be fixed before acquisition. Both candidates share coefficient coordinates and dimensions. All coefficients, including any attribute intercepts, receive the L2 penalty.

## What the decision means

The target is the exact mean paired binary cross-entropy (BCE) difference on the **fixed finite CV plan**, conditional on the pool, its labels, and its features. Candidate A is acceptable when its score is at most `tolerance` above B's; B is acceptable under the symmetric condition. The probability statement concerns the randomized queries. Algorithm 2 instead averages test risk over random training sets.

With natural logarithms, tolerance `g` means that the selected candidate's geometric-mean probability assigned to the observed validation labels is at least `exp(-g)` times the better candidate's. Thus `g = 0.005` permits a relative decrease of at most **0.499%** in that score.

The approximation is not an unbiased estimator of the exact logistic score. Sampled terms are unbiased for the **approximate** score, and the interval includes its feature-based error bound. Large error bounds can require exact-CV fallback. See [the algorithm guide](docs/ALGORITHM.md) for the formulas and [scope and numerical checks](docs/SCOPE.md) for implementation conditions.

## Layout

| Path | Purpose |
|---|---|
| `src/whichlabels/api.py` | Prepare, query, stop, fall back, and train the selected candidate |
| `src/whichlabels/labels.py` | Callback-based label acquisition and distinct-label accounting |
| `src/whichlabels/_bounds.py` | Feature-only paired and separate approximation bounds |
| `src/whichlabels/_moments.py` | Sparse label polynomial and fixed sampling probabilities |
| `src/whichlabels/_intervals.py` | Exact binomial endpoints, numerical guard, and stopping |
| `src/whichlabels/_active.py` | Same-permission active-CV comparator |
| `tests/` | Algebra, cache, decision branches, and archived-code regression checks |
| `paper_reproduction/` | Original scripts with their relative imports preserved |
| `paper_evidence/` | Compact frozen summaries and input-cache hashes |
| `source_provenance.json` | Function-level provenance of extracted implementations |

The private modules retain the public-feature and fitting implementations; `_intervals.py` supplies the CP revision. The public API adds input validation, a real acquisition callback, explicit numerical-residual checks, and one complete decision function. `paper_reproduction` preserves the original experiment scripts separately so the package refactoring does not replace the frozen research record.

## Reproduce the manuscript

See [REPRODUCING.md](docs/REPRODUCING.md). The controlled simulation is self-contained. GoEmotions and ADE20K reruns require the documented prepared feature/label caches. Those datasets, model weights, and hidden label arrays are not bundled in this code repository.

The revised experiment index is `paper_reproduction/cp_revision_20261106/README.txt`. The current main simulation uses `Simulation_Refinement_20261106.zip`. Real-data and extension runners use `CP_Revision_20261106.zip`. Each companion archive includes its execution commands and full result records. This keeps the installable package small. The older `paper_reproduction/simulation`, `paper_reproduction/main_cv`, and corresponding `paper_evidence` folders preserve the empirical-Bernstein (EB) version as historical evidence. Their percentages are not the CP revision results.

The CP update keeps sampling probabilities, batches, inspection times, approximation bounds, caching, fallback, and final training unchanged. The active-CV comparator retains its original confidence sequences. Uniform-term and general prediction-assisted statistics retain their applicable bounded-variable intervals. Complete-label counts include final training; coverage and acceptable choices are reported separately.

## Citation and license

The manuscript title is **Label-Efficient Comparison of Shared Logistic Models with Finite-Sample Guarantees**. The repository name, Python package name (`which-labels-logistic`), and import name (`whichlabels`) remain stable.

The [citation template](docs/CITATION.cff.example) contains the current title and package version; author names and a public manuscript link remain to be filled in. A code license has not yet been selected. See [the repository maintenance notes](docs/GITHUB_UPLOAD_ZH.md).
