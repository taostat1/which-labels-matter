# Label-Efficient Comparison of Shared Logistic Models with Finite-Sample Guarantees

**Label-Efficient Comparison (LEC)** acquires training and evaluation labels in batches and stops once they support a reliable model choice.

[Algorithm 1](docs/ALGORITHM.md) · [Population extension: Algorithm 2](docs/POPULATION.md) · [Results](docs/RESULTS.md) · [Reproduction](docs/REPRODUCING.md) · [中文算法说明](docs/ALGORITHM_ZH.md)

Model selection can consume labels beyond those needed for the final fit. In a multi-label task, each object also has several attributes that can be annotated separately. LEC compares two shared logistic training rules **before acquiring all labels in a prescribed cross-validation (CV) plan**. Public features determine a weighted sum of label products and an approximation error bound. Each draw queries at most two labels, and a shared cache avoids repeat purchases.

Algorithm 1 has three stages: **label-free preparation, sequential label queries, and selection with final training**. Its [pseudocode](docs/ALGORITHM.md#queries-and-decisions) and [LaTeX source](docs/ALGORITHM1.tex) follow the manuscript.

The main method starts at a zero anchor and needs **no pilot labels**. It uses a Clopper--Pearson (CP) exact binomial interval at each prespecified inspection, with error probability `delta/J` per look. Acquired labels also tighten the approximation error bound over all completions of the remaining unknown labels. The anchor, risk polynomial, sampling probabilities, and accumulated CP counts stay fixed. The interval supports a choice when its upper endpoint is at most `g`, or its lower endpoint is at least `-g`. Unresolved trials complete exact CV through the same cache before final training.

## Two comparison targets

| Target | Procedure | Entry point |
|---|---|---|
| Fixed CV plan | Zero anchor, cache-refined approximation bound, exact binomial intervals, and exact-CV fallback | **Algorithm 1**, implemented by `whichlabels.compare` |
| Expected test risk over random training sets | An unresolved public check starts batched queries. Cumulative betting and finite-pool cache bounds guide stopping. An unresolved choice at the query cap triggers paired refits. | [Algorithm 2 research implementation and protocol, November 25](docs/POPULATION.md) |

The paper also derives necessary label-information bounds for fixed feature designs. Sensitivity, dispersion, and approximation conditions yield matching inverse-square dependence on tolerance, with a logarithmic confidence factor. The sequential algorithm uses `log(2J/delta)` in place of `log(1/delta)`, where `J` is the number of inspections. See [the theory map and results](docs/RESULTS.md).

## Manuscript results

The experiments evaluate LEC on simulations, repeated random splits, full validation coverage, and population program risk.

- Across the two fixed CV plans, LEC used 70.5%–94.3% fewer image labels and 89.2%–89.8% fewer text labels on average than full annotation, counting final training. The largest image saving occurred with native features, where both rules met the tolerance in all 20 pools.
- In the nine population-risk settings, all 108 final LEC intervals contained their respective independent audit estimates. Mean savings relative to the least costly external baseline ranged from 92.7% to 99.1%.
- Three ablations vary the purchase unit, the update to the approximation bound, and early stopping. Comparison, fallback, and final training share one label cache.

The fixed-plan API in **0.5.0** implements the November 28 prediction-space bound. [RESULTS.md](docs/RESULTS.md) documents the archived November 25 experiments.

## Reproduce the November 25 tables and audits

```bash
python -m pip install -e ".[reproduce]"
python paper_reproduction/current/reproduce.py --output build/paper
```

This generates Tables 1–8, Figure 3, and audit summaries from committed path-level records. The self-contained demo below runs fresh acquisitions; rerunning real-data fits additionally requires the documented frozen feature/label caches. [Reproduction details](docs/REPRODUCING.md).

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
python -m whichlabels.demo --relative-drop 0.005 --output demo_output
python -m unittest discover -s tests -v
```

The example generates correlated binary attributes and compares:

1. **LEC** (recorded as selective moments in the demo output): the manuscript's Algorithm 1.
2. **Active CV:** complete training labels, selective evaluation attributes, valid sequential intervals, label reuse, and early stopping.

It writes `results.json`, `summary.csv`, and `progress.json`. Results include the supporting interval, distinct label counts by stage, fallback, final fitted coefficients, and measured computation time. A separate full-label audit runs **after** both decisions. The example is an executable illustration, not additional evidence for the paper's reported percentages.

## Use your own label provider

```python
from whichlabels import prepare_comparison, compare, LabelCache

# a.shape == (objects, attributes, d_a), b.shape == (objects, attributes, d_b).
# plan: fixed list of (training_object_ids, evaluation_object_ids).
# Each feature row has Euclidean norm at most one.
prepared = prepare_comparison(a, b, plan, penalty=0.5)

def acquire(object_ids, attribute_ids):
    # Return one binary label for every requested object-attribute pair.
    return annotation_service(object_ids, attribute_ids)

cache = LabelCache(a.shape[0], a.shape[1], acquire)
result = compare(
    prepared, cache, final_indices,
    relative_drop=0.005, delta=0.05, seed=17,
)
print(result['tolerance'], result['choice'], result['interval'], result['total_labels'])
```

`annotation_service`, the feature arrays, the fixed plan, and `final_indices` are supplied by the caller. The provider is called only for distinct, unbought labels. For a labeled dataset replay, `ArrayLabelProvider(y)` supplies the same interface.

Match the object and attribute order in both arrays. Each candidate fits its own coefficient vector. Designs of shape `(100, 6, 3)` and `(100, 6, 4)` compare predictions for the same 600 labels. LEC computes a paired bound on these common prediction rows for both equal and unequal feature widths. When the bound alone prevents stopping, it recomputes the split bounds using the training labels already in the cache.

The final subset has the same training size as the CV splits and must be fixed before acquisition. All coefficients, including any attribute intercepts, receive the L2 penalty.

## What the decision means

The target is the exact mean paired binary cross-entropy (BCE) difference on the **fixed finite CV plan**, conditional on the pool, its labels, and its features. Candidate A is acceptable when its score is at most `tolerance` above B's; B is acceptable under the symmetric condition. The probability statement concerns the randomized queries. Algorithm 2 instead averages test risk over random training sets.

Before querying, specify either `relative_drop` or `tolerance`. The example allows a **0.5% decrease** in the geometric-mean probability assigned to the observed validation labels. The API converts this fraction to an excess log-loss tolerance using `g = -log1p(-relative_drop)`, about `0.00501254`. To set the loss tolerance directly, pass `tolerance=g`. Both use the same query and stopping rules.

The approximation is not an unbiased estimator of the exact logistic score. Sampled terms are unbiased for the **approximate** score, and the interval includes its feature-based error bound. Large error bounds can require exact-CV fallback. See [the algorithm guide](docs/ALGORITHM.md) for the formulas and [scope and numerical checks](docs/SCOPE.md) for implementation conditions.

## Layout

| Path | Purpose |
|---|---|
| `src/whichlabels/api.py` | Prepare, query, stop, fall back, and train the selected candidate |
| `src/whichlabels/labels.py` | Callback-based label acquisition and distinct-label accounting |
| `src/whichlabels/_prediction_bound.py` | Prediction-space paired certificate for public features and partially observed training labels |
| `src/whichlabels/_bounds.py`, `_cache_bounds.py` | Bounds used by the baseline or explicitly selected historical controls |
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

The [November 25 reproduction index](paper_reproduction/current) contains the records for that version. The November 28 manuscript source archive includes its updated table renderers and records, including the native-width image comparisons.

For fixed CV, each batch uses the same public sampling probabilities and preserves all earlier draws. Refinement runs when the raw CP interval supports a choice but its current approximation enlargement does not. Otherwise the previous bound is retained. The active-CV comparator retains its original confidence sequences. Uniform-term and general prediction-assisted statistics retain their applicable bounded-variable intervals. Complete-label counts include final training; coverage and acceptable choices are reported separately.

## Citation and license

The manuscript title is **Label-Efficient Comparison of Shared Logistic Models with Finite-Sample Guarantees**. The repository name, Python package name (`which-labels-logistic`), and import name (`whichlabels`) remain stable.

The [citation template](docs/CITATION.cff.example) contains the current title and package version; author names and a public manuscript link remain to be filled in. A code license has not yet been selected. See [the repository maintenance notes](docs/GITHUB_UPLOAD_ZH.md).
