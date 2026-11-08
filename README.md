# Which Labels Matter for Model Comparison?

Selective label comparison for shared logistic regression.

[Algorithm 1: 14 steps](docs/ALGORITHM.md#queries-and-decisions) · [中文算法说明](docs/ALGORITHM_ZH.md)

Repeated CV can involve many more labeled objects than the final fit. This code compares two fixed feature representations **before acquiring all labels in a prescribed CV plan**. It uses public features to identify weighted label products, queries their labels, and stops when its interval supports a candidate within a specified tolerance.

Algorithm 1 has three stages: **label-free preparation, sequential label queries, and selection with final training**. Its [14-step pseudocode](docs/ALGORITHM.md#queries-and-decisions) and [LaTeX source](docs/ALGORITHM1.tex) follow the manuscript.

The main method starts at a zero anchor and needs **no pilot labels**. The current 0.2.1 method uses a Clopper--Pearson (CP) exact binomial interval at each prespecified inspection, with error probability `delta/J` per look. Its endpoints include the approximation error bound. Labels acquired for comparison, fallback, and final training are counted once through a shared cache.

Version 0.2.1 also recognizes exact cancellation for identical training designs under weak regularization. Identical candidates need only the prescribed final-training labels. The reported experimental configurations are unchanged. See [the reproduction guide](docs/REPRODUCING.md) for the current simulation and real-data archives.

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

1. **Selective moments:** the manuscript's Algorithm 1.
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

The target is the exact mean paired BCE difference on the **fixed finite CV plan**, conditional on the pool, its labels, and its features. Candidate A is acceptable when its score is at most `tolerance` above B's; B is acceptable under the symmetric condition. The probability statement concerns the randomized queries. Population program risk is a separate extension in the paper.

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

Add the real author names, citation information, and manuscript URL. Choose a code license and add `LICENSE`; no license has been assigned on the authors' behalf. A citation template is in `docs/CITATION.cff.example`. See [the upload notes](docs/GITHUB_UPLOAD_ZH.md).
