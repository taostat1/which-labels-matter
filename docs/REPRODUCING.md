# Reproduce the manuscript

There are two software routes. The installable `whichlabels` package is a clean algorithm interface. `paper_reproduction` preserves the original relative imports and protocols for rerunning the published experiment design. Function-level source hashes are in `source_provenance.json`.

Three reporting scripts now create their output directories explicitly, so a fresh checkout can generate figures and tables. These filesystem-only changes are recorded in `packaging_changes.json`; the experiment algorithms and frozen protocols are unchanged.

## Controlled simulation

Install the reproduction dependencies, then use a fresh result directory:

```bash
python -m pip install -e ".[reproduce]"
cd paper_reproduction/simulation
python experiments/shared_resource/logistic_simulation_20261101/run.py --smoke
```

The smoke mode evaluates one separate pool across the twelve grid cells. The formal protocol uses 100 independent pools and four methods (4800 method runs):

```bash
python experiments/shared_resource/logistic_simulation_20261101/run.py --workers 4
python experiments/shared_resource/logistic_simulation_20261101/summarize.py
python experiments/shared_resource/logistic_simulation_20261101/time_serial.py
python experiments/shared_resource/logistic_simulation_20261101/artifacts.py
```

The 12 cells vary candidate interpolation, regularization, and signal strength. All cells are retained. The smoke example is separate from the formal evidence. Completed formal pool files are reused on restart; use a new working copy for a new independent rerun. Plot generation may require LaTeX.

## Main text and image comparison

The main Table 1 experiment is the tolerance-grid and matched-ablation run:

```bash
cd paper_reproduction/main_cv
python experiments/shared_resource/logistic_cv_attribution_20261031/run.py --smoke
python experiments/shared_resource/logistic_cv_attribution_20261031/run.py
python experiments/shared_resource/logistic_cv_attribution_20261031/summarize.py
python experiments/shared_resource/logistic_cv_attribution_20261031/plot_results.py
```

Set `LOGISTIC_DATA_ROOT` before running. It must contain:

```text
goemotions_comparison_20261028/prepared/features_public.npz
goemotions_comparison_20261028/prepared/labels_oracle.npz
cross_backbone_20261028/prepared/features_public.npz
cross_backbone_20261028/prepared/labels_oracle.npz
```

The loader matches object IDs between feature and label files. Text and backbone caches contain `a`, `other`, `ids`, and `roles` in the feature file; the label file contains `ids` and signed binary `y`. The original loader converts signed responses to 0/1. Hashes and expected sizes are in `paper_evidence/data_manifest.json`.

The paper describes frozen text feature extraction, image feature extraction, and attribute construction. This release starts from those prepared caches. Dataset downloads, pretrained weights, and feature-extraction pipelines are outside this compact algorithm package. Users can also construct their own normalized `(objects, attributes, coefficients)` arrays and call the public API directly.

The primary run uses 20 pools per domain, three query paths per pool, 32 splits, and the prespecified tolerance grid. Full-label audits and acquisition policies remain separate. `run.py` refuses to overwrite completed formal evidence.

## Other paper results

The Overleaf/source archive is the full evidence archive. It also contains ordinary-fivefold checks, the cache ablation, information-bound checks, and the population-risk extension. These are not prerequisites for running Algorithm 1. In particular, the historical paid-pilot/planner scripts should not be used as the entry point for the main fixed-CV method.

