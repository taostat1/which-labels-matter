# Reproduce the 11.25 manuscript

The November 25 reporter is `paper_reproduction/current/reproduce.py`. It reconstructs **Tables 1–8, Figure 3, and the reported audit summaries** from committed path-level records. For new comparisons with the November 28 prediction bound and the relative-score tolerance, use the [fixed-plan API](ALGORITHM.md).

```bash
python -m pip install -e ".[reproduce]"
python paper_reproduction/current/reproduce.py --output build/paper
```

Outputs include numbered TeX tables, the plotted values and bars in `full_coverage_bars.tikz`, a standalone `Figure_3.pdf` displaying the same values, unrounded `summary.json`, `population_paths.csv`, `audit.json`, and an output hash manifest. Compile the TikZ source inside the manuscript to reproduce its exact layout. The reporter checks input hashes, distinct-label accounting, decisions, and saved cache-refinement records.

## Run new acquisitions and mathematical checks

```bash
python -m unittest discover -s tests -v
python -m whichlabels.demo --output build/demo
python paper_reproduction/current/audits/independent_cache_math_check.py
node paper_reproduction/current/audits/verify_dense_conditions.mjs
```

The demo freshly fits candidates, runs LEC and active CV, and performs a separate all-label audit. It is illustrative; it is not a replacement for the formal manuscript experiments. The tests include exhaustive small-label cache checks, simultaneous CP coverage, and finite-path checks of adaptive bounded-mean betting. Node.js is required only for the dense-design rational-arithmetic certificate.

## Current evidence

| Manuscript artifact | Committed records |
|---|---|
| Tables 1 and 2, Figure 3, and Table 8 | Fixed-plan and simulation paths in `aligned/paths.csv.gz`, cost and interval controls in `aligned/`, and the source records in `data/` |
| Tables 3, 5, and 6 | Design, independent audit, and refinement records in `data/` |
| Table 4 | All 960 matched method-path records in `aligned/population/paths.csv.gz` |
| Table 7 | All 384 matched initialization and cache paths in `aligned/initialization_paths.csv.gz` |
| Query-first path histories | `aligned/population/query_first_history.json.gz` |

Paths are relative to `paper_reproduction/current`. Its `manifest.json` identifies the inputs and their hashes. Twelve matched population paths per configuration use groups 6–17 and the seed formulas in `data/matched_protocol.json`. All four initialization and cache arms share the cumulative interval. In T4 the two paid-initialization seeds with a public bound above the tolerance reached a decision after 8,192 query blocks. The separate-bound variant in T4 and V3 reached the query cap before paired refits.

Saved-record checks reconstruct reporting and verify recorded algebra and bookkeeping. They do not rerun all original feature extraction and optimizer calls. [PROOF_AUDIT.md](PROOF_AUDIT.md) describes the independent mathematical review and its scope.

## Rerun original experiments from features

`paper_reproduction/current/reference/` preserves the research modules and their relative imports. The simulation, fixed-plan, zero-anchor population, paired-bound, and matched-ablation sources are retained there. Their original absolute output paths and environment-specific caches are research configuration, not package defaults. A full fresh replay requires the frozen feature arrays, synthetic design schedules, and query-input files identified in those protocols. These large files are not all committed. The current primary and paid population runners accept the `LEC_SHARED` environment variable pointing to the prepared `experiments/shared_resource` directory. Set `LEC_QUERY_OUTPUT` or `LEC_PAID_OUTPUT` to write a fresh run to a separate directory. The [separate-bound replay](../paper_reproduction/current/reference/aligned_20261121/population/replay_separate.py) exercises the cap and refit branch. The historical simulation runner under `paper_reproduction/simulation/` reproduces an older experiment version; use the current records for current paper tables.

Prepared real-data cache formats and hashes are documented in `paper_evidence/data_manifest.json`. GoEmotions provides eight emotion attributes per comment; ADE20K provides six semantic-presence attributes per image. The original datasets remain available from their providers. Raw datasets, pretrained weights, and per-object real-data feature/label arrays are excluded from this compact repository.

For a new application, provide normalized arrays of shape `(objects, attributes, coefficients)`, a fixed split plan, a label provider, and a final-training subset to `whichlabels.compare`. Its default uses a fixed zero anchor, cumulative CP intervals, and free cache refinement. Algorithm 2 is a research implementation rather than an additional public API.
