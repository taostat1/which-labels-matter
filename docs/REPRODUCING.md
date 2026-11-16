# Reproduce the 11.17 manuscript

The current entry point is `paper_reproduction/current/reproduce.py`. It requires no external ZIP or raw data to reconstruct **Tables 1–8, Figure 3, and the reported audit summaries** from committed path-level records.

```bash
python -m pip install -e ".[reproduce]"
python paper_reproduction/current/reproduce.py --output build/paper
```

Outputs include numbered TeX tables, the manuscript's `full_coverage_table.tex` TikZ figure, a standalone `Figure_3.pdf` displaying the same values, unrounded `summary.json`, `population_paths.csv`, `audit.json`, and an output hash manifest. Compile the TikZ source inside the manuscript to reproduce its exact layout. The reporter checks input hashes, distinct-label accounting, decisions, and saved cache-refinement records.

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
| Tables 1–2, 6, 8; Figure 3 | `data/fixed_records.json.gz` and `data/cache_normalized.json.gz` |
| Table 3 | `data/geometry_pools.json` |
| Table 4 | `data/population_zero.json.gz` and `data/population_baselines.json.gz` |
| Table 5 | `data/matched_protocol.json` and `data/risk_audit.json` |
| Table 7 | `data/population_zero.json.gz` and `data/population_paid.json.gz` |
| Fivefold and reanchoring checks | fixed records and `data/reanchor_screen.json` |

Paths are relative to `paper_reproduction/current`. `manifest.json` hashes the record files and retained research sources. Twelve matched population paths per configuration use groups 6–17 and the seed formulas in `matched_protocol.json`. All four initialization/cache arms share the current cumulative interval. The paid arms reproduce the earlier preparation procedure while holding current inference fixed. The external baseline replay contains 288 new executions with independently stopped Newton fits; label counts and choices match the earlier records. The two paid T4 fallback paths reuse exact-fit statistics from the identical seed streams and recompute their label unions.

Saved-record checks reconstruct reporting and verify recorded algebra and bookkeeping. They do not rerun all original feature extraction and optimizer calls. [PROOF_AUDIT.md](PROOF_AUDIT.md) describes the independent mathematical review and its scope.

## Rerun original experiments from features

`paper_reproduction/current/reference/` preserves the research modules and their relative imports. The simulation, fixed-plan, zero-anchor population, paired-bound, and matched-ablation sources are retained there. Their original absolute output paths and environment-specific caches are research configuration, not package defaults. A full fresh replay requires the frozen feature arrays, synthetic design schedules, and query-input files identified in those protocols. These large files are not all committed. The historical simulation runner under `paper_reproduction/simulation/` reproduces an older experiment version; use the current records for current paper tables.

Prepared real-data cache formats and hashes are documented in `paper_evidence/data_manifest.json`. GoEmotions provides eight emotion attributes per comment; ADE20K provides six semantic-presence attributes per image. The original datasets remain available from their providers. Raw datasets, pretrained weights, and per-object real-data feature/label arrays are excluded from this compact repository.

For a new application, provide normalized arrays of shape `(objects, attributes, coefficients)`, a fixed split plan, a label provider, and a final-training subset to `whichlabels.compare`. Its default uses a fixed zero anchor, cumulative CP intervals, and free cache refinement. Algorithm 2 is a research implementation rather than an additional public API.
