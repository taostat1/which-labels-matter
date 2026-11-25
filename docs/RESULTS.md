# Manuscript results and theoretical guarantees

Algorithm 1 uses exact binomial intervals for a fixed CV plan. Algorithm 2 uses a cumulative betting interval for expected test risk over random training sets. The results below follow the 11.25 manuscript.

## Experimental setup

GoEmotions supplies eight binary emotion attributes per comment; ADE20K supplies six semantic-presence attributes per image. Each domain uses 20 pools of 4,096 objects and three query paths per pool. Each prescribed plan contains 32 splits. Reported counts include final training.

The image candidates use DINOv2-S and DINOv2-B features, 22 shared coefficients, training size 128, and L2 penalty 0.5. The text candidates use 20 coefficients, training size 64, penalty 0.125, and a 0.9/0.1 mixture for the second feature map. The error level is 0.05.

## Repeated random splits

Counts are mean distinct labels. Active CV acquires complete training labels and selects evaluation attributes.

| Domain | g | Active CV | LEC | Saved vs active CV | Acceptable / covered |
|---|---:|---:|---:|---:|---:|
| ADE20K | 0.0025 | 17,021.8 | **7,133.2** | 58.1% | 60/60 / 58/60 |
| ADE20K | 0.00375 | 17,021.8 | **5,781.6** | 66.0% | 60/60 / 58/60 |
| ADE20K | 0.005 | 17,021.8 | **4,412.8** | 74.1% | 60/60 / 58/60 |
| GoEmotions | 0.005 | 15,852.4 | **4,759.1** | 70.0% | 60/60 / 60/60 |

Across the four LEC configurations: 240/240 acceptable choices, 234/240 covering stopping intervals, and 0 exact-CV fallbacks.

## Full validation coverage

Counts are mean distinct labels. Active CV acquires complete training labels and selects evaluation attributes.

| Domain | g | Active CV | LEC | Saved vs active CV | Acceptable / covered |
|---|---:|---:|---:|---:|---:|
| ADE20K | 0.0025 | 23,862.2 | **7,245.0** | 69.6% | 60/60 / 59/60 |
| ADE20K | 0.00375 | 23,862.2 | **5,797.8** | 75.7% | 58/60 / 57/60 |
| ADE20K | 0.005 | 23,862.2 | **4,559.2** | 80.9% | 60/60 / 57/60 |
| GoEmotions | 0.005 | 18,992.8 | **5,073.9** | 73.3% | 60/60 / 60/60 |

Every object enters validation once. Relative to full annotation (24,576 image labels or 32,768 text labels), mean savings are 70.5%–84.5%.

Across the four LEC configurations: 238/240 acceptable choices, 233/240 covering stopping intervals, and 0 exact-CV fallbacks.

## Other experiments

The research record covers 1,872 fixed designs. Whole-object and separate-bound variants use the corresponding local certificate; uniform-term sampling uses a general bounded-variable interval. Acquisition masks, confidence intervals, decisions, and complete label counts are retained for every path.

The main simulation uses twelve cells and 100 pools per cell. Four additional weak-penalty cells examine the boundary of the approximation. Ordinary fivefold results use their specified preparation gate. The cache appendix includes the public entry check and reports zero-query decisions separately.

## Population program risk

Each of the eight configurations uses 12 matched paths. Counts are mean distinct labels including final training. The saving compares LEC with the least costly of the four external baselines in that row.

| Case | Direct refits | Active testing | AMS | Completion certificate | LEC | Saving |
|---|---:|---:|---:|---:|---:|---:|
| T1 | 111,600.7 | 111,584.4 | 111,587.1 | 111,598.2 | 1,023.2 | 99.1% |
| T2 | 111,487.3 | 111,473.8 | 111,447.7 | 109,778.2 | 1,021.8 | 99.1% |
| T3 | 111,332.7 | 111,314.5 | 111,312.8 | 111,327.0 | 1,022.0 | 99.1% |
| T4 | 363,104.7 | 369,307.5 | 366,810.6 | 365,393.0 | 14,856.6 | 95.9% |
| V1 | 113,686.0 | 114,249.0 | 113,941.7 | 112,763.6 | 1,556.9 | 98.6% |
| V2 | 119,009.5 | 119,456.3 | 118,759.5 | 119,192.3 | 4,983.0 | 95.8% |
| V3 | 120,264.0 | 120,636.7 | 119,913.6 | 118,283.3 | 8,035.6 | 93.2% |
| V4 | 120,248.0 | 120,625.8 | 119,904.8 | 119,603.9 | 7,782.1 | 93.5% |

All 96 main LEC paths stopped before paired refits. Across all ten methods and variants, all 960 decisions met the tolerance against the independent 2,048-refit audit and their intervals contained the audit mean. The audit estimates population risk by Monte Carlo.

Cache refinement reduced the mean cost in V4 by 1.9% and left the other seven main means unchanged. Most of the saving came from selective purchases of training labels. The separate-bound variant used all 32,768 query blocks before refitting on all 24 T4 and V3 paths. See the [reproduction guide](REPRODUCING.md) for the matched initialization and interval comparisons.

## Theoretical guarantees

| Result | Role |
|---|---|
| Theorem 1 | Public paired approximation error |
| Theorem 2 | CP coverage combined with uniform cache refinements |
| Theorem 3 | Valid stopping, total label cap, and same-stream nonincrease from refinement |
| Theorem 4 and Corollary 3 | Necessary information and matching-order conditions |
| Theorems 5–7 | Global and calibrated population error, and unbiased approximate-risk representation |

Under the stated sensitivity, dispersion, and approximation conditions, the fixed-design benchmark has inverse-square tolerance dependence with a logarithmic confidence factor. The upper bound includes the prescribed-inspection factor.
