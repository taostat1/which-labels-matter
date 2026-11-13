# Manuscript results and theoretical guarantees

This page summarizes **Label-Efficient Comparison of Shared Logistic Models with Finite-Sample Guarantees**, revised 12 November 2026. Numbers come from the frozen CP experiment records, not from a new run.

## Experimental setup

GoEmotions supplies eight binary emotion attributes per comment; ADE20K supplies six semantic-presence attributes per image. Each corpus has 20 independently drawn pools of 4,096 distinct objects; pools may overlap within a corpus. Each pool has 32 prescribed splits and three query paths. Variability is summarized across pool means.

The image comparison uses DINOv2-S and DINOv2-B with a 22-coefficient shared output layer, training size 128, and L2 penalty 0.5. The text comparison uses 20 coefficients, training size 64, and penalty 0.125; its second map mixes mean-pooling and first-token features with weights 0.9 and 0.1. The error level is 0.05.

All counts below include distinct labels for comparison and final training. Active CV fits each opened training subset with complete labels and actively queries evaluation attributes. LEC can also query selected training-label contributions.

## Repeated random splits

Each evaluation subset has 128 objects and is disjoint from its training subset. The candidate designs and tolerance grid are fixed before querying. The image grid includes both single-acceptable and dual-acceptable comparisons.

Counts are means in thousands. Savings use the unrounded means.

| Dataset | Tolerance | Active CV | LEC | Saving vs active CV | Acceptable LEC choices |
|---|---:|---:|---:|---:|---:|
| ADE20K | .0025 | 17.02 | **7.13** | 58.1% | 60/60 |
| ADE20K | .00375 | 17.02 | **5.78** | 66.0% | 60/60 |
| ADE20K | .005 | 17.02 | **4.46** | 73.8% | 60/60 |
| GoEmotions | .005 | 15.85 | **5.62** | 64.6% | 60/60 |

All 240 LEC paths return an acceptable candidate without fallback. Across all five methods, 1,200/1,200 choices are acceptable and 1,182/1,200 stopping intervals cover the exact target. Choice acceptability and interval coverage measure different events.

## Full validation coverage at a fixed training size

Each pool is partitioned into 32 validation blocks of 128 objects. Split b evaluates block b and trains on the first n objects of the next block, cyclically. Every object enters validation once; labels are acquired only when queried or needed for final training.

Counts are means in thousands. Full annotation costs 24,576 labels for ADE20K and 32,768 for GoEmotions.

| Dataset | Tolerance | Full annotation | Active CV | LEC | Saving vs full | Saving vs active CV | Acceptable LEC choices |
|---|---:|---:|---:|---:|---:|---:|---:|
| ADE20K | .0025 | 24.58 | 23.86 | **7.24** | 70.5% | 69.6% | 60/60 |
| ADE20K | .00375 | 24.58 | 23.86 | **5.80** | 76.4% | 75.7% | 58/60 |
| ADE20K | .005 | 24.58 | 23.86 | **4.56** | 81.4% | 80.9% | 60/60 |
| GoEmotions | .005 | 32.77 | 18.99 | **5.52** | 83.2% | 71.0% | 60/60 |

LEC returns acceptable choices on 238/240 paths, with no fallback. Both unacceptable choices occur at image tolerance 0.00375. The nominal guarantee concerns each comparison's query randomization.

The [full summary CSV](../paper_evidence/cp_20261106/real_summary.csv) includes pool standard deviations, coverage, acceptability, fallback counts, and the whole-object, no-early-stopping, and separate-bound ablations. The frozen identifier `selective_paired` denotes LEC.

## Simulations and population program risk

The main simulation varies regularization, candidate difference, and signal strength over 12 configurations and 100 independent pools. Its 4,800 runs compare LEC, uniform terms, separate error bounds, and active CV. LEC has the smallest mean count in each cell, including ties; all 4,800 choices are acceptable. The appendix retains earlier weak-penalty configurations where active CV uses fewer labels.

The [population extension](POPULATION.md) evaluates expected test risk over random training sets in eight configurations. At the common 80,000-label budget, LEC completes 75% of text paths and every image path. The active-testing and active-model-selection adaptations complete none at that budget. Preparation and final training are included. Full protocols and evidence are in the companion archives listed in [REPRODUCING.md](REPRODUCING.md).

## What the seven theorems establish

| Result | Role |
|---|---|
| Theorem 1 | Bound paired fixed-CV approximation error from public features |
| Theorem 2 | Combine that bound with CP intervals to control selection error at prescribed inspections |
| Theorem 3 | Guarantee completion and bound total label acquisition, including exact-CV fallback |
| Theorem 4 | Bound necessary information from distinct labels under the same query permissions |
| Theorem 5 | Control one-step approximation error for population program risk |
| Theorem 6 | Tighten population approximation error using local curvature and independent calibration |
| Theorem 7 | Give an unbiased statistic for the approximate population-risk difference |

The label-order result is conditional. Under interior sensitivity, dispersed effects, a sufficiently small approximation bound, and sensitivity comparable to coefficient mass, the optimal worst-case total label count has order

$$|\mathcal F|+\frac{S_1^2}{g^2}\log(1/\delta).$$

Algorithm 1 attains the corresponding bound with `log(2J/delta)` when its query budget includes a sufficient horizon and its inspection grid has bounded overshoot. Here `F` is the prescribed final-training label set, `S1` the total absolute coefficient mass, `g` the tolerance, and `J` the number of inspections.

The application is a comparison at a fixed final-training budget. If final training requires every label in the pool, the complete-label count equals full annotation. Ordinary fivefold CV with final training on four fifths of the pool has a 20% saving ceiling.
