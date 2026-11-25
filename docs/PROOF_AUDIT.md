# Proof and implementation audit for 11.17

Three independent agent reviews examined the cache proofs, population interval proofs, and nonempty matching conditions. They used the manuscript and code directly. Numerical checks supplement these analytic reviews; they do not establish a universal theorem by testing alone.

The 11.25 query-first rule changes when Algorithm 2 invokes refits. Every unresolved public check enters the scheduled query batches. The betting interval covers simultaneously over those batches and each finite-pool cache bound covers every compatible completion. The same error allocations therefore apply when refits follow an unresolved final inspection.

## Clarifications included in the manuscript

- **A.2:** cache bounds cover every compatible completion. Taking their running minimum is therefore valid after adaptive queries without allocating another failure probability.
- **C.2:** expectations concern the fixed complete empirical law and a new independent draw. They are not posterior expectations conditional on which labels were bought. Public bounds and cache bounds dominate the same target moments.
- **C.6:** the public-design Monte Carlo certificate is fixed before comparison. Only deterministic compatible-completion bounds refine it during acquisition. The paired cap, local derivatives, and positive-denominator rules are explicit.
- **C.7:** a fresh random block restores the common mean after exact cache substitution. The cumulative betting factors thus remain valid. The proof specifies degenerate ranges, endpoint means, running intersections, and the certification/query/refit error allocation.
- **Numerical refits:** batchwise solver stopping can couple numerical errors. The proof applies concentration to exact independent scores, then bounds perturbations in both their mean and sample standard deviation. The implementation's numerical allowance exceeds this correction at its inspection schedule.
- **Active-testing and AMS proposals:** each Newton fit now freezes at its own residual tolerance. This makes its proposal a function of its own block. All 288 external-baseline paths were rerun with the same seeds: label counts and choices were unchanged. Tests compare single-row, batched, permuted, and partitioned fits and their proposals.

## Dense-design certificate

Appendix B provides four dense, full-column-rank feature matrices and a tied interior label-mean vector. At penalty 10, an explicit tolerance and disjoint copies satisfy every matching condition. The original bounded-difference condition is verified with 24,442,334 copies. A separate Bernstein variance condition reduces the sufficient count to 307,100; it is not claimed to satisfy the original dispersion inequality. Rational arithmetic and outward square-root/log bounds verify these statements in `verify_dense_conditions.mjs`.

The copies must use distinct labels. Requerying the same labels does not disperse their effect. Some matrices have orthogonal columns in the displayed example; strict inequalities and continuity also give nearby designs with nonorthogonal columns.

A second implementation, `paper_reproduction/current/audits/recheck_dense_fractions.py`, uses Python fractions, generic Gaussian elimination, all 256 label vectors, and exact product-law expectations. It reproduces the published `B1 = 0.0013546069039013769` and `g = 9.202937589231276e-6`; its intermediate means and gradient are recorded in `paper_evidence/dense_fraction_recheck_20261117.json`.

## Executable checks

`independent_cache_math_check.py` derives formulas separately from the experiment implementation and enumerates compatible cache completions for fixed-plan and finite-pool population bounds. `test_population_interval.py` enumerates adaptive common-mean paths and checks betting capital and inversion, including endpoint cases. Package tests also check simultaneous CP coverage and parity with the research code.

The current reporter verifies saved records, including label unions, bound refinement, and decision counts. It reports failures and noncovering intervals rather than filtering them. Fresh optimizer replay and saved-record reporting are separate tasks, as described in `REPRODUCING.md`.
