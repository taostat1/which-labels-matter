# Scope and numerical checks

- The public API implements the zero-anchor **fixed finite CV** algorithm. It does not treat a finite-plan confidence interval as a population-risk interval.
- Responses are binary. The API uses mean binary cross-entropy and a strictly positive L2 penalty on every coefficient.
- Both designs use the same objects, attributes, coefficient dimensions, and aligned coordinates. Feature rows have norm at most one.
- The plan, inspection schedule, and final training subset are fixed before label acquisition. Within each split, training and evaluation subsets are disjoint. Overlap across different splits is allowed.
- `delta` applies to one comparison. Simultaneous decisions across multiple feature pairs or tolerances need a joint error allocation.
- Conditional on the fixed pool labels, the term draws are independent. The same object or attribute can appear in many terms. The cache reuses the label but does not remove the draw from the estimator.
- The default signed-product sampler uses CP intervals with `delta/J` per prespecified look. It requires fixed coefficient-proportional probabilities and binary signed products. General prediction-assisted or uniform-term statistics need a different interval engine.
- CP endpoints are asymmetric about the sample mean. Use `interval` or each history entry's `lower` and `upper`; `statistical_radius` is the larger one-sided distance retained for diagnostics.
- The approximation error bound remains in every sampling interval. A low sampling variance does not remove approximation error.
- The implementation checks the public range before sampling. It does not implement the experimental remaining-contribution sampler or a post-query deterministic cache stopping rule.
- The CP probability endpoints receive `2e-12` outward padding. A CDF/grid check bounds drift from ideal coefficient sampling. The scalar `2e-9` is the numerical guard used in the experiments. The public wrapper also checks fitted gradients. Strong convexity bounds parameter error by `gradient_norm / lambda`; normalized evaluation rows bound the resulting BCE error. A failed numerical check raises an error rather than returning an unsupported decision.
- The public API requires `g > 1e-8`, which leaves room for this numerical guard. Smaller tolerances require revisiting numerical precision.
- The polynomial stores potentially many label pairs. With `B` splits, `n r` training rows and `q r` evaluation rows per split, construction can involve order `B[(nr)^2 + (nr)(qr)]` entries before duplicate aggregation. The demonstration uses small fixed training budgets.
- Label count and computation time are different resources. This implementation reports both. It does not assign artificial annotation prices.
- The code is a research implementation. Standard floating-point arithmetic is used; the reported intervals are not interval-arithmetic certificates for every floating-point operation.

