# Algorithm 2: LEC for population risk

Algorithm 2 compares expected test risk over random training sets of fixed size. Algorithm 1 conditions on a fixed cross-validation plan. Both now start from **zero anchors and an empty label cache**. The installable `whichlabels.compare` API implements Algorithm 1; Algorithm 2 uses the research sources in [current reproduction](../paper_reproduction/current).

## Target and initialization

For training law P, test law Q, and training size n,

$$R_j(n)=\mathbb E_{S\sim P^n}\mathbb E_{W\sim Q}\ell_j(\widehat\theta_j(S);W),\qquad\Delta_n=R_A(n)-R_B(n).$$

Training draws are independent, and test draws are independent of training. Attributes on one object may be dependent. For the finite-pool experiments, P and Q are the fixed complete empirical laws, sampled with replacement. Conditioning on the acquired-label history does not redefine this target.

Public features supply score-moment, curvature, and paired approximation bounds. Exact public sums have no certification error; Monte Carlo integration over public designs uses a preassigned error probability and is frozen before querying. The default buys **no pilot or calibration labels**. If its public interval supports a choice, it proceeds to final training. An unresolved case with epsilon_0 >= g goes to independent paired refits; otherwise it queries in batches.

## Cumulative queries and interval

Theorem 7 represents the approximate population risk difference through one test object and two training positions. Its coefficients use all n public training features. Each sampled product requests at most two labels.

On a fresh independent feature block, substitute cached labels exactly and collect the remaining polynomial. For known constant c_t and remaining coefficients a_(t,k),

$$Z_t=c_t+\frac{a_{t,K}\psi_{t,K}}{p_{t,K}},\qquad p_{t,k}\propto|a_{t,k}|.$$

If no terms remain, use c_t. Averaging over the query and fresh block gives E(Z_t | history) = mu, the same approximate risk mean at every step. A public envelope bounds |Z_t| <= M even after cache substitution.

Algorithm 1 has fixed two-point contributions, enabling cumulative Clopper–Pearson intervals. Algorithm 2 has changing contribution magnitudes and uses cumulative **bounded-mean betting**. For fixed fractions c in G,

$$K_N^\pm(m)=\frac1{|G|}\sum_{c\in G}\prod_{t=1}^N\left(1\pm\frac{c(Z_t-m)}{M\pm m}\right),\qquad -M<m<M.$$

Invert both capitals at 2/delta_s and intersect over inspections to obtain J_N. Each batch retains every earlier factor. Endpoint means are retained only when all observations equal that endpoint; if M = 0, J_N = {0}.

For a known finite pool, acquired labels restrict feasible completions. Public plus cached score bounds hold for every compatible completion, so their running minimum epsilon_N is valid after adaptive queries without additional purchases or error allocation. The exact-risk interval is

$$I_N=[\inf J_N-\varepsilon_N,\ \sup J_N+\varepsilon_N].$$

An upper endpoint <= g supports A; a lower endpoint >= -g supports B. Empty or unresolved capped intervals invoke paired refits with separate error allocation and numerical-error correction. An independent final training sample shares the same purchase cache. Total labels count distinct acquisitions across all stages.

## Matched preparation and cache ablation

Table 7 uses four arms: paid independent pilot / zero anchor, each with a fixed / cache-refined error bound. All use the current cumulative interval, the same 12 query seeds per configuration, final samples, caps, and independent audit.

Paid preparation fits on 512 pilot draws and calibrates independently. Its means range from 20,839 to 236,859 labels; zero-anchor LEC ranges from 1,022 to 14,857. Cache refinement changes only V4's population stopping time, reducing its mean by 1.91%. The main advantage therefore comes from avoiding complete training-label purchases and preparation. All 384 ablation choices are acceptable relative to the audit mean; each paid arm has two T4 refit fallbacks.

Theorems 5–6 and Appendix C give the bounds; Theorem 7 gives the unbiased approximate-risk statistic. Appendix C.7 proves the adaptive-cache betting argument, endpoint rules, and total error allocation. [ALGORITHM2.tex](ALGORITHM2.tex) contains the manuscript pseudocode. Reanchoring is a phased ablation, not the main algorithm.
