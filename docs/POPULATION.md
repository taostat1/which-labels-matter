# Algorithm 2: November 25 population-risk protocol

The [November 25 research sources](../paper_reproduction/current) implement Algorithm 2 for expected test risk over random training sets of fixed size. The installable [Algorithm 1](ALGORITHM.md) compares two rules on a fixed cross-validation plan. Both start from **zero anchors and an empty label cache**.

## Target and initialization

For training law P, test law Q, and training size n,

$$R_j(n)=\mathbb E_{S\sim P^n}\mathbb E_{W\sim Q}\ell_j(\widehat\theta_j(S);W),\qquad\Delta_n=R_A(n)-R_B(n).$$

Training draws are independent, and test draws are independent of training. Attributes on one object may be dependent. For the finite-pool experiments, P and Q are the fixed complete empirical laws, sampled with replacement. Conditioning on the acquired-label history does not redefine this target.

The zero anchors need **no pilot or calibration labels**. If the public interval supports a choice, LEC proceeds to final training. Every unresolved comparison starts querying, including when the initial approximation bound is at least the tolerance. New blocks can establish a sufficiently large loss difference even if the approximation bound stays wide.

Public features bound the approximation error for each candidate. Distinct coefficient dimensions admit the sum of these two bounds. Common coefficient coordinates also admit a paired bound, and LEC keeps the tightest available bound. Monte Carlo integration over public designs uses a prespecified error allocation and is completed before querying.

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

An upper endpoint at most g supports A. A lower endpoint at least −g supports B. An empty betting intersection supports neither choice, so the scheduled batches continue. A comparison still open at the query cap passes to independent paired refits with their own error allocation and numerical-error correction. If the refit budget expires without a decision, LEC reports unresolved. Final training on a new sample reuses all labels already acquired.

## Matched preparation and cache ablation

Table 7 uses four arms: paid independent pilot / zero anchor, each with a fixed / cache-refined error bound. All use the current cumulative interval, the same 12 query seeds per configuration, final samples, caps, and independent audit.

Paid preparation uses 512 pilot draws followed by independent calibration. In T4 two pilot samples gave initial bounds just above g = .005. Both paid variants chose B after 8,192 query blocks on each path, with the bound still above g. Each paid variant averaged 206,964 distinct labels across all 12 T4 paths. All 384 initialization and bound comparisons chose an acceptable candidate under the independent audit.

The separate-bound ablation in Table 4 illustrates the other outcome. In T4 and V3 all 24 paths used their 32,768 query blocks before requiring paired refits. The 96 main LEC paths across the eight settings all reached a choice during the query stage.

Theorems 5–6 and Appendix C give the bounds; Theorem 7 gives the unbiased approximate-risk statistic. Appendix C.7 proves the adaptive-cache betting argument, endpoint rules, and total error allocation. [ALGORITHM2.tex](ALGORITHM2.tex) contains the manuscript pseudocode. Reanchoring is a phased ablation, not the main algorithm.
