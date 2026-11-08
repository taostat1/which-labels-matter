# Algorithm 1: selective labels for a fixed CV comparison

## Statistical target

There are `m` objects and `r` binary attributes per object. Candidate `j` has public feature rows `z_j(i,a)` in a shared coefficient space. Each row has norm at most one. All coefficients receive an L2 penalty `lambda > 0`.

For a training subset `S`, the exact parameter minimizes

$$
\widehat\theta_{j,S}=\arg\min_\theta\left\{
\frac{1}{|S|r}\sum_{i\in S,a}\left[\log(1+e^{z_j(i,a)^\top\theta})-Y_{ia}z_j(i,a)^\top\theta\right]
+\frac\lambda2\|\theta\|^2\right\}.
$$

A fixed plan contains `B` pairs of training and evaluation subsets. Its target difference is

$$\Delta_{\mathcal V}=\frac1B\sum_{b=1}^B
\bigl[R_{A,b}(\widehat\theta_{A,b})-R_{B,b}(\widehat\theta_{B,b})\bigr].$$

`R` is mean evaluation BCE, without the training penalty. A is acceptable if `Delta <= g`; B is acceptable if `Delta >= -g`. Both can be acceptable.

## Public preparation

Write each binary label as `x = 2Y - 1`. For one split, let `Z_j` contain the training rows, `W_j` the evaluation rows, and `N = n r`, `M = n_test r`.

$$H_j=\lambda I+\frac{Z_j^\top Z_j}{4N},\qquad
L_j=\frac{H_j^{-1}Z_j^\top}{2N},\qquad t_j=L_jx_{\rm tr},\qquad Q_j=W_jL_j.$$

The zero-anchor Newton update is linear in the signed labels. The quadratic BCE expansion gives the paired split statistic

$$T_b=-\frac{x_{\rm ev}^\top(Q_A-Q_B)x_{\rm tr}}{2M}
+\frac{x_{\rm tr}^\top(Q_A^\top Q_A-Q_B^\top Q_B)x_{\rm tr}}{8M}.$$

After averaging over splits and combining repeated label pairs,

$$T=c_0+\sum_k a_k\psi_k,\qquad\psi_k=x_{i_k}x_{j_k},\qquad S_1=\sum_k|a_k|.$$

Diagonal products equal one and enter `c0`. Public matrices also give a uniform bound `|Delta_V - T| <= epsilon`. `_bounds.py` evaluates the detailed paired bound. It preserves cancellation between candidates and takes the smaller of the valid paired and separate bounds. This step reads **no labels**.

## Queries and decisions

The 14 numbered steps below match the paper's Algorithm 1. The three stages separate public preparation, sequential querying, and final training. The small numerical guard used by the implementation is omitted from the mathematical display. The matching LaTeX source is [ALGORITHM1.tex](ALGORITHM1.tex); its equation reference uses the paper's definition of the CP limits.

```text
Input: public designs, fixed CV plan, lambda, g, delta, inspections N_1,...,N_J, final subset.
Stage 1: Label-free Preparation
 1. Approximation: obtain T = c0 + sum(a_k psi_k) and error bound epsilon.
 2. Set S1 = sum(abs(a_k)), cache C = {}, and I = [c0-S1-epsilon, c0+S1+epsilon].
 3. Go to Stage 3 if upper(I) <= g, lower(I) >= -g, or S1 = 0.
 4. Sampling law: set p_k = abs(a_k)/S1.
Stage 2: Sequential Label Queries
 5. For each prescribed inspection N:
 6.   Query: extend independent K_i ~ p draws to N; buy and cache missing labels.
 7.   Count: C_N = sum_{i=1}^N 1{sign(a[K_i])*psi[K_i] = 1}.
 8.   CP limits: compute (ell_N,u_N) with error probability delta/J.
 9.   Risk interval: I = [c0+S1*(2*ell_N-1)-epsilon, c0+S1*(2*u_N-1)+epsilon].
10.   Stop: if upper(I) <= g or lower(I) >= -g, break.
11. End for.
Stage 3: Selection and Final Training
12. If unresolved, complete exact CV through C; obtain I with numerical radius <= g.
13. Select A if upper(I) <= g; otherwise select B.
14. Buy missing final-subset labels through C and fit the selected candidate.
Output: selected candidate, supporting interval I, and distinct-label count |C|.
```

For `J` prespecified inspections, define the binary contribution bit and its accumulated success count:

$$B_i=\frac{1+\operatorname{sign}(a_{K_i})\psi_{K_i}}2,\qquad
C_N=\sum_{i=1}^N B_i.$$

Conditional on the fixed complete labels and features, the independently sampled indices make these bits independent Bernoulli observations. Their success probability is `pi`, and

$$C_N\sim\operatorname{Binomial}(N,\pi),\qquad T=c_0+S_1(2\pi-1).$$

This condition does not require independence between attributes, objects, or overlapping splits. A cached label supplies the same answer to a fresh random term draw. The draw still enters `C_N` and `N`, while no label is paid for twice.

Put `q = delta/(2J)`. The equal-tailed Clopper--Pearson (CP) limits are

$$\ell_N=\operatorname{Beta}^{-1}_q(C_N,N-C_N+1),\qquad
u_N=\operatorname{Beta}^{-1}_{1-q}(C_N+1,N-C_N),$$

with `ell_N = 0` at zero successes and `u_N = 1` at `N` successes. The beta quantiles invert binomial tails; they are not Bayesian credible limits. Mapping back to the score and adding the approximation bound gives

$$I_N=[c_0+S_1(2\ell_N-1)-\varepsilon,\ c_0+S_1(2u_N-1)+\varepsilon].$$

Each inspection receives total error probability `delta/J`. The union bound therefore covers all prescribed looks with probability at least `1-delta`. Stopping at a supported look preserves the choice guarantee. The failure probability applies separately to each specified comparison; it is not a promise that every simulated path has zero error.

CP is generally asymmetric about the sample mean. Decisions use the returned lower and upper endpoints directly. For interpretation only, the exact interval before adding `epsilon` has midpoint `m_N = c0 + S1*(ell_N + u_N - 1)` and half-length `h_N = S1*(u_N - ell_N)`. It supports a choice precisely when `h_N + epsilon <= g + abs(m_N)`.

`N` counts sampled terms, not new labels. A term needs at most two labels. Query probabilities stay fixed; the estimate, interval, and continuation decision update after each batch. The API keeps its original inspection schedule, whose first count is at least two. A zero coefficient mass bypasses sampling and uses the public-range or exact-completion branch.

The implementation adds outward numerical padding and checks that its fixed floating-point sampling distribution differs from the ideal coefficient law by less than the numerical guard. The mathematical interval above omits those small guards.

## Complete label use

Let `C` be the set of labels acquired for comparison and fallback, and `F` the labels required by the preassigned final subset. The complete count is `|C union F|`. The main algorithm uses zero pilot and zero calibration labels. If all pool labels are required for final training, acquisition cannot reduce that final union.

## The active-CV comparator

The comparator fits both candidates using complete training labels for a selected split. It actively queries evaluation attributes, corrects the samples by their probabilities, and uses sequential intervals inside and across splits. It reuses all acquired labels and may stop early. Thus the remaining difference concerns whether complete labels must be purchased for the training subsets used in comparison.

