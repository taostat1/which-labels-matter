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

The following summarizes the implemented flow. The small numerical guard used by the implementation is omitted from the mathematical display.

```text
1. Fix feature maps, lambda, tolerance g, error probability delta,
   CV splits, inspection times, query cap, and the final training subset.
2. Compute c0, coefficients a, S1, and epsilon from public features.
3. Set I = [c0 - S1 - epsilon, c0 + S1 + epsilon].
4. If I supports a candidate, go to step 11.
5. Fix p[k] = abs(a[k]) / S1.
6. At each inspection, extend an independent sequence of K ~ p draws.
7. Buy missing labels in each sampled product through the shared cache.
8. Form Z = c0 + a[K] * psi[K] / p[K].
9. Update the mean, sample variance, and I = mean +/- (h + epsilon).
10. Stop if upper(I) <= g or lower(I) >= -g; otherwise continue to the cap.
    If still unresolved, complete exact CV through the same cache.
11. Choose A if upper(I) <= g; otherwise choose B.
12. Buy missing final-subset labels and fit the chosen candidate once.
13. Return the candidate, interval, distinct-label count, and elapsed time.
```

For `J` prespecified inspections and `N >= 2` term draws, the interval radius is

$$h_N=\sqrt{\frac{2\widehat v_N\log(4J/\delta)}{N}}
+\frac{14S_1\log(4J/\delta)}{3(N-1)},\qquad
I_N=[\overline Z_N-h_N-\varepsilon,\overline Z_N+h_N+\varepsilon].$$

`N` counts sampled terms, not new labels. A term needs at most two labels, and repeated requests are free in the distinct-label count. Query probabilities stay fixed. The estimate, interval, and continuation decision are updated as observations arrive.

## Complete label use

Let `C` be the set of labels acquired for comparison and fallback, and `F` the labels required by the preassigned final subset. The complete count is `|C union F|`. The main algorithm uses zero pilot and zero calibration labels. If all pool labels are required for final training, acquisition cannot reduce that final union.

## The active-CV comparator

The comparator fits both candidates using complete training labels for a selected split. It actively queries evaluation attributes, corrects the samples by their probabilities, and uses sequential intervals inside and across splits. It reuses all acquired labels and may stop early. Thus the remaining difference concerns whether complete labels must be purchased for the training subsets used in comparison.

