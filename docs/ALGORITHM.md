# Algorithm 1: Label-Efficient Comparison (LEC) for a Fixed CV Plan

This guide describes Algorithm 1 of **Label-Efficient Comparison of Shared Logistic Models with Finite-Sample Guarantees**. The public `whichlabels.compare` API implements this fixed-plan algorithm. [Algorithm 2](POPULATION.md) extends the paper's analysis to expected test risk over random training sets.

## Why partial labels suffice

Model comparison needs a reliable estimate of the performance difference. With public features fixed, one Newton update from zero makes the approximate coefficients weighted sums of training labels coded as `-1` or `1`. A quadratic loss expansion then expresses the approximate score difference through label products with known weights. LEC estimates this sum by querying selected products, with at most two labels per draw. It adds a feature-based approximation bound to the sampling interval and stops when that interval supports a choice about the exact CV scores.

## Statistical target

There are `m` objects and `r` binary attributes per object. Candidate `j` uses a public feature row `z_j(i,a)` of length `d_j` for each object and attribute. Its coefficient vector is shared across attributes. The two candidates may use different feature widths. Each row has norm at most one. All coefficients receive an L2 penalty `lambda > 0`. Cross-validation (CV) refers to the prescribed training/evaluation splits below.

The prediction is $\widehat p_{ja}(X)=\sigma(z_{ja}(X)^\top\widehat\theta_j)$. Changing candidates changes the feature map and its fitted coefficient vector. Within a candidate, changing attributes changes the feature row while retaining the same vector. Three applications illustrate these distinctions.

- Dehghani et al. compare five ViT representations on the same CelebA faces for each of two attributes, attractiveness and smiling. This illustrates comparing feature maps with a common response. [ICML 2023, Figure 7](https://proceedings.mlr.press/v202/dehghani23a/dehghani23a.pdf#page=7)
- MedKLIP applies one two-class linear head to disease-specific representations of a chest radiograph. The difference between its two weight vectors gives the shared logistic direction. [Wu et al., ICCV 2023, Eq. 6](https://openaccess.thecvf.com/content/ICCV2023/papers/Wu_MedKLIP_Medical_Knowledge_Enhanced_Language-Image_Pre-Training_for_X-ray_Diagnosis_ICCV_2023_paper.pdf#page=4) · [Author implementation](https://github.com/MediaBrain-SJTU/MedKLIP/blob/adc2eb8672a37ad7e41b365aea41ac3af29878e2/PreTrain_MedKLIP/models/model_MedKLIP.py#L181)
- RAM++ applies the same scalar linear head and sigmoid to every tag representation. Two tags on the same image can therefore receive different probabilities from the same scoring vector. RAM++ trains with asymmetric loss. LEC fits a ridge-penalized binary logistic loss on fixed features. [Huang et al., ACM Multimedia 2025](https://doi.org/10.1145/3746027.3755316) · [Author implementation](https://github.com/xinyu1205/recognize-anything/blob/560542c0f4c5f0f60480de3fa9ea6cf483b50bb3/ram/models/ram_plus.py#L311-L325)

For a training subset `S`, the exact parameter minimizes

$$
\widehat\theta_{j,S}=\arg\min_\theta\left\{
\frac{1}{|S|r}\sum_{i\in S,a}\left[\log(1+e^{z_j(i,a)^\top\theta})-Y_{ia}z_j(i,a)^\top\theta\right]
+\frac\lambda2\|\theta\|^2\right\}.
$$

With label factors held fixed, Yu et al. obtain a ridge-penalized logistic fit on observed object-attribute pairs. They sum losses over pairs, so the numerical penalty scale depends on the loss normalization. [ICML 2014, Eq. (6)](https://proceedings.mlr.press/v32/yu14.pdf#page=4)

A fixed plan contains `B` pairs of training and evaluation subsets. Its target difference is

$$\Delta_{\mathcal V}=\frac1B\sum_{b=1}^B
\bigl[R_{A,b}(\widehat\theta_{A,b})-R_{B,b}(\widehat\theta_{B,b})\bigr].$$

`R` is mean evaluation binary cross-entropy (BCE), without the training penalty. A is acceptable if `Delta <= g`; B is acceptable if `Delta >= -g`. Both can be acceptable. The guarantee is conditional on the fixed pool and plan; `delta` bounds the probability of an unacceptable choice over the randomized queries.

The tolerance also has a relative interpretation. Write `G_j = exp(-mean_CV_loss_j)` for the geometric mean of probabilities assigned to the observed validation labels, with the same weights as the CV score. An acceptable choice satisfies

$$\mathcal G_{\widehat j}\ge e^{-g}\max(\mathcal G_A,\mathcal G_B).$$

Thus `g = 0.005` allows a relative decrease of at most `1 - exp(-0.005)`, or about **0.499%**, in this score. An allowed decrease `rho` corresponds to `g = -log(1-rho)`.

| Symbol | Meaning |
|---|---|
| `n`, `r`, `B` | Training objects per fit, attributes per object, and splits |
| `Delta_V` | Exact candidate-A-minus-candidate-B score on the fixed plan |
| `T`, `epsilon_0`, `epsilon_N` | Fixed approximate score difference, public error bound, and cache-refined bound |
| `a_k`, `psi_k` | Public coefficient and signed-label product |
| `S1`, `p_k` | Sum of absolute coefficients and term-query probabilities |
| `g`, `delta` | Excess-risk tolerance and selection-error probability bound |
| `N`, `J` | Sampled terms at an inspection and number of inspections |
| `C`, `F` | Acquired-label cache and prescribed final-training labels |

In the matrix derivation below, `N = n r` instead counts training feature rows. Neither use of `N` is the number of distinct labels purchased.

## Public preparation

Write each binary label as `x = 2Y - 1`. For one split, let `Z_j` contain the training rows, `W_j` the evaluation rows, and `N = n r`, `M = n_test r`.

$$H_j=\lambda I+\frac{Z_j^\top Z_j}{4N},\qquad
L_j=\frac{H_j^{-1}Z_j^\top}{2N},\qquad t_j=L_jx_{\rm tr},\qquad Q_j=W_jL_j.$$

The zero-anchor Newton update is linear in the signed labels. The quadratic BCE expansion gives the paired split statistic

$$T_b=-\frac{x_{\rm ev}^\top(Q_A-Q_B)x_{\rm tr}}{2M}
+\frac{x_{\rm tr}^\top(Q_A^\top Q_A-Q_B^\top Q_B)x_{\rm tr}}{8M}.$$

After averaging over splits and combining repeated label pairs,

$$T=c_0+\sum_k a_k\psi_k,\qquad\psi_k=x_{i_k}x_{j_k},\qquad S_1=\sum_k|a_k|.$$

Diagonal products equal one and enter `c0`. Both `Q_A` and `Q_B` map the same training labels to the same validation rows, so their difference is defined for any feature widths. Public matrices give a uniform bound `|Delta_V - T| <= epsilon`. With common coefficient coordinates, LEC computes the paired and separate bounds and uses the tighter valid bound. With different widths, it adds the two individual loss-error bounds. Cache refinement uses the same rule after labels arrive.

## Queries and decisions

The steps below follow the paper's Algorithm 1. The three stages separate public preparation, sequential querying, and final training. The small numerical guard used by the implementation is omitted from the mathematical display. The matching LaTeX source is [ALGORITHM1.tex](ALGORITHM1.tex).

```text
Input: public designs, fixed CV plan, lambda, g, delta, inspections N_1,...,N_J, final subset.
Stage 1: Label-free Preparation
 1. Initialize both anchors to zero and create an empty cache C.
 2. Approximation: obtain T = c0 + sum(a_k psi_k) and error bound epsilon.
 3. Set S1 = sum(abs(a_k)), and I = [c0-S1-epsilon, c0+S1+epsilon].
 4. Go to Stage 3 if upper(I) <= g, lower(I) >= -g, or S1 = 0.
 5. Sampling law: set p_k = abs(a_k)/S1.
Stage 2: Sequential Label Queries
 6. For each prescribed inspection N:
 7.   Query: extend independent K_i ~ p draws to N; buy and cache missing labels.
 8.   Count: C_N = sum_{i=1}^N 1{sign(a[K_i])*psi[K_i] = 1}.
 9.   CP limits: compute (ell_N,u_N) with error probability delta/J.
10.   Refine epsilon using cached labels if the raw CP interval supports a choice
      but its enlargement by the current epsilon does not; retain the running minimum.
11.   Risk interval: I = [c0+S1*(2*ell_N-1)-epsilon, c0+S1*(2*u_N-1)+epsilon].
12.   Stop: if upper(I) <= g or lower(I) >= -g, break.
13. End for.
Stage 3: Selection and Final Training
14. If unresolved, complete exact CV through C; obtain I with numerical radius <= g.
15. Select A if upper(I) <= g; otherwise select B.
16. Buy missing final-subset labels through C and fit the selected candidate.
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

$$I_N=[c_0+S_1(2\ell_N-1)-\varepsilon_N,\ c_0+S_1(2u_N-1)+\varepsilon_N].$$

At a refinement, `_cache_bounds.py` encloses the Newton scores for every binary completion agreeing with the acquired labels. Local curvature and gradient bounds control the resulting risk error. The algorithm retains

$$\varepsilon_N=\min\{\varepsilon_{N^-},\varepsilon_{\rm loc}(\mathcal C_N)\}.$$

The true labels belong to that completion set, including after adaptive purchases. Thus this deterministic bound can use the same labels as the CP statistic without a new independence assumption or error allocation. The code refines only when the raw sampling interval could support a decision. At other checkpoints it keeps the preceding bound. This schedule leaves the polynomial, proposal, and cumulative counts unchanged.

Each inspection receives total error probability `delta/J`. The union bound therefore covers all prescribed looks with probability at least `1-delta`. Stopping at a supported look preserves the choice guarantee. The failure probability applies separately to each specified comparison; it is not a promise that every simulated path has zero error.

CP is generally asymmetric about the sample mean. Decisions use the returned lower and upper endpoints directly. For interpretation only, the exact interval before adding `epsilon` has midpoint `m_N = c0 + S1*(ell_N + u_N - 1)` and half-length `h_N = S1*(u_N - ell_N)`. It supports a choice precisely when `h_N + epsilon <= g + abs(m_N)`.

`N` counts sampled terms, not new labels. A term needs at most two labels. Query probabilities stay fixed; the estimate, error bound, interval, and continuation decision can update after each batch. The API keeps its original inspection schedule, whose first count is at least two. A zero coefficient mass bypasses sampling and uses the public-range or exact-completion branch. With the same random stream and final subset, the refined rule acquires no more labels than `refine_error=False`.

The implementation adds outward numerical padding and checks that its fixed floating-point sampling distribution differs from the ideal coefficient law by less than the numerical guard. The mathematical interval above omits those small guards.

## Complete label use

Let `C` be the set of labels acquired for comparison and fallback, and `F` the labels required by the preassigned final subset. The complete count is `|C union F|`. The main algorithm uses zero pilot and zero calibration labels. If all pool labels are required for final training, acquisition cannot reduce that final union.

## The active-CV comparator

The comparator fits both candidates using complete training labels for a selected split. It actively queries evaluation attributes, corrects the samples by their probabilities, and uses sequential intervals inside and across splits. It reuses all acquired labels and may stop early. Thus the remaining difference concerns whether complete labels must be purchased for the training subsets used in comparison.

## How the guarantees support the algorithm

| Paper result | Role |
|---|---|
| Theorem 1 | Bounds the paired approximation error using public features in Stage 1 |
| Theorem 2 | Covers the exact CV score at every prescribed inspection in Stage 2 |
| Theorem 3 | Controls selection error and total distinct labels through fallback and final training |
| Theorem 4 and Corollary 3 | Give an information lower bound and matching-order conditions; these results benchmark acquisition rather than drive the implementation |

The matching order requires sensitivity, dispersion, and approximation conditions. Algorithm 1's upper bound also includes the logarithmic cost of the prescribed inspections. The population extension uses Theorems 5-7 and a different interval construction; see [Algorithm 2](POPULATION.md).

