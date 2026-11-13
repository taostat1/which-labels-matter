# Algorithm 2: LEC for Population Risk

Algorithm 2 is the population-risk extension in Section 4.4 of **Label-Efficient Comparison of Shared Logistic Models with Finite-Sample Guarantees**. It compares training rules that will be reused on random training sets of a fixed size. The installable `whichlabels.compare` API implements [Algorithm 1](ALGORITHM.md); population experiments use the research scripts in the companion archive listed in [REPRODUCING.md](REPRODUCING.md).

## Target and preparation

For training law `P`, test law `Q`, and training size `n`, the target is

$$R_j(n)=\mathbb E_{S\sim P^n}L_j(\widehat\theta_j(S)),
\qquad L_j(\theta)=\mathbb E_{W\sim Q}\ell_j(\theta;W),
\qquad \Delta_n=R_A(n)-R_B(n).$$

Algorithm 1 conditions on a fixed pool and CV plan. Algorithm 2 averages test risk over random training sets. This additional randomness requires independent calibration of the approximation error. A fixed-plan interval supports a conclusion about that plan; transferring it to `Delta_n` also requires controlling variation under data sampling.

The extension uses normalized public features, binary attributes, and a positive L2 penalty. Training observations are independent, test observations are independent of training observations, and pilot, calibration, and comparison blocks use independent draws. Attributes within an object may be dependent. Public test-feature expectations are available or supplied with certified integration errors.

An independent pilot fits anchors `theta_A0` and `theta_B0`. Separate calibration bounds score and gradient moments, giving

$$|\Delta_n-\widetilde\Delta_n|\le\varepsilon,
\qquad \varepsilon=\varepsilon_A+\varepsilon_B.$$

Here `tilde_Delta_n` is the mean approximate risk difference conditional on preparation. The bound includes any certified feature-integration error. Theorem 5 supplies a global approximation bound, and Theorem 6 refines it using local curvature and calibrated moments.

## Query statistic and interval

Theorem 7 represents the approximate risk difference using one test object and two distinct training positions. Its coefficients use all training features, the pilot, and public test-feature expectations. Sampling one term needs at most two labels. Prediction assistance uses

$$Z_{\mathrm{pred}}
=a_0+\sum_k a_km_k(H)
+\frac{a_K}{p_K(H)}\{\psi_K-m_K(H)\}.$$

`H` contains public features and preparation quantities; `m_k(H)` predicts a signed-label product; `p_k(H)` is its known positive query probability. The inverse-probability residual correction preserves the mean of the approximate statistic.

The coefficients vary across feature blocks. The statistic therefore generally takes more than the two fixed values used by Algorithm 1's Clopper-Pearson (CP) construction. Algorithm 2 uses an empirical Bernstein interval for bounded variables. For certified range width `W`, unbiased sample variance `v_hat_N`, and `J` prescribed inspections,

$$h_N^{\mathrm{pop}}
=\sqrt{\frac{2\widehat v_N\log(4J/\delta_s)}{N}}
+\frac{7W\log(4J/\delta_s)}{3(N-1)},\qquad N\ge2.$$

At each inspection it uses

$$I_N=[\overline Z_N-h_N^{\mathrm{pop}}-\varepsilon,\,
\overline Z_N+h_N^{\mathrm{pop}}+\varepsilon].$$

The error allocation satisfies `delta_c + delta_s + delta_r <= delta` for calibration, sampling, and paired refits, respectively. Candidate A is supported if `upper(I_N) <= g`, and B if `lower(I_N) >= -g`.

## Three stages

1. **Independent preparation.** Acquire pilot and calibration labels, fit anchors, certify the approximation bound, and fix product predictors and the range-constrained query rule.
2. **Selective comparison.** If `epsilon < g`, query independent blocks in batches within the trial budget. Update the bounded-variable interval and check whether it supports a choice.
3. **Refits and final training.** If still unresolved, switch to independent paired exact refits with separate statistics and error allocation. Stop with a supported choice or return `Unresolved` when the budget ends. For a supported choice, draw an independent final sample of size `n`, acquire its missing labels, and fit the selected candidate.

Preparation, queries, refits, and final training share the label cache. The total count is the number of distinct labels acquired. The exact manuscript pseudocode is [ALGORITHM2.tex](ALGORITHM2.tex); its equation reference is resolved within the paper.

## Reproduction

The current extension experiments are in `CP_Revision_20261106.zip`, distributed separately from this compact package. The archive preserves the population scripts and preparation schedule. See [the reproduction guide](REPRODUCING.md) for the archive index and data-cache requirements.
