# Population comparison on the frozen finite pools

Each configuration uses the twelve recorded paths with group indices 6 through 17. The target is expected test loss over an independently drawn training set and test observation from the known pools. Every arm shares the feature matrices, penalty, tolerance, training distribution, test distribution, prespecified final-fit draw, and independent 2,048-fit audit. The audit mean estimates the target. The twelve paths in each configuration share that audit. Acceptance and interval coverage in the CSV refer to its mean.

## Comparators

Direct refits, active testing, and active model selection use the existing 11.18 cumulative betting results with error allocation .04. Each comparison block fully labels its random training set. Direct refits evaluate all attributes of the first member of the random test panel. Active testing and active model selection sample evaluation attributes using their existing proposals and inclusion weights.

The additional completion control uses the direct-refit block stream. Write its fully observed binary log-loss contrast for a test object as

$$D_t=c_t+\sum_a w_{ta}x_{ta},\qquad x_{ta}=2Y_{ta}-1,\qquad
 w_{ta}=-(v_{Ata}-v_{Bta})/(2r).$$

Here the public intercept, conditional on the paid training fit, is the average over attributes of
$\log(1+\exp v_{Ata})-\log(1+\exp v_{Bta})-(v_{Ata}-v_{Bta})/2$.
The arm substitutes known test labels and asks for unobserved attributes in descending order of $|w_{ta}|$ until the remaining coefficient mass is at most $g/4$. Thus, if $m_t$ is the partial sum and $s_t$ the remaining mass, the latent complete score lies in $[m_t-s_t,m_t+s_t]$. Intersect this interval with the common public score range to produce $[l_t,u_t]$.

For a fixed candidate mean, each positive betting factor is nondecreasing in its observation, and each negative factor is nonincreasing. On every path the positive wealth fed $l_t$ is bounded by the corresponding latent complete-block wealth. The negative wealth fed $u_t$ obeys the analogous inequality. The mixture of the prespecified fractions preserves both inequalities. Hence an anytime rejection from either sequence implies a rejection from the valid latent wealth with the same tail allocation. We invert the two wealths at total level .04, add the shared numerical refit allowance, and intersect across the same inspection grid. This allows cache-dependent within-block acquisition. The rule does not treat observed block bounds as a deterministic bound for the population mean.

## LEC controls

LEC and its fixed-bound control replay the existing zero-anchor paths. All new moment ablations use the same initial public checks, random feature blocks, inspections at 256 through 32768, geometric mixture fractions, query error .02, fallback error .02, and cache-refinement mechanism as the main method. Simulated design integration, where used, has its existing error allocation .005. The query law for each moment phase conditions on responses available at phase entry. All purchases are charged as they occur.

* **Whole objects.** Draw the same coefficient-weighted term after cache substitution. For every object touched by this draw, buy all its attributes. Newly paid attributes enter the next phase's reduction and approximation bound.
* **Uniform terms.** Uniformly draw among the $K$ nonzero terms in the reduced block polynomial. The unbiased contribution is $Z=c+K a_k\psi_k$. If $K=0$, use $Z=c$. With $M$ the guarded original bound on $|c|+\sum |a_k|$, a deterministic support for this implementation is $|Z|\le (4r^2+2r)M$. Before reduction there are at most $2r^2$ test-training products, $r(r-1)$ within-training products, $r^2$ between-training products, and $3r$ linear terms. Substitution and merging do not increase their number or their absolute mass. For $K\ge1$, $|c|+K\max |a_k|\le K M$. The latent full-polynomial mean still lies in $[-M,M]$. Reported uniform costs depend on this conservative support as well as the probability rule.
* **No early stopping.** Run the coefficient-weighted arm through all 32768 moment blocks. Use cumulative betting on all prior contributions, retain earlier valid intervals, and update the approximation bound. A supported result is returned at the cap. If unresolved there, use the unchanged independent paired-refit fallback. The public check leaves the choice open in all eight configurations.
* **Separate bounds.** Add the two candidate-specific bounds and update their sum from acquired labels. Every unresolved public check starts the weighted query stream, even if the initial sum exceeds the tolerance. On each path, the first supported choice stops acquisition. An unresolved comparison proceeds to independent paired refits after 32768 query blocks, with all previously purchased labels available for reuse.
* **Fixed bound.** Freeze the original entry approximation bound throughout acquisition and preserve the original query rule.

For an accepted choice the label total includes the prescribed final fit after cache reuse. An unresolved budget would be reported with all charges incurred and no final fit. Population fallback in the LEC ablations has the published independent streams and empirical-Bernstein inspection spending. When the exact independently seeded fallback statistics have been archived, the block sequence and label union are regenerated using each ablation's own cache. The four comparator arms start from direct fits, so their fallback field is always false.

The 24 T4 and V3 separate-bound paths are also archived by inspection in `../population/query_first_history.json.gz`. Each path reached 32,768 query blocks before refitting. The stage counts charge only previously unseen object–attribute labels.

## Execution and audits

The manuscript reporter uses `../population/paths.csv.gz` for stage charges and decisions, and `../initialization_paths.csv.gz` for the matched paid and zero starts. Each row gives the configuration, method, path, stopping interval, audit target, choice and distinct-label charges by stage. The reporter checks the 24 revised separate-bound paths against their per-inspection histories. The replay script in `../../reference/aligned_20261121/population/replay_separate.py` accepts a prepared feature pool and reconstructs those comparisons.
