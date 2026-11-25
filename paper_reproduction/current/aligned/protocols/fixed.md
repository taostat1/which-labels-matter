Two matched fixed-plan applications
==================================

This replay extends the 20-pool, three-path repeated-split and full-validation
studies. It uses the registered pools, fit/evaluation indices, final subsets
and random seeds in logistic_cp_revision_20261106. Within an application all
methods target the same 32-split mean optimized log loss. An independent
complete-label audit is kept separate from acquisition and stopping. Reports
include final-fit labels, count each object-attribute purchase once and use
the sample SD of the 20 pool means.

Both designs compare the proposed method, active CV, the fixed-plan active
testing and AMS adaptations, and the completion certificate. Both also include
whole-object purchases, uniform terms, separate bounds, and acquisition to
the cap. The cap arm uses identical weighted draws and cumulative CP intervals
with alpha=.05/9, retains the usual cache-update rule, suppresses entry and
intermediate stops, and applies the choice rule after 32768 draws. An
unresolved interval triggers exact CV at the cap. Full-pool annotation is
retained as the reference denominator for the full-coverage application.

The archived study checked each exact target against a separately labeled ledger. The full aligned path record in `../paths.csv.gz` contains the repeated-split and complete-coverage results. The fixed schedules and seeds are in `../fixed/protocol.json`. The manuscript reporter reconstructs Table 2 from the path ledger and Figure 3 from recorded label counts and object coverage.
