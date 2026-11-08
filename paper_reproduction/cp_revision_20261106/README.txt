CP revision reproducibility index

The current algorithm is installed from this package as whichlabels 0.2.1.
Its small executable example runs with:
  python -m whichlabels.demo --output demo_output

The full experiment source tree and regenerated evidence are supplied in the
separate companion archive CP_Revision_20261106.zip. Extract that archive,
read its root README.txt, and run from:
  reproduction/experiments/shared_resource/logistic_cp_revision_20261106

The archive includes the earlier controlled simulation, repeated-split and complete
validation-coverage experiments, ordinary fivefold checks, cache ablation,
population-risk extension, interval audits and dependency manifests.
Real-data feature/label caches are not redistributed. Their required layout
and LOGISTIC_DATA_ROOT are documented in the companion README.

For the current main simulation table, use Simulation_Refinement_20261106.zip.
Its penalty grid is 0.25, 0.5, and 2, with 100 independent pools and all twelve
configurations retained. The CP archive's 0.125 simulation cells supply the
weak-penalty appendix checks. The two experiment versions should not be mixed.

Version 0.2.1 handles exact equality of training designs without requiring
a positive sufficient contraction denominator. Frozen experimental snapshots
retain their recorded source hashes; this boundary-case correction does not
change their reported paths.

Only this index is included here; the large evidence tree is deliberately
kept outside the installable package. The older sibling reproduction folders
retain the historical empirical-Bernstein experiments. They do not produce
the current CP tables. This local revision does not imply a GitHub release
or upload of the companion archive.
