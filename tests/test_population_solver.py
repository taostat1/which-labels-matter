"""The evaluation proposal must not depend on other fitted blocks."""
import ast
from pathlib import Path
import unittest

import numpy as np
from scipy.special import expit


class PopulationSolverChecks(unittest.TestCase):
    def test_fit_and_proposal_are_invariant_to_batch_grouping(self):
        source = (Path(__file__).resolve().parents[1] / 'paper_reproduction/current'
                  / 'reference/logistic_submission_audit_20261117/baselines/run.py')
        tree = ast.parse(source.read_text(encoding='utf-8'))
        scope = {'np': np, 'expit': expit}
        for name in ('independent_batch_fit', 'allocation', 'fitted_residual'):
            function = next(n for n in tree.body
                            if isinstance(n, ast.FunctionDef) and n.name == name)
            exec(compile(ast.Module(body=[function], type_ignores=[]),
                         str(source), 'exec'), scope)
        fit = scope['independent_batch_fit']
        rng = np.random.default_rng(2026111731)
        for penalty in (.001, .125, .5, 2.):
            z = rng.normal(size=(12, 32, 6, 9))
            z /= np.maximum(1, np.linalg.norm(z, axis=-1))[..., None]
            y = rng.integers(0, 2, z.shape[:-1])
            start = np.zeros(9)
            full, _ = fit(z, y, penalty, start)
            single = np.concatenate([fit(z[i:i+1], y[i:i+1], penalty, start)[0]
                                     for i in range(12)])
            permutation = rng.permutation(12)
            shuffled = fit(z[permutation], y[permutation], penalty, start)[0]
            shuffled = shuffled[np.argsort(permutation)]
            split = np.concatenate([fit(z[i:i+3], y[i:i+3], penalty, start)[0]
                                    for i in range(0, 12, 3)])
            for result in (single, shuffled, split):
                np.testing.assert_allclose(result, full, rtol=0, atol=1e-13)
            residual = scope['fitted_residual'](z, y, full, penalty)
            self.assertLessEqual(float(residual.max()), 1e-10)
            panel = rng.normal(size=(12, 64, 6, 9))
            panel /= np.maximum(1, np.linalg.norm(panel, axis=-1))[..., None]
            proposals = []
            for theta in (full, single, shuffled, split):
                a = np.einsum('bprd,bd->bpr', panel, theta).reshape(12, -1)
                b = .75 * a
                sa, sb = np.logaddexp(0, a), np.logaddexp(0, b)
                pred = (expit(a) + expit(b)) / 2
                d0, d1 = sa - sb, sa - sb - a + b
                scores = (.5 * (sa + sb - pred * (a + b)),
                          (1-pred) * abs(d0) + pred * abs(d1))
                proposals.append([scope['allocation'](s, np.zeros_like(s))
                                  for s in scores])
            for result in proposals[1:]:
                np.testing.assert_allclose(result, proposals[0], rtol=0, atol=1e-13)


if __name__ == '__main__':
    unittest.main()
