"""Cache certificates hold for every completion, including adaptive purchases."""
import itertools
import unittest
import numpy as np
from whichlabels import prepare_comparison, compare, audit_score, LabelCache, ArrayLabelProvider
from whichlabels._cache_bounds import approximation_bound
from whichlabels._moments import audit_polynomial
from whichlabels.demo import synthetic_problem


class CacheRefinementChecks(unittest.TestCase):
    def test_partial_cache_bound_over_every_compatible_completion(self):
        rng = np.random.default_rng(20261114)
        a = rng.normal(size=(6, 1, 3))
        a /= np.maximum(1., np.linalg.norm(a, axis=-1, keepdims=True))
        b = .7*a + .1*np.roll(a, 1, axis=0)
        plan = [(np.array([0, 1, 2]), np.array([3, 4, 5]))]
        p = prepare_comparison(a, b, plan, .1)
        cache = np.array([1., np.nan, 0., np.nan, 1., np.nan])
        paired = min(p.epsilon, approximation_bound(p, cache))
        separate = min(p.separate_epsilon, approximation_bound(p, cache, separate=True))
        for bits in itertools.product([0., 1.], repeat=3):
            y = cache.copy()
            y[np.isnan(y)] = bits
            y = y[:, None]
            exact = audit_score(p, y)
            approx = audit_polynomial(p.polynomial, y, plan, a, b, .1)['polynomial_delta']
            error = abs(exact['difference']-approx)
            self.assertLessEqual(error, paired+exact['numerical_radius'])
            self.assertLessEqual(error, separate+exact['numerical_radius'])

    def test_refinement_never_increases_matched_distinct_labels(self):
        a,b,y,plan,final = synthetic_problem(seed=202611010911,
            objects=48,splits=3,train_size=6,test_size=6)
        p = prepare_comparison(a,b,plan,.125)
        for seed in [7, 19, 53]:
            rows = [compare(p, LabelCache(len(a),4,ArrayLabelProvider(y)), final,
                tolerance=.01, seed=seed, inspections=(32,128,512,2048),
                refine_error=flag) for flag in [False,True]]
            self.assertLessEqual(rows[1]['total_labels'],rows[0]['total_labels'])
            radii=[h['approximation_radius'] for h in rows[1]['history']]
            self.assertTrue(all(x>=z for x,z in zip(radii,radii[1:])))
            self.assertEqual(rows[1]['total_labels'],sum(rows[1]['stage_labels'].values()))


if __name__ == '__main__':
    unittest.main()
