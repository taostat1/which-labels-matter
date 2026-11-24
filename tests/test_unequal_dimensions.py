"""Exercise Algorithm 1 with a different feature width for each candidate."""
import itertools
import unittest

import numpy as np

from whichlabels import ArrayLabelProvider, LabelCache, audit_score, compare, prepare_comparison
from whichlabels._cache_bounds import approximation_bound
from whichlabels._moments import audit_polynomial


class UnequalDimensionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(1124)
        cls.a = rng.uniform(-1, 1, (8, 2, 1))
        cls.b = rng.normal(size=(8, 2, 3))
        cls.b /= np.maximum(1, np.linalg.norm(cls.b, axis=-1, keepdims=True))
        cls.y = (cls.b[..., 1] > 0).astype(float)
        first, second = np.arange(3), np.arange(3, 6)
        cls.plan = [(first, second), (second, first)]
        cls.final = np.array([1, 6, 7])
        cls.prepared = prepare_comparison(cls.a, cls.b, cls.plan, penalty=.1)

    def compare(self, **kwargs):
        cache = LabelCache(*self.y.shape, ArrayLabelProvider(self.y))
        return compare(self.prepared, cache, self.final, seed=13, **kwargs)

    def test_final_fit_on_public_stop_exact_fallback_and_active_cv(self):
        audit = audit_score(self.prepared, self.y)
        for method, tolerance, branch, choice in [
            ('selective', 3., 'public_range', 'A'),
            ('selective', 1e-6, 'exact_cv_fallback', 'B'),
            ('active_cv', 1e-6, 'active_cv', 'B'),
        ]:
            with self.subTest(branch=branch):
                result = self.compare(method=method, tolerance=tolerance, inspections=(2,))
                self.assertEqual((result['branch'], result['choice']), (branch, choice))
                self.assertEqual(len(result['final_parameter']), 1 if choice == 'A' else 3)
                self.assertLessEqual(result['interval'][0], audit['difference'])
                self.assertGreaterEqual(result['interval'][1], audit['difference'])
                self.assertEqual(result['total_labels'], sum(result['stage_labels'].values()))
                if branch == 'public_range':
                    self.assertEqual(result['comparison_labels'], 0)
                    self.assertEqual(result['total_labels'], self.y[self.final].size)
                elif branch == 'exact_cv_fallback':
                    self.assertEqual(result['total_labels'], self.y.size)

    def test_default_queries_and_refines_with_separate_bounds(self):
        settings = dict(tolerance=.05, inspections=(4, 16, 64, 256, 1024, 4096, 16384))
        default = self.compare(**settings)
        separate = self.compare(error_bound='separate', **settings)
        self.assertEqual(self.prepared.epsilon, self.prepared.separate_epsilon)
        self.assertEqual(default['branch'], 'selective_moments')
        self.assertTrue(any(row['refined'] for row in default['history']))
        self.assertLess(default['approximation_error_bound'], self.prepared.epsilon)
        for key in ['history', 'choice', 'interval', 'stage_labels', 'final_parameter']:
            self.assertEqual(default[key], separate[key])

    def test_bounds_and_polynomial_over_every_binary_labeling(self):
        rng = np.random.default_rng(24)
        a = rng.normal(size=(4, 1, 4))
        b = rng.normal(size=(4, 1, 3))
        for design in (a, b):
            design /= np.maximum(1, np.linalg.norm(design, axis=-1, keepdims=True))
        plan = [(np.array([0, 1]), np.array([2, 3]))]
        penalty = .25
        prepared = prepare_comparison(a, b, plan, penalty)
        cache = np.array([1., np.nan, 0., np.nan])
        local = approximation_bound(prepared, cache)
        for bits in itertools.product((0., 1.), repeat=4):
            labels = np.array(bits)[:, None]
            audit = audit_score(prepared, labels)
            gap = audit['difference']
            approx = audit_polynomial(prepared.polynomial, labels, plan, a, b, penalty)
            self.assertLess(approx['identity_error'], 1e-12)
            error = abs(gap - approx['polynomial_delta'])
            self.assertLessEqual(error, prepared.epsilon + audit['numerical_radius'])
            self.assertLessEqual(prepared.bounds['lower'][0], gap)
            self.assertGreaterEqual(prepared.bounds['upper'][0], gap)
            if bits[0] == 1. and bits[2] == 0.:
                self.assertLessEqual(error, local + audit['numerical_radius'])

    def test_rejects_misaligned_labels_and_empty_axes(self):
        for bad in (self.b[:-1], self.b[:, :1], self.b[:, 0], self.b[:0],
                    self.b[:, :0], self.b[:, :, :0]):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                prepare_comparison(self.a, bad, self.plan, penalty=.1)
