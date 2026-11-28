"""Exercise Algorithm 1 with a different feature width for each candidate."""
import itertools
import unittest

import numpy as np

from whichlabels import ArrayLabelProvider, LabelCache, audit_score, compare, prepare_comparison
from whichlabels._moments import audit_polynomial
from whichlabels._prediction_bound import bound, split_allowances


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

    def test_default_uses_prediction_bounds_for_public_and_cached_signs(self):
        def split_mean(signs):
            return np.mean([bound(self.a[tr], self.b[tr], self.a[ev], self.b[ev], .1,
                                  signs[tr].ravel()) for tr, ev in self.plan])

        signs = np.full(self.y.shape, np.nan)
        expected = split_mean(signs)
        default = self.compare(tolerance=.02, inspections=(4, 16, 64, 256, 1024, 4096))
        self.assertAlmostEqual(default['initial_approximation_error_bound'], expected)
        self.assertEqual(default['branch'], 'selective_moments')
        self.assertTrue(any(row['refined'] for row in default['history']))
        rng = np.random.default_rng(13)
        previous = 0
        for row in default['history']:
            draws = rng.random(row['draws'] - previous)
            indices = np.searchsorted(self.prepared.proposal['cdf'], draws, side='right')
            for side in ('left', 'right'):
                keys = self.prepared.polynomial[side][indices]
                signs.ravel()[keys] = 2 * self.y.ravel()[keys] - 1
            if row['refined']:
                expected = min(expected, split_mean(signs))
            self.assertAlmostEqual(row['approximation_radius'], expected)
            previous = row['draws']
        self.assertLess(expected, self.prepared.epsilon)

    def test_explicit_historical_controls_still_agree_at_unequal_widths(self):
        settings = dict(tolerance=.05, inspections=(4, 16, 64, 256, 1024, 4096, 16384))
        paired = self.compare(error_bound='paired', **settings)
        separate = self.compare(error_bound='separate', **settings)
        for key in ['history', 'choice', 'interval', 'stage_labels', 'final_parameter']:
            self.assertEqual(paired[key], separate[key])

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
        local = split_allowances(a, b, plan, penalty, cache).mean()
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
