import unittest
import numpy as np
from whichlabels import prepare_comparison, compare, audit_score, LabelCache, ArrayLabelProvider
from whichlabels.demo import synthetic_problem
from whichlabels._moments import audit_polynomial


class AlgorithmChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a, cls.b, cls.y, cls.plan, cls.final = synthetic_problem(
            seed=202611010911, objects=48, splits=3, train_size=6, test_size=6)
        cls.prepared = prepare_comparison(cls.a, cls.b, cls.plan, .125)

    def cache(self):
        return LabelCache(len(self.a), 4, ArrayLabelProvider(self.y))

    def test_cache_calls_provider_once_per_label(self):
        requests = []
        def provider(objects, attributes):
            requests.extend(zip(objects.tolist(), attributes.tolist()))
            return self.y[objects, attributes]
        cache = LabelCache(len(self.a), 4, provider)
        x = cache.query_keys(np.array([0, 0, 3]), 'comparison')
        cache.query_keys(np.array([0, 4]), 'final_training')
        self.assertEqual(len(requests), 3)
        self.assertEqual(cache.count, 3)
        self.assertEqual(cache.stage_counts, {'comparison': 2, 'final_training': 1})
        np.testing.assert_equal(x, self.y.ravel()[[0,0,3]])

    def test_no_provider_needed_for_public_preparation(self):
        self.assertGreater(self.prepared.polynomial['terms'], 0)
        self.assertTrue(np.isfinite(self.prepared.epsilon))
        self.assertLessEqual(self.prepared.epsilon, self.prepared.separate_epsilon)

    def test_polynomial_identity_and_bound(self):
        check = audit_polynomial(self.prepared.polynomial, self.y, self.plan,
                                 self.a, self.b, .125)
        exact = audit_score(self.prepared, self.y)
        self.assertLess(check['identity_error'], 1e-12)
        self.assertLessEqual(abs(exact['difference']-check['polynomial_delta']),
                             self.prepared.epsilon+exact['numerical_radius'])

    def test_entry_stop_buys_only_final_labels(self):
        p = prepare_comparison(self.a, self.a, self.plan, .125)
        result = compare(p, self.cache(), self.final, tolerance=.01)
        self.assertEqual(result['branch'], 'public_range')
        self.assertEqual(result['draws'], 0)
        self.assertEqual(result['comparison_labels'], 0)
        self.assertEqual(result['total_labels'], len(self.final)*4)

    def test_cap_fallback_and_union_accounting(self):
        result = compare(self.prepared, self.cache(), self.final, tolerance=.0001,
                         inspections=(2,), seed=11)
        self.assertTrue(result['fallback'])
        self.assertEqual(result['branch'], 'exact_cv_fallback')
        ids = np.unique(np.concatenate([self.final]+[x for split in self.plan for x in split]))
        self.assertEqual(result['total_labels'], len(ids)*4)
        self.assertEqual(result['total_labels'], sum(result['stage_labels'].values()))
        audit = audit_score(self.prepared, self.y)
        self.assertLessEqual(result['interval'][0], audit['difference'])
        self.assertGreaterEqual(result['interval'][1], audit['difference'])

    def test_identical_candidates_cancel_under_weak_penalties(self):
        # Regression: a nonpositive sufficient denominator used to force
        # unnecessary exact-CV annotation even though the two risks are equal.
        rng = np.random.default_rng(20261107)
        a = rng.normal(size=(8, 2, 3))
        a /= np.maximum(1, np.linalg.norm(a, axis=-1, keepdims=True))
        y = rng.integers(0, 2, (8, 2))
        plan = [(np.array([0, 1, 2]), np.array([3, 4, 5]))]
        final = np.array([5, 6, 7])
        for penalty in (.5, .125, .01, .001):
            with self.subTest(penalty=penalty):
                p = prepare_comparison(a, a, plan, penalty)
                result = compare(p, LabelCache(8, 2, ArrayLabelProvider(y)), final, tolerance=.01)
                self.assertEqual(p.epsilon, 0.)
                self.assertEqual(result['branch'], 'public_range')
                self.assertEqual(result['comparison_labels'], 0)
                self.assertEqual(result['total_labels'], 6)

    def test_identical_training_designs_cancel_parameter_error(self):
        from whichlabels._bounds import paired_preconditioned
        a = self.a[self.plan[0][0]]
        qa, qb = self.a[self.plan[0][1]], self.b[self.plan[0][1]]
        bound = paired_preconditioned(a, a, qa, qb, .001)
        self.assertEqual(bound['parameter_error_difference'], 0.)

    def test_active_cv_has_same_label_union_cap(self):
        result = compare(self.prepared, self.cache(), self.final, tolerance=.001,
                         method='active_cv', seed=17)
        ids = np.unique(np.concatenate([self.final]+[x for split in self.plan for x in split]))
        self.assertLessEqual(result['total_labels'], len(ids)*4)
        audit = audit_score(self.prepared, self.y)
        self.assertLessEqual(result['interval'][0], audit['difference'])
        self.assertGreaterEqual(result['interval'][1], audit['difference'])

    def test_invalid_features_and_checks_are_rejected(self):
        with self.assertRaises(ValueError):
            prepare_comparison(2*self.a, self.b, self.plan, .125)
        with self.assertRaises(ValueError):
            compare(self.prepared, self.cache(), self.final, tolerance=.01, inspections=(8,4))

    def test_invalid_provider_values_are_not_cached(self):
        cache = LabelCache(len(self.a),4,lambda i,j:np.full(len(i),.5))
        with self.assertRaises(ValueError):
            cache.query_keys(np.array([0,1]))
        self.assertEqual(cache.count,0)

    def test_bound_over_all_small_binary_labelings(self):
        rng = np.random.default_rng(21)
        a = rng.normal(size=(4,1,3))
        a /= np.maximum(1,np.linalg.norm(a,axis=-1,keepdims=True))
        b = .8*a
        plan = [(np.array([0,1]),np.array([2,3]))]
        prepared = prepare_comparison(a,b,plan,.5)
        for bits in range(16):
            y = np.array([(bits >> i)&1 for i in range(4)],dtype=float)[:,None]
            approx = audit_polynomial(prepared.polynomial,y,plan,a,b,.5)['polynomial_delta']
            exact = audit_score(prepared,y)
            self.assertLessEqual(abs(exact['difference']-approx),prepared.epsilon+exact['numerical_radius'])


if __name__=='__main__':
    unittest.main()

