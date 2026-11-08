"""Check the clean interface against the frozen experiment's numerical core."""
from pathlib import Path
import importlib.util
import unittest
import numpy as np
from whichlabels.demo import synthetic_problem
from whichlabels import prepare_comparison, LabelCache, ArrayLabelProvider
from whichlabels._moments import query_trial


class ReferenceParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        source = root/'paper_reproduction/main_cv/experiments/shared_resource/logistic_cv_attribution_20261031/run.py'
        spec = importlib.util.spec_from_file_location('frozen_attribution', source)
        cls.ref = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(cls.ref)
        except ModuleNotFoundError as exc:
            raise unittest.SkipTest('Install .[reproduce] for archived-code checks: '+str(exc))

    def test_polynomial_bound_query_weights_and_path(self):
        a,b,y,plan,final = synthetic_problem(seed=202611010922,objects=48,
                                            splits=3,train_size=6,test_size=6)
        prepared=prepare_comparison(a,b,plan,.125)
        old=self.ref.base.polynomial(a,b,plan,.125)
        public=self.ref.previous.public_bounds(a,b,plan,.125)
        paired=np.mean([self.ref.paired_preconditioned(a[tr],b[tr],a[te],b[te],.125)['epsilon'] for tr,te in plan])
        old['epsilon']=min(paired,public['epsilon'])
        proposal=self.ref.base.query_plan(old,self.ref.zero_helper(len(a),4),4)
        for key in ['left','right','coef']:
            np.testing.assert_allclose(prepared.polynomial[key],old[key],rtol=0,atol=1e-14)
        self.assertAlmostEqual(prepared.epsilon,old['epsilon'],14)
        np.testing.assert_allclose(prepared.proposal['q'],proposal['q'],rtol=0,atol=1e-15)
        current_cache=LabelCache(len(a),4,ArrayLabelProvider(y))
        reference_cache=self.ref.base.Oracle(y)
        current=query_trial(prepared.polynomial,prepared.proposal,current_cache,
                            np.random.default_rng(7),[8,16,32],1e-8,.05)
        reference=self.ref.base.query_trial(old,proposal,reference_cache,
                            np.random.default_rng(7),[8,16,32],1e-8,.05)
        self.assertEqual(current_cache.count,reference_cache.count)
        self.assertEqual(current['draws'],reference['draws'])
        # The polynomial, query indices, purchased labels and point estimates
        # still agree. Interval endpoints intentionally change from EB to CP.
        np.testing.assert_array_equal(current_cache.seen,reference_cache.seen)
        for new,old in zip(current['history'],reference['history']):
            self.assertAlmostEqual(new['estimate'],old['estimate'],13)
            self.assertEqual(new['interval'],'CP')


if __name__=='__main__':
    unittest.main()
