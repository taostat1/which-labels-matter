"""Finite enumeration checks the nominal guarantee, not zero observed errors."""
import itertools
import unittest
import numpy as np
from scipy.stats import binom
from whichlabels._intervals import endpoints

class ExactBinomialChecks(unittest.TestCase):
    def test_fixed_horizon_coverage_and_boundary_counts(self):
        for n in [1,2,8,32,128]:
            lo,hi=np.array([endpoints(k,n,.05) for k in range(n+1)]).T
            self.assertEqual(lo[0],0)
            self.assertEqual(hi[-1],1)
            self.assertAlmostEqual(hi[0],1-.025**(1/n),10)
            self.assertAlmostEqual(lo[-1],.025**(1/n),10)
            for p in np.r_[0,.001,np.linspace(.01,.99,39),.999,1]:
                covered=(lo<=p)&(p<=hi)
                self.assertGreaterEqual(float(binom.pmf(np.arange(n+1),n,p)[covered].sum()),.95-1e-12)

    def test_simultaneous_inspections_by_full_path_enumeration(self):
        checks=[2,4,8];delta=.05
        intervals={(n,k):endpoints(k,n,delta/len(checks)) for n in checks for k in range(n+1)}
        for p in [.01,.05,.2,.5,.8,.95,.99]:
            failure=0.
            for bits in itertools.product([0,1],repeat=8):
                miss=any(not (intervals[n,sum(bits[:n])][0]<=p<=intervals[n,sum(bits[:n])][1]) for n in checks)
                if miss:failure+=p**sum(bits)*(1-p)**(8-sum(bits))
            self.assertLessEqual(failure,delta+1e-12)

if __name__=='__main__':unittest.main()
