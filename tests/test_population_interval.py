"""Check the paper's bounded-mean mixture on adaptive observation laws."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

SOURCE=Path(__file__).resolve().parents[1]/'paper_reproduction/current/reference/logistic_population_betting_20261115/intervals.py'
spec=importlib.util.spec_from_file_location('population_intervals',SOURCE)
iv=importlib.util.module_from_spec(spec);spec.loader.exec_module(iv)

class PopulationIntervalChecks(unittest.TestCase):
    def test_boundary_and_degenerate_ranges(self):
        self.assertEqual(iv.interval([],1),(-1,1))
        self.assertEqual(iv.interval([0,0],0),(0,0))
        self.assertEqual(iv.interval([-1]*8,1)[0],-1)
        self.assertEqual(iv.interval([1]*8,1)[1],1)

    def test_adaptive_laws_keep_the_mean_and_crossing_control(self):
        for mu in (-.4,.1,.7):
            alpha=.1;total=badmass=0.;capital=np.zeros(2)
            pending=[([],1.,False)]
            while pending:
                values,weight,bad=pending.pop()
                if values:
                    log=np.array([iv.log_capital(values,mu,1,side) for side in (1,-1)])
                    bad=bad or np.max(log)>np.log(2/alpha)
                if len(values)==9:
                    total+=weight;badmass+=weight*bad;capital+=weight*np.exp(log);continue
                low,high=((-1,(mu+1)/2) if not values or values[-1]<=mu else ((mu-1)/2,1))
                prob=(mu-low)/(high-low)
                pending.extend([(values+[high],weight*prob,bad),(values+[low],weight*(1-prob),bad)])
            self.assertAlmostEqual(total,1);self.assertLessEqual(badmass,alpha+1e-12)
            np.testing.assert_allclose(capital,1,rtol=1e-10,atol=1e-10)

    def test_outward_inversion_contains_all_accepted_grid_values(self):
        z=np.random.default_rng(92817).uniform(-1,1,127)
        L,U=iv.interval(z,1,.05)
        for m in np.linspace(-.999,.999,199):
            accept=max(iv.log_capital(z,m,1,side) for side in (1,-1))<=np.log(40)
            self.assertEqual(accept,L<=m<=U)

if __name__=='__main__':unittest.main()
