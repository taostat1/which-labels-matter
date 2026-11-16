from pathlib import Path
import json
import numpy as np
from scipy.special import expit
from run import allocation, ROOT, save

def main():
    rng=np.random.default_rng(202611142305)
    error=0.;viol=0.
    for size in (7,128,384,512):
        va=rng.normal(size=(50,size));vb=rng.normal(size=(50,size))
        d0=np.logaddexp(0,va)-np.logaddexp(0,vb);d1=d0-va+vb
        c=((d0+d1)/2).mean(1);w=(d1-d0)/2
        R=np.max(abs(c)+abs(w).mean(1))*1.1
        pred=(expit(va)+expit(vb))/2
        floor=abs(w)/(size*(R-abs(c[:,None])))
        signed=rng.choice([-1.,1.],size=(50,size))
        scores=[.5*(np.logaddexp(0,va)+np.logaddexp(0,vb)-pred*(va+vb)),(1-pred)*abs(d0)+pred*abs(d1)]
        for score in scores:
            q=allocation(score,floor)
            values=c[:,None]+w*signed/(size*q)
            expected=(q*values).sum(1)
            truth=c+(w*signed).mean(1)
            error=max(error,float(np.max(abs(expected-truth))))
            viol=max(viol,float(np.max(abs(values)-R)))
    assert error<1e-12 and viol<1e-10
    files=sorted((ROOT/'results').glob('case*.json'))
    assert len(files)==96
    groups=set();solver=0.;ranges=-float('inf')
    for file in files:
        data=json.loads(file.read_text('utf-8'));rows=data['records']
        assert len(rows)==3
        key=(rows[0]['case_index'],data['group']);assert key not in groups;groups.add(key)
        solver=max(solver,data['validation']['max_solver_residual'])
        ranges=max(ranges,data['validation']['max_range_excess'])
        for row in rows:
            assert row['pilot_labels']==row['calibration_labels']==0
            assert row['error_probability']==.04
            assert row['labels']==sum(row['stage_labels'].values())
            assert not row['conflict']
            assert row['final_fit'] is not None if row['decision_issued'] else True
    result=dict(passed=True,external_records=288,groups=96,identity_checks=400,
                max_unbiasedness_error=error,max_synthetic_range_excess=viol,
                max_solver_residual=solver,max_recorded_range_excess=ranges,
                distinct_label_accounting=True,zero_preparation=True,common_error_probability=.04)
    save(ROOT/'validation.json',result)
    save(ROOT/'paper/evidence/population_external_validation.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
