"""Decision auditing and label-cache reconstruction, without resource prices."""
import numpy as np
from study import CASES, SEED, decision

def adaptive_refit_intervals(rec):
    """Reserve calibration error after pilot-dependent selection of refits.

    Recover the empirical standard error from the recorded .05 interval and
    evaluate it at .04. The recorded .02 path extends at least as far.
    """
    eps=2e-10/rec['case']['lam'];M=rec['bounds']['R']
    t0=np.log(32/.05);t1=np.log(32/.04)
    for row in rec['checkpoints']:
        if row['method']!='refit':continue
        n=row['blocks'];standard=max(0.,row['single']['radius']-eps-14*M*t0/(3*(n-1)))
        rad=standard*np.sqrt(t1/t0)+14*M*t1/(3*(n-1))+eps+1e-12
        lo=row['mean']-rad;hi=row['mean']+rad
        row['adaptive']=dict(lower=float(lo),upper=float(hi),radius=float(rad),
              decision=decision(lo,hi,rec['case']['g']),both=bool(lo>=-rec['case']['g'] and hi<=rec['case']['g']))

def first_row(rec,method,mode='single'):
    rows=[v for v in rec['checkpoints'] if v['method']==method]
    return None if not rows else next((v for v in rows if v[mode]['decision']!='unresolved'),rows[-1])


def refit_requests(rec,pools,r,st):
    case=rec['case'];ci=next(i for i,c in enumerate(CASES) if c['id']==case['id'])
    row=first_row(rec,'refit');rng=np.random.default_rng(SEED+7000000+ci*100000+rec['rep'])
    rng.choice(pools['training'],(64,case['n']));rng.choice(pools['comparison'],64);rng.random(64)
    seen=np.zeros(max(int(pools['training'].max()),int(pools['comparison'].max()))+1,bool)
    seen[np.asarray(st['pilot'])]=True
    for start in range(0,row['blocks'],128):
        ids=rng.choice(pools['training'],(128,case['n']));test=rng.choice(pools['comparison'],128);rng.random(128)
        seen[ids.ravel()]=True;seen[test]=True
    keys=set((np.flatnonzero(seen)[:,None]*r+np.arange(r)).ravel().tolist())
    return keys,row


def assess(out,audit,g):
    delta=audit['difference']['mean'];se=audit['difference']['se'];lo,hi=delta-1.96*se,delta+1.96*se
    out['audit_gap']=delta;out['audit_se']=se
    out['covered_audit_mean']=out['lower'] is not None and out['lower']<=delta<=out['upper']
    out['excludes_audit_band']=out['lower'] is not None and (out['lower']>hi or out['upper']<lo)
    out['unacceptable_audit_mean']=(out['decision']=='A' and delta>g) or (out['decision']=='B' and delta<-g)
    out['unacceptable_audit_band']=(out['decision']=='A' and lo>g) or (out['decision']=='B' and hi<-g)
    return out
