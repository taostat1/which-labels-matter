"""Final-subset pilots for fixed-CV LEC; all choices use purchased labels only.

The original data, plans, final subsets, tolerances, and query seeds are fixed.
Always-pilot and triggered-pilot policies are evaluated without oracle tuning.
"""
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys, json, time, argparse, hashlib, contextlib, traceback, gc
ROOT = Path(__file__).resolve().parent
SHARED = ROOT.parent
OLD = SHARED / 'logistic_sequential_bound_revision_20261114/experiments'
sys.path.insert(0, str(OLD))
import validate_full as v
e, np = v.e, v.np
from whichlabels import prepare_comparison
from whichlabels._intervals import endpoints, mapping_guard

PILOT_OBJECTS = 32
TRIGGER_RATIO = .5
METHODS = ('zero_anchor', 'early_labels_zero_anchor', 'pilot_anchor', 'adaptive_pilot')


def save(path, value):
    v.save(path, value)


def substitute(prep, values, epsilon, anchors):
    """Exactly reduce the zero-anchor polynomial using only the paid cache."""
    p = prep.polynomial
    c = float(p['constant'])
    left, right, co = p['left'], p['right'], p['coef']
    x = 2*values-1
    lk, rk = np.isfinite(x[left]), np.isfinite(x[right])
    both = lk & rk
    c += float(np.sum(co[both]*x[left[both]]*x[right[both]]))
    linear = np.zeros(len(values))
    mask = lk & ~rk
    np.add.at(linear, right[mask], co[mask]*x[left[mask]])
    mask = rk & ~lk
    np.add.at(linear, left[mask], co[mask]*x[right[mask]])
    mask = ~lk & ~rk
    single = np.flatnonzero(linear != 0)
    cc = np.r_[co[mask], linear[single]]
    ll = np.r_[left[mask], single].astype(np.int32)
    rr = np.r_[right[mask], np.full(len(single), -1)].astype(np.int32)
    mass = float(abs(cc).sum())
    cdf = np.cumsum(abs(cc)/mass) if mass else np.array([])
    if mass:
        cdf[-1] = 1.
    return dict(constant=c, coef=cc, left=ll, right=rr, mass=mass,
                cdf=cdf, epsilon=epsilon, anchors=anchors)


def evaluate(poly, signed):
    prod = signed[poly['left']].copy()
    mask = poly['right'] >= 0
    prod[mask] *= signed[poly['right'][mask]]
    return float(poly['constant']+poly['coef']@prod)


def planning_key(poly, g):
    """A predeclared certificate/range proxy, not an observed performance score."""
    eps, mass, c = poly['epsilon'], poly['mass'], poly['constant']
    certified = e.decision(c-mass-eps-2e-9, c+mass+eps+2e-9, g) != 'unresolved'
    if certified:
        return (0, 0., eps)
    if eps < g:
        return (1, (mass/(g-eps))**2, eps)
    return (2, eps, mass)


def sample(a, b, plan, lam, g, poly, cache, seed, entry=True):
    """One frozen polynomial, cumulative CP counts, free local-bound updates."""
    c, mass, eps = poly['constant'], poly['mass'], poly['epsilon']
    initial = eps
    cdf, co = poly['cdf'], poly['coef']
    guard = 2e-9+(mapping_guard(co, cdf, mass) if mass else 0.)
    L, U = c-mass-eps-guard, c+mass+eps+guard
    choice = e.decision(L, U, g) if entry else 'unresolved'
    n = k = 0
    history = []
    rng = np.random.default_rng(seed)
    if mass and choice == 'unresolved':
        for end in e.CHECKS:
            ix = np.searchsorted(cdf, rng.random(end-n), side='right')
            xx = 2*cache.query_keys(poly['left'][ix], 'comparison')-1
            rr = poly['right'][ix]
            pair = rr >= 0
            xx[pair] *= 2*cache.query_keys(rr[pair], 'comparison')-1
            k += int(np.count_nonzero(np.sign(co[ix])*xx > 0))
            n = end
            pl, pu = endpoints(k, n, .05/len(e.CHECKS))
            lo, hi = c+mass*(2*pl-1)-guard, c+mass*(2*pu-1)+guard
            refined = False
            if e.decision(lo-eps, hi+eps, g) == 'unresolved' and e.decision(lo, hi, g) != 'unresolved':
                eps = min(eps, e.conditional_bound(a, b, plan, poly['anchors'], lam, cache._values))
                refined = True
            L, U = lo-eps, hi+eps
            choice = e.decision(L, U, g)
            history.append(dict(draws=n, labels=cache.count, epsilon=eps,
                                interval=[L,U], choice=choice, successes=k, refined=refined))
            if choice != 'unresolved':
                break
    return dict(choice=choice, interval=[L,U], draws=n, history=history,
                epsilon=eps, initial_epsilon=initial)


def baseline_row(j, rep, archive):
    main = {'attribution':'selective_paired', 'coverage':'selective_paired',
            'cache':'current', 'fivefold':'paired_moments',
            'simulation':'paired_weighted', 'boundary':'paired_weighted'}[j['family']]
    row = next(r for r in archive['records'] if r['method'] == main and r.get('rep',0) == rep)
    out = dict(row, method='zero_anchor', source_method=main, rep=rep,
               pilot_labels=0, selected_anchor='zero', source='matched published record')
    out['total_labels'] = row.get('total_labels', row.get('labels_total'))
    out['covered'] = row.get('covered', row.get('coverage'))
    out['acceptable'] = row.get('acceptable', row.get('cv_acceptable'))
    out['fallback'] = row.get('fallback', row.get('branch') == 'refit')
    return out


def worker(j):
    dest = ROOT/'fixed_results'/f'job_{j["id"]:04d}.json'
    if dest.exists():
        return j['id']
    with (ROOT/'logs'/f'fixed_{j["id"]:04d}.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            tick = time.perf_counter()
            a,b,y,plan,finals,seed,refs,methods = v.data(j)
            archive = json.loads((OLD/'results'/f'job_{j["id"]:04d}.json').read_text('utf-8'))
            target = archive['target']
            prep = prepare_comparison(a,b,plan,j['lam'])
            zero_anchors = [[np.zeros(a.shape[-1]), np.zeros(b.shape[-1])] for _ in plan]
            c0 = prep.polynomial['constant']; m0 = float(abs(prep.polynomial['coef']).sum())
            initial_choice = e.decision(c0-m0-prep.epsilon-2e-9, c0+m0+prep.epsilon+2e-9, j['g'])
            trigger = initial_choice == 'unresolved' and prep.epsilon >= TRIGGER_RATIO*j['g']
            rows = []; audits = []
            for rep, final in enumerate(finals):
                base = baseline_row(j,rep,archive)
                rows.append(base)
                # Already certified comparisons need no pilot under either policy.
                if initial_choice != 'unresolved' and j['family'] in ('coverage','simulation','boundary'):
                    for method in METHODS[1:]:
                        rows.append(dict(base,method=method,triggered=False,source='public-check identity'))
                    continue
                pilot = np.asarray(final[:min(PILOT_OBJECTS,len(final))],dtype=int)
                cache = e.LabelCache(len(a), y.shape[1], e.ArrayLabelProvider(y))
                yy = cache.query_objects(pilot,'pilot')
                assert cache.seen.reshape(len(a),-1)[pilot].all()
                assert np.isin(pilot,final).all()
                pair = [e.fit(z[pilot],yy,j['lam'])[0] for z in (a,b)]
                anchors = [pair for _ in plan]
                eps_zero = min(prep.epsilon, e.conditional_bound(a,b,plan,zero_anchors,j['lam'],cache._values))
                pzero = substitute(prep,cache._values,eps_zero,zero_anchors)
                pfit = e.prepare(a,b,plan,anchors,j['lam'],cache._values)
                use_fit = planning_key(pfit,j['g']) < planning_key(pzero,j['g'])
                # Both certificates apply to their own polynomial. Do not move an
                # error bound from one anchor to the other.
                diagnostic = dict(pilot_ids=pilot.tolist(),pilot_labels=cache.count,
                    initial_epsilon=prep.epsilon,zero_cached_epsilon=pzero['epsilon'],
                    nonzero_epsilon=pfit['epsilon'],zero_mass=pzero['mass'],nonzero_mass=pfit['mass'],
                    zero_proxy=planning_key(pzero,j['g']),nonzero_proxy=planning_key(pfit,j['g']),
                    anchor_norms=[float(np.linalg.norm(t)) for t in pair],selected_nonzero=use_fit)
                seen = cache.seen.copy(); values = cache._values.copy()
                pairrows = {}
                for name,poly in [('early_labels_zero_anchor',pzero),('pilot_anchor',pfit if use_fit else pzero)]:
                    if name == 'pilot_anchor' and not use_fit:
                        row = dict(pairrows['early_labels_zero_anchor'],method=name,selected_anchor='zero')
                        rows.append(row);pairrows[name]=row;continue
                    cc = e.LabelCache(len(a),y.shape[1],e.ArrayLabelProvider(y))
                    cc.query_objects(pilot,'pilot')
                    trial = sample(a,b,plan,j['lam'],j['g'],poly,cc,seed+rep)
                    row = e.summarize_record(trial,cc,a,b,plan,j['lam'],final,j['g'],
                        dict(method=name,rep=rep,seed=seed+rep,pilot_labels=int(seen.sum()),
                             selected_anchor='nonzero' if name=='pilot_anchor' else 'zero'),0.)
                    # Full responses are inspected only after the decision is final.
                    val = evaluate(poly,(2*y-1).ravel())
                    actual = abs(target-val)
                    assert actual <= trial['epsilon']+2e-8, ('certificate',j,rep,name,actual,trial['epsilon'])
                    assert all(actual <= h['epsilon']+2e-8 for h in trial['history'])
                    row.update(target=target,surrogate=val,actual_error=actual,
                        covered=bool(row['interval'][0]<=target<=row['interval'][1]),
                        acceptable=bool(target<=j['g'] if row['choice']=='A' else target>=-j['g']),
                        pilot_outside_final=0,distinct_union_verified=bool(cc.count==cc.seen.sum()),
                        pilot_ids_sha256=hashlib.sha256(pilot.tobytes()).hexdigest())
                    rows.append(row);pairrows[name]=row
                    print(j['family'],j['case'],rep,name,row['total_labels'],'zero',base['total_labels'],'anchor',row['selected_anchor'],flush=True)
                chosen = pairrows['pilot_anchor'] if trigger else base
                rows.append(dict(chosen,method='adaptive_pilot',triggered=trigger,
                                 source='pilot_anchor' if trigger else 'zero_anchor'))
                audits.append(dict(rep=rep,**diagnostic))
                del pzero,pfit,cache;gc.collect()
            save(dest,dict(job=j,pool_objects=len(a),attributes=y.shape[1],train_size=len(plan[0][0]),
                final_size=len(finals[0]),paths=len(finals),trigger=trigger,target=target,
                regime='both' if abs(target)<=j['g'] else 'A_only' if target<0 else 'B_only',
                epsilon0=prep.epsilon,records=rows,initialization=audits,seconds=time.perf_counter()-tick))
            return j['id']
        except Exception:
            traceback.print_exc();raise


def register():
    for sub in ('fixed_results','logs'):
        (ROOT/sub).mkdir(parents=True,exist_ok=True)
    protocol = dict(experiment='One initial pilot, with fixed anchor during comparison',
        pilot_objects=PILOT_OBJECTS,trigger_ratio=TRIGGER_RATIO,
        trigger='Public check unresolved and zero-anchor epsilon >= g/2',
        pilot_selection='First min(32,n) objects in the preassigned final-training order',
        methods=METHODS,anchor_selection='Choose zero/nonzero using certified public decision, then S1^2/(g-epsilon)^2 when feasible, otherwise smaller epsilon; ties retain zero',
        accounting='Distinct union of pilot, comparison, fallback, final training. Pilot is a subset of final training in Algorithm 1.',
        confidence='Conditional on initialization, one fixed polynomial and cumulative CP counts. Nine delta/9 inspections; uniform cache bounds may shrink after labels arrive.',
        replication='All existing fixed-plan designs and query paths; no data-dependent omissions',
        unchanged=['features','labels','plans','final subsets','lambda','g','checkpoints','query seeds','query cap'],
        fivefold='Published zero baseline retains its historical active-CV preparation gate. Pilot variants use the main fixed-CV exact-completion fallback; this difference is reported separately.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),jobs=v.configs())
    if not (ROOT/'fixed_protocol.json').exists():
        save(ROOT/'fixed_protocol.json',protocol)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=3)
    ap.add_argument('--jobs',nargs='*',type=int);ap.add_argument('--max-pool',type=int)
    ap.add_argument('--family');args=ap.parse_args();register()
    jobs=[j for j in v.configs() if (args.jobs is None or j['id'] in args.jobs)
          and (args.max_pool is None or j['pool']<args.max_pool)
          and (args.family is None or j['family']==args.family)
          and not (ROOT/'fixed_results'/f'job_{j["id"]:04d}.json').exists()]
    from concurrent.futures import ProcessPoolExecutor,as_completed
    began=time.perf_counter();errors=[]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        fs={ex.submit(worker,j):j for j in jobs}
        for fu in as_completed(fs):
            try: i=fu.result(); status='done'
            except Exception as exc:
                i=fs[fu]['id'];status=repr(exc);errors.append(dict(job=i,error=status))
            done=len(list((ROOT/'fixed_results').glob('job_*.json')))
            save(ROOT/'fixed_progress.json',dict(done=done,total=1872,last=i,status=status,
                errors=errors,elapsed=time.perf_counter()-began))
            print(done,'/1872',i,status,flush=True)
    if errors:raise RuntimeError(errors)


if __name__=='__main__':main()
