"""A self-contained example with correlated binary attributes."""
from pathlib import Path
import argparse
import csv
import json
import time
import numpy as np
from scipy.special import expit, ndtr
from . import prepare_comparison, compare, audit_score, LabelCache, ArrayLabelProvider


def synthetic_problem(seed=202611010901, objects=1024, splits=8, train_size=64,
                      test_size=64, alpha=0.1, signal=4.0):
    if train_size + test_size > objects:
        raise ValueError('pool is too small for disjoint train and evaluation subsets')
    rng = np.random.default_rng(seed)
    u = (2*rng.integers(0, 2, (objects, 4))-1)/2
    v = (2*rng.integers(0, 2, (objects, 4))-1)/2
    h = np.array([[1,1,1,1], [1,-1,1,-1], [1,1,-1,-1], [1,-1,-1,1]])
    eye = np.broadcast_to(np.eye(4), (objects,4,4))
    a = np.concatenate([u[:,None,:]*h[None,:,:], eye], axis=2)/np.sqrt(2)
    b = np.concatenate([((1-alpha)*u+alpha*v)[:,None,:]*h[None,:,:], eye], axis=2)/np.sqrt(2)
    uniforms = ndtr(np.sqrt(.3)*rng.normal(size=(objects,1)) + np.sqrt(.7)*rng.normal(size=(objects,4)))
    orders = [rng.permutation(objects) for _ in range(splits)]
    plan = [(o[:train_size], o[train_size:train_size+test_size]) for o in orders]
    final = rng.choice(objects, train_size, replace=False)
    theta = signal*np.array([1,.5,-.5,.25,0,0,0,0])
    labels = (uniforms <= expit(np.einsum('ird,d->ir', a, theta))).astype(float)
    return a, b, labels, plan, final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('demo_output'))
    parser.add_argument('--seed', type=int, default=202611010901)
    parser.add_argument('--objects', type=int, default=1024)
    parser.add_argument('--splits', type=int, default=8)
    parser.add_argument('--penalty', type=float, default=.125)
    parser.add_argument('--tolerance', type=float, default=.01)
    args = parser.parse_args()
    if (args.output / 'results.json').exists():
        parser.error('output contains a completed run; select a fresh --output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    def serializable(x):
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, np.generic):
            return x.item()
        return x.tolist()
    def save(name, obj):
        (args.output / name).write_text(json.dumps(obj, indent=2, default=serializable), encoding='utf-8')
    def progress(stage):
        save('progress.json', dict(stage=stage, updated=time.strftime('%Y-%m-%d %H:%M:%S')))
        print(stage, flush=True)
    progress('Preparing public coefficients and error bounds')
    a,b,y,plan,final = synthetic_problem(args.seed, args.objects, args.splits)
    prepared = prepare_comparison(a,b,plan,args.penalty)
    rows=[]
    for method in ['selective','active_cv']:
        progress('Running '+method)
        cache=LabelCache(len(a),a.shape[1],ArrayLabelProvider(y))
        row=compare(prepared,cache,final,tolerance=args.tolerance,seed=args.seed+1,method=method)
        rows.append(row)
        print(f"{method}: {row['choice']}, {row['total_labels']} distinct labels, {row['branch']}",flush=True)
    # The audit happens only after both decisions, and has a separate provider/cache.
    progress('Auditing completed decisions with all labels')
    audit=audit_score(prepared,y)
    for row in rows:
        d=audit['difference'];lo,hi=row['interval']
        row['audit_covered']=bool(lo<=d<=hi)
        row['audit_acceptable']=bool(d<=args.tolerance if row['choice']=='A' else d>=-args.tolerance)
    save('results.json',dict(configuration=vars(args),audit=audit,methods=rows,
        note='One synthetic example, not a new manuscript experiment.',
        label_saving_percent=100*(1-rows[0]['total_labels']/rows[1]['total_labels'])))
    columns=['method','choice','branch','total_labels','comparison_labels','new_final_training_labels','total_seconds','audit_covered','audit_acceptable']
    with (args.output/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    progress('Complete')


if __name__=='__main__':
    main()
