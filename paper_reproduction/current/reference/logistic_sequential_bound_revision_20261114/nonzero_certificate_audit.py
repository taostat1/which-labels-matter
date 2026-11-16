"""Exhaustive check of the nonzero-anchor cache-completion certificate."""
from pathlib import Path
import os, sys, itertools, json
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'logistic_local_anchor_refinement_20261114'))
from local_core import candidate_geometry, certificate
from anchor_core import fit, risk, sigmoid_weight
import numpy as np
from scipy.special import expit

rng = np.random.default_rng(202611142117)
a = rng.normal(size=(3, 2, 3))
a /= np.maximum(1, np.linalg.norm(a, axis=-1, keepdims=True))
b = .6 * a + .4 * rng.normal(size=a.shape)
b /= np.maximum(1, np.linalg.norm(b, axis=-1, keepdims=True))
ys = np.array(list(itertools.product([0., 1.], repeat=6)))
xs = 2 * ys - 1
patterns = [[], [0, 4], [0, 1, 4], list(range(6))]
fixed_anchors = rng.normal(size=(2, 3))
details = []
for lam in [.125, .5, 2.]:
    exact = []
    parameters = []
    for y in ys:
        yy = y.reshape(3, 2)
        fits = [fit(z[:2], yy[:2], lam)[0] for z in (a, b)]
        parameters.append(fits)
        exact.append(risk(a[2:], yy[2:], fits[0]) - risk(b[2:], yy[2:], fits[1]))
    exact = np.array(exact)
    parameters = np.array(parameters)
    for anchor_mode in ('fixed_nonzero', 'cache_midpoint'):
        max_excess = -float('inf')
        max_difference_excess = -float('inf')
        completion_checks = 0
        states = 0
        for pat in patterns:
            for values in itertools.product([-1., 1.], repeat=len(pat)):
                known = np.full(6, np.nan)
                known[pat] = values
                midpoint = np.where(np.isfinite(known[:4]), (known[:4] + 1) / 2, .5)
                anchors = fixed_anchors if anchor_mode == 'fixed_nonzero' else np.array([
                    fit(z[:2], midpoint, lam)[0] for z in (a, b)])
                ga, gb = [candidate_geometry(z[:2], z[2:], t, lam, known[:4])
                          for z, t in zip((a, b), anchors)]
                old, new, aa, bb = certificate(ga, gb, lam, known[4:])
                mask = np.ones(64, dtype=bool) if not pat else np.all(xs[:, pat] == known[pat], axis=1)
                for ix in np.flatnonzero(mask):
                    approx = []
                    errors = []
                    for z, g, anchor, fitted in zip((a, b), (ga, gb), anchors, parameters[ix]):
                        step = g['c'] + g['L'] @ xs[ix, :4]
                        v = z[2:].reshape(2, 3) @ anchor
                        u = z[2:].reshape(2, 3) @ step
                        approx.append(np.mean(np.logaddexp(0, v) - ys[ix, 4:] * v
                                      + (expit(v) - ys[ix, 4:]) * u + .5 * sigmoid_weight(v) * u * u))
                        errors.append(fitted - anchor - step)
                    risk_error = abs(exact[ix] - (approx[0] - approx[1]))
                    max_excess = max(max_excess, risk_error - new['epsilon'])
                    difference_error = np.linalg.norm(errors[0] - errors[1])
                    max_difference_excess = max(max_difference_excess,
                                               difference_error - new['parameter_error_difference'])
                    assert risk_error <= new['epsilon'] + 1e-9
                    assert risk_error <= new['separate'] + 1e-9
                    assert difference_error <= new['parameter_error_difference'] + 1e-9
                    completion_checks += 1
                states += 1
        details.append(dict(lambda_value=lam, anchor_mode=anchor_mode, cache_states=states,
                            completion_checks=completion_checks, max_risk_bound_excess=max_excess,
                            max_parameter_difference_excess=max_difference_excess))
out = dict(passed=True, details=details,
           cache_states=sum(x['cache_states'] for x in details),
           completion_checks=sum(x['completion_checks'] for x in details),
           interpretation='Exhaustive implementation checks; the proof is in cache_bound.tex.')
(ROOT / 'nonzero_certificate_audit.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
print(json.dumps(out, indent=2))
