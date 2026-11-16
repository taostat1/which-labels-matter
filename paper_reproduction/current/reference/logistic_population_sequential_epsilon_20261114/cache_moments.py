"""Uniform population score bounds for a finite empirical feature/label law.

Only acquired values (NaN means unobserved) are accepted. No label oracle is
passed to this module. The anchor stays fixed. Bounds hold for every completion
of the observed finite label array and hence for adaptive query histories.
"""
import numpy as np
from scipy.special import expit


def score_bounds(z, theta, penalty, observed):
    n, r, d = z.shape
    assert observed.shape == (n, r)
    assert r <= 12
    signs = 2 * ((np.arange(2**r)[:, None] >> np.arange(r)) & 1) - 1
    max_sq = np.empty(n)
    center_sum = np.zeros(d)
    radius_sum = np.zeros(d)
    unknown_gram = np.zeros((d, d))
    unknown_count = 0
    for start in range(0, n, 128):
        zz = z[start:start+128]
        yy = observed[start:start+128]
        known = np.isfinite(yy)
        xx = np.where(known, 2 * yy - 1, 0.)
        pp = expit(np.einsum('ird,d->ir', zz, theta))
        c = np.einsum('ird,ir->id', zz, pp - .5) / r + penalty * theta
        center_sum += (c - np.einsum('ird,ir->id', zz, xx)/(2*r)).sum(0)
        unknown = ~known
        radius_sum += (abs(zz)*unknown[..., None]).sum((0, 1))/(2*r)
        unknown_gram += np.einsum('ird,ir,ire->de', zz, unknown, zz, optimize=True)/r
        unknown_count += int(unknown.sum())
        # Exact maximum of the squared score over compatible attribute vectors.
        scores = c[:, None, :] - np.einsum('hr,ird->ihd', signs, zz, optimize=True)/(2*r)
        sq = np.einsum('ihd,ihd->ih', scores, scores)
        compatible = np.all((signs[None] == xx[:, None]) | ~known[:, None], axis=2)
        sq[~compatible] = -np.inf
        max_sq[start:start+len(zz)] = sq.max(1)
    center = center_sum/n
    radius = radius_sum/n
    gram = unknown_gram/n
    fraction = unknown_count/(n*r)
    spectral = max(0., float(np.linalg.eigvalsh(gram)[-1]))
    second = float(max_sq.mean())
    guard = 1e-11 * (1 + float(max_sq.max()))
    norm_bound = min(np.linalg.norm(abs(center)+radius),
                     np.linalg.norm(center)+.5*np.sqrt(fraction*spectral))
    return dict(mu2=min(second, float(norm_bound**2))+guard,
                second=second+guard, B=float(np.sqrt(max_sq.max()+guard)),
                objects=n, known_labels=int(np.isfinite(observed).sum()),
                total_labels=n*r, certificate='all compatible finite-pool label completions')


def from_moments(train, test, z, case, par, candidate, study):
    local = min((study.localized_allowance(train, test, case['n'], case['lam'],
        par['pars'][candidate], par['geo']['Kq'][candidate], par['jq'][candidate],
        z.shape[-1], eta) for eta in (1e-4,1e-5,1e-6,1e-7,1e-8)),
        key=lambda row:row['allowance'])
    glob = study.approx_bound(train, case['n'], case['lam'])
    return dict(training=train, test=test, local=local, global_bound=glob,
                used=min(glob, local['allowance']))


def cache_allowance(a, b, pools, anchors, case, par, values, study):
    observed = values.reshape(-1, a.shape[1])
    details = []
    for j, (z, theta) in enumerate(zip((a, b), anchors)):
        train = score_bounds(z[pools['training']], theta, case['lam'], observed[pools['training']])
        test = score_bounds(z[pools['comparison']], theta, 0., observed[pools['comparison']])
        details.append(from_moments(train, test, z, case, par, j, study))
    return float(sum(row['used'] for row in details)+1e-10), details
