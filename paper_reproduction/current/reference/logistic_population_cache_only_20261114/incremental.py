"""Deterministic finite-pool score envelopes; no label oracle is accepted."""
import numpy as np
from scipy.special import expit


class EnvelopeTemplate:
    def __init__(self, z, theta, penalty):
        self.z = np.asarray(z)
        self.n, self.r, self.d = z.shape
        assert self.r <= 12
        self.signs = 2*((np.arange(2**self.r)[:, None] >> np.arange(self.r)) & 1)-1
        self.c = np.einsum('ird,ir->id', z, expit(z @ theta)-.5)/self.r + penalty*theta
        self.maximum = np.empty(self.n)
        for start in range(0, self.n, 256):
            ids = np.arange(start, min(start+256, self.n))
            self.maximum[ids] = self.squares(ids).max(1)
        self.center = self.c.mean(0)
        self.radius = abs(z).sum((0, 1))/(2*self.n*self.r)
        self.gram = np.einsum('ird,ire->de', z, z, optimize=True)/(self.n*self.r)

    def squares(self, ids):
        score = self.c[ids, None] - np.einsum('hr,ird->ihd', self.signs, self.z[ids], optimize=True)/(2*self.r)
        return np.einsum('ihd,ihd->ih', score, score)


class EnvelopeState:
    def __init__(self, template):
        self.t = template
        self.observed = np.full((template.n, template.r), np.nan)
        self.maximum = template.maximum.copy()
        self.center = template.center.copy()
        self.radius = template.radius.copy()
        self.gram = template.gram.copy()
        self.unknown = template.n*template.r

    def update(self, observed):
        t = self.t
        old_known = np.isfinite(self.observed)
        assert np.all(observed[old_known] == self.observed[old_known])
        new = np.isfinite(observed) & ~old_known
        i, j = np.where(new)
        if len(i):
            z = t.z[i, j]
            x = 2*observed[i, j]-1
            self.center -= (z*x[:, None]).sum(0)/(2*t.n*t.r)
            self.radius -= abs(z).sum(0)/(2*t.n*t.r)
            self.gram -= z.T @ z/(t.n*t.r)
            self.unknown -= len(i)
            dirty = np.unique(i)
            for start in range(0, len(dirty), 128):
                ids = dirty[start:start+128]
                yy = observed[ids]
                known = np.isfinite(yy)
                xx = np.where(known, 2*yy-1, 0.)
                compatible = np.all((t.signs[None] == xx[:, None]) | ~known[:, None], axis=2)
                sq = t.squares(ids)
                sq[~compatible] = -np.inf
                self.maximum[ids] = sq.max(1)
            self.observed[new] = observed[new]
        spectral = max(0., float(np.linalg.eigvalsh(self.gram)[-1]))
        # Same upward guard as the full-recomputation reference implementation.
        guard = 1e-11*(1+float(self.maximum.max()))
        radius = np.maximum(self.radius, 0.)
        norm_bound = min(np.linalg.norm(abs(self.center)+radius),
                         np.linalg.norm(self.center)+.5*np.sqrt(self.unknown/(t.n*t.r)*spectral))
        second = float(self.maximum.mean())
        return dict(mu2=min(second, float(norm_bound**2))+guard,
                    second=second+guard, B=float(np.sqrt(self.maximum.max()+guard)),
                    objects=t.n, known_labels=t.n*t.r-self.unknown, total_labels=t.n*t.r,
                    certificate='all compatible finite-pool label completions')


class BoundTemplate:
    def __init__(self, a, b, pools, anchors, case):
        self.pools, self.case, self.zs = pools, case, (a,b)
        self.templates = [[EnvelopeTemplate(z[pools[name]], theta, case['lam'] if name=='training' else 0.)
                           for name in ('training','comparison')] for z,theta in zip((a,b),anchors)]

    def state(self):
        return [[EnvelopeState(t) for t in pair] for pair in self.templates]

    def update(self, states, values, par, study, from_moments):
        observed = values.reshape(-1, self.zs[0].shape[1])
        details = []
        for j,z in enumerate(self.zs):
            train,test = [state.update(observed[self.pools[name]])
                          for state,name in zip(states[j],('training','comparison'))]
            details.append(from_moments(train,test,z,self.case,par,j,study))
        return float(sum(row['used'] for row in details)+1e-10), details
