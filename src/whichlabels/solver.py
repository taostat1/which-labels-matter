"""The archived Newton solver with an explicit residual check."""
import numpy as np
from ._fit import fit as _newton_fit, risk


def fit(z, y, lam, start=None):
    # Strong convexity gives ||theta - theta*|| <= ||gradient|| / lambda.
    # Normalized evaluation rows make this also a bound on the BCE error.
    tol = min(1e-10, 2e-11 * lam)
    theta, info = _newton_fit(z, y, lam, tol=tol, start=start)
    distance = info['residual'] / lam
    if not np.isfinite(distance) or distance > 5e-10:
        raise ArithmeticError('Newton accuracy is insufficient for the numerical interval')
    info['parameter_error_bound'] = distance
    return theta, info

