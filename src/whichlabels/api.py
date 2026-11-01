"""Public API for Algorithm 1 and its same-permission active-CV baseline."""
from dataclasses import dataclass
import time
import numpy as np
from ._bounds import paired_preconditioned
from ._moments import polynomial, query_plan, query_trial, decision
from ._active import public_bounds, zero_helper, active_cv
from .solver import fit, risk
from .labels import LabelCache, ArrayLabelProvider

NUMERICAL_GUARD = 2e-9


@dataclass
class PreparedComparison:
    a: np.ndarray
    b: np.ndarray
    splits: tuple
    penalty: float
    polynomial: dict
    proposal: dict | None
    bounds: dict
    epsilon: float
    separate_epsilon: float
    setup_seconds: float
    baseline_setup_seconds: float


def _ids(values, size, name):
    arr = np.asarray(values)
    if arr.ndim != 1 or not np.issubdtype(arr.dtype, np.integer) or not arr.size:
        raise ValueError(f'{name} must be a nonempty integer vector')
    if arr.min() < 0 or arr.max() >= size or len(np.unique(arr)) != len(arr):
        raise ValueError(f'{name} must contain distinct valid object indices')
    arr = arr.astype(np.int64, copy=True)
    arr.setflags(write=False)
    return arr


def prepare_comparison(a, b, splits, penalty):
    """Compute coefficients and error bounds using public features only.

    a and b have shape (objects, attributes, shared_coefficients). Every row
    has norm at most one. Both candidates use the same coefficient coordinates.
    Each split is a pair (training_indices, evaluation_indices). Splits and
    final training indices must be fixed independently of acquisition labels.
    """
    start = time.perf_counter()
    a, b = np.array(a, dtype=float, copy=True), np.array(b, dtype=float, copy=True)
    if a.ndim != 3 or a.shape != b.shape or min(a.shape) < 1:
        raise ValueError('both designs must have the same nonempty three-dimensional shape')
    if not np.isfinite(penalty) or penalty <= 0:
        raise ValueError('penalty must be positive')
    for z in (a, b):
        if not np.isfinite(z).all() or np.max(np.linalg.norm(z, axis=-1)) > 1 + 1e-10:
            raise ValueError('design rows must be finite and have norm at most one')
        z.setflags(write=False)
    plan = []
    for tr, te in splits:
        tr, te = _ids(tr, len(a), 'train'), _ids(te, len(a), 'evaluation')
        if np.intersect1d(tr, te).size:
            raise ValueError('training and evaluation objects must be disjoint within each split')
        plan.append((tr, te))
    if not plan or len({len(tr) for tr, _ in plan}) != 1:
        raise ValueError('a nonempty plan with a common training size is required')
    plan = tuple(plan)
    bounds = public_bounds(a, b, plan, penalty)
    pairs = [paired_preconditioned(a[tr], b[tr], a[te], b[te], penalty) for tr, te in plan]
    epsilon = min(float(np.mean([v['epsilon'] for v in pairs])), bounds['epsilon'])
    poly = polynomial(a, b, plan, penalty)
    poly['epsilon'] = epsilon
    mass = float(np.abs(poly['coef']).sum())
    proposal = query_plan(poly, zero_helper(len(a), a.shape[1]), a.shape[1]) if mass else None
    return PreparedComparison(a, b, plan, float(penalty), poly, proposal, bounds,
        epsilon, bounds['epsilon'], time.perf_counter() - start, bounds['seconds'])


def _exact_score(prepared, cache, stage):
    """Complete the finite plan with a gradient-based numerical error bound."""
    losses, errors = [], []
    for tr, te in prepared.splits:
        yt = cache.query_objects(tr, stage)
        ye = cache.query_objects(te, stage)
        values, radii = [], []
        for z in (prepared.a, prepared.b):
            theta, info = fit(z[tr], yt, prepared.penalty)
            values.append(risk(z[te], ye, theta))
            radii.append(float(np.linalg.norm(z[te], axis=-1).mean()) * info['parameter_error_bound'])
        losses.append(values[0] - values[1])
        errors.append(sum(radii))
    return float(np.mean(losses)), float(np.mean(errors)) + NUMERICAL_GUARD


def audit_score(prepared, labels):
    """Separate all-label audit; never used by the acquisition rule.

    This is an evaluation tool for simulations/replay. Do not supply its result
    to prepare_comparison or compare, or use it to tune a query path.
    """
    cache = LabelCache(len(prepared.a), prepared.a.shape[1], ArrayLabelProvider(labels))
    value, error = _exact_score(prepared, cache, 'audit')
    return {'difference': value, 'numerical_radius': error, 'audit_labels': cache.count}


def compare(prepared, cache, final_indices, *, tolerance, delta=0.05,
            inspections=(128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768),
            seed=1, method='selective', error_bound='paired'):
    """Run a fixed-CV decision and fit the selected candidate once at the end.

    method is 'selective' (Algorithm 1) or 'active_cv' (paper comparator).
    error_bound='separate' is the matched approximation-bound ablation.
    Returned label counts are the cache union, including final training.
    The finite-plan guarantee is conditional on the pool, labels and designs.
    """
    start = time.perf_counter()
    if cache.objects != len(prepared.a) or cache.r != prepared.a.shape[1]:
        raise ValueError('cache dimensions do not match the public designs')
    if cache.count:
        raise ValueError('use a fresh cache for each independent method comparison')
    if not np.isfinite(tolerance) or tolerance <= 1e-8 or not 0 < delta < 1:
        raise ValueError('tolerance must exceed 1e-8 and delta must lie in (0, 1)')
    checks = np.asarray(inspections)
    if (checks.ndim != 1 or not checks.size or not np.issubdtype(checks.dtype, np.integer)
            or checks[0] < 2 or (np.diff(checks) <= 0).any()):
        raise ValueError('inspections must be strictly increasing integers >= 2')
    final_indices = _ids(final_indices, len(prepared.a), 'final_indices')
    if len(final_indices) != len(prepared.splits[0][0]):
        raise ValueError('the final subset must have the prescribed training size')
    if method not in {'selective', 'active_cv'} or error_bound not in {'paired', 'separate'}:
        raise ValueError('unknown method or approximation bound')
    poly = dict(prepared.polynomial)
    eps = prepared.epsilon if error_bound == 'paired' else prepared.separate_epsilon
    poly['epsilon'] = eps
    fallback = False
    if method == 'active_cv':
        cfg = dict(delta=delta, active_initial_draws=32, active_max_draws_per_fold=4096,
            inner_lambdas=[.005, .01, .02, .05, .1, .2, .4, .6, .8, .95],
            outer_betting_fractions=[.05, .1, .2, .4, .6, .8, .95])
        trial = active_cv(prepared.a, prepared.b, cache, prepared.splits,
            prepared.penalty, tolerance, prepared.bounds, cfg, seed, seed + 1000000)
        branch = 'active_cv'
        fallback = trial['empty_protection']
        setup = prepared.baseline_setup_seconds
    else:
        mass = float(np.abs(poly['coef']).sum())
        center = float(poly['constant'])
        half = mass + eps + NUMERICAL_GUARD
        choice = decision(center - half, center + half, tolerance)
        trial = dict(choice=choice, lower=center-half, upper=center+half,
            estimate=center, draws=0, history=[])
        branch = 'public_range'
        if choice == 'unresolved' and mass:
            trial = query_trial(poly, prepared.proposal, cache, np.random.default_rng(seed),
                list(map(int, checks)), tolerance, delta)
            branch = 'selective_moments'
        if trial['choice'] == 'unresolved':
            value, radius = _exact_score(prepared, cache, 'exact_cv_fallback')
            trial.update(choice=decision(value-radius, value+radius, tolerance),
                         lower=value-radius, upper=value+radius)
            branch, fallback = 'exact_cv_fallback', True
        setup = prepared.setup_seconds
    if trial['choice'] == 'unresolved':
        raise ArithmeticError('numerical precision does not support the requested tolerance')
    before_final = cache.count
    labels = cache.query_objects(final_indices, 'final_training')
    features = prepared.a if trial['choice'] == 'A' else prepared.b
    theta, fit_info = fit(features[final_indices], labels, prepared.penalty)
    elapsed = time.perf_counter() - start
    assert cache.count == sum(cache.stage_counts.values())
    return dict(method=method, choice=trial['choice'], interval=[trial['lower'], trial['upper']],
        tolerance=float(tolerance), delta=float(delta), approximation_error_bound=float(eps),
        branch=branch, fallback=bool(fallback), comparison_labels=before_final,
        new_final_training_labels=cache.count-before_final, total_labels=cache.count,
        stage_labels=dict(cache.stage_counts), requests=dict(cache.requests),
        draws=int(trial.get('draws', trial.get('evaluation_draws', 0))),
        history=trial['history'], final_parameter=theta.tolist(), final_fit=fit_info,
        setup_seconds=setup, decision_and_final_fit_seconds=elapsed,
        total_seconds=setup+elapsed)

