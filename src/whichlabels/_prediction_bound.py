"""A prediction-space bound for two ridge-logistic training rules, in native widths."""
import numpy as np
from scipy.special import expit


def residual(t):
    # f(t) = t/4 - tanh(t/2)/2 for t >= 0. The cubic is an upper bound.
    return np.where(t < .001, t**3 / 48, np.maximum(t / 4 - .5 * np.tanh(t / 2), 0)) + 2e-15


def affine(K, signs):
    """Maximize each absolute row of K @ signs / 2 over all missing signs."""
    known = np.isfinite(signs)
    if not known.any():
        return np.abs(K).sum(axis=1) / 2
    fixed = np.abs(K[:, known] @ signs[known])
    free = np.abs(K[:, ~known]).sum(axis=1)
    return (fixed + free) / 2


def design(z, q, lam, signs):
    """Bound one rule's training and evaluation logits on the split."""
    m = len(z)
    G = z.T @ z / m
    H = lam * np.eye(z.shape[1]) + G / 4
    L = np.linalg.solve(H, z.T) / m
    A, V = z @ L, q @ L
    a, v = np.abs(A), np.abs(V)
    step, stepq = affine(A, signs), affine(V, signs)
    # Strong convexity at the one-step fit. The available labels give a box
    # for the training logits and hence for the score residual.
    radius = np.sqrt(np.linalg.eigvalsh(G)[-1]) * np.sqrt(np.mean(residual(step)**2)) / lam
    e = np.linalg.norm(z, axis=1) * radius
    eq = np.linalg.norm(q, axis=1) * radius
    for _ in range(6):
        e = np.minimum(e, a @ residual(step + e)) + 1e-14
    eq = np.minimum(eq, v @ residual(step + e)) + 1e-14
    return dict(A=A, V=V, a=a, v=v, step=step, stepq=stepq, e=e, eq=eq)


def bound(za, zb, qa, qb, lam, signs=None):
    """Return the split certificate for arrays with matched object-attribute rows.

    The last axis indexes features and may have a different size in each rule.
    Signs are the known training labels, coded -1 or +1, and NaN for unknowns.
    """
    za, zb, qa, qb = (x.reshape(-1, x.shape[-1]) for x in (za, zb, qa, qb))
    signs = np.full(len(za), np.nan) if signs is None else np.asarray(signs)
    A = design(za, qa, lam, signs)
    B = design(zb, qb, lam, signs)
    difference = A["A"] - B["A"]
    difference_q = A["V"] - B["V"]
    h, hq = affine(difference, signs), affine(difference_q, signs)
    ab, abq = np.abs(difference), np.abs(difference_q)
    ta, tb = A["step"] + A["e"], B["step"] + B["e"]
    fa, fb = residual(ta), residual(tb)

    # Propagate the difference between the two rules on shared prediction rows.
    slope = .25 * np.tanh(np.maximum(ta, tb) / 2)**2
    delta = A["e"] + B["e"]
    part_a, part_b = ab @ fa, ab @ fb
    for _ in range(6):
        diff_f = slope * (h + delta)
        paired = np.minimum(part_a + B["a"] @ diff_f, part_b + A["a"] @ diff_f)
        delta = np.minimum(delta, paired) + 1e-14
    diff_f = slope * (h + delta)
    paired_q = np.minimum(abq @ fa + B["v"] @ diff_f, abq @ fb + A["v"] @ diff_f)
    dq = np.minimum(A["eq"] + B["eq"], paired_q) + 1e-14
    cross = hq / 4 + dq / 8
    first = expit(A["stepq"] + A["eq"]) * dq + B["eq"] * cross
    second = expit(B["stepq"] + B["eq"]) * dq + A["eq"] * cross
    inc = np.minimum(first, second)
    # The two zero-anchor Taylor remainders have the same sign. Each
    # absolute remainder is bounded by t^4/192 and is 0 at logit 0.
    max_step = np.maximum(A["stepq"], B["stepq"])
    rem = np.minimum(residual(max_step) * hq, max_step**4 / 192)
    return float(np.mean(inc + rem)) * (1 + 2e-10) + 2e-12


def split_allowances(a, b, splits, penalty, cached_labels=None):
    """Compute one B_b(C) per split from the published prediction-space bound.

    The cache is in global object/attribute order. Only its known training
    labels enter each split bound. NaNs stay unknown when converted to signs.
    """
    signs = None
    if cached_labels is not None:
        labels = np.asarray(cached_labels).reshape(a.shape[:2])
        signs = 2 * labels - 1

    allowances = []
    for train, evaluation in splits:
        za, zb = a[train], b[train]
        qa, qb = a[evaluation], b[evaluation]
        if np.array_equal(za, zb) and np.array_equal(qa, qb):
            # Identical rules have zero exact and approximate loss gaps.
            allowances.append(0.0)
        else:
            known = None if signs is None else signs[train].ravel()
            allowances.append(bound(za, zb, qa, qb, penalty, known))
    return np.asarray(allowances)
