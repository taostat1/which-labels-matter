"""Decision for the population risk gap from the cumulative mean interval."""


def assess_interval(mean_interval, epsilon, tolerance):
    """Certify a direction only when the raw intersection is nonempty.

    A pair of null endpoints records an uninformative risk interval in JSON.
    """
    lower, upper = mean_interval
    if lower > upper:
        return 'unresolved', [None, None]

    interval = [lower - epsilon, upper + epsilon]
    if interval[1] <= tolerance:
        return 'A', interval
    if interval[0] >= -tolerance:
        return 'B', interval
    return 'unresolved', interval
