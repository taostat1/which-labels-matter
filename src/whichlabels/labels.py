"""Acquire each object-attribute label at most once."""
from collections.abc import Callable
import numpy as np


class ArrayLabelProvider:
    """A replay provider for examples and audits; not an acquisition policy."""

    def __init__(self, labels):
        labels = np.asarray(labels, dtype=float)
        if labels.ndim != 2 or not np.isin(labels, [0, 1]).all():
            raise ValueError('labels must be a two-dimensional binary array')
        self._labels = labels.copy()

    def __call__(self, objects, attributes):
        return self._labels[objects, attributes].copy()


class LabelCache:
    """A paid-label cache around provider(object_ids, attribute_ids).

    The provider returns one binary label per requested pair. It is called only
    for distinct, previously unseen pairs. The policy cannot read unbought
    labels from this class. Its accounting includes the final training subset.
    """

    def __init__(self, objects: int, attributes: int, provider: Callable):
        if objects < 1 or attributes < 1 or not callable(provider):
            raise ValueError('positive dimensions and a callable provider are required')
        self.objects, self.r = int(objects), int(attributes)
        self.provider = provider
        self.seen = np.zeros(self.objects * self.r, dtype=bool)
        self._values = np.full(self.objects * self.r, np.nan)
        self.stage_counts = {}
        self.requests = {}

    def query_keys(self, keys, stage='comparison'):
        keys = np.asarray(keys)
        if not np.issubdtype(keys.dtype, np.integer):
            raise ValueError('label keys must be integers')
        if keys.size and (keys.min() < 0 or keys.max() >= self.seen.size):
            raise IndexError('label key outside the data pool')
        unique = np.unique(keys)
        new = unique[~self.seen[unique]]
        if new.size:
            values = np.asarray(self.provider(new // self.r, new % self.r), dtype=float)
            if values.shape != new.shape or not np.isin(values, [0, 1]).all():
                raise ValueError('provider must return one binary value per requested pair')
            self._values[new] = values
            self.seen[new] = True
        self.stage_counts[stage] = self.stage_counts.get(stage, 0) + int(new.size)
        self.requests[stage] = self.requests.get(stage, 0) + int(keys.size)
        return self._values[keys].copy()

    def query_objects(self, ids, stage='comparison'):
        ids = np.asarray(ids)
        return self.query_keys(ids[..., None] * self.r + np.arange(self.r), stage)

    @property
    def count(self):
        return int(self.seen.sum())

