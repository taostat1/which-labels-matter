"""Selective label comparison for shared logistic regression."""
from .api import prepare_comparison, compare, audit_score
from .labels import LabelCache, ArrayLabelProvider

__version__ = '0.2.1'
__all__ = ['prepare_comparison', 'compare', 'audit_score', 'LabelCache', 'ArrayLabelProvider']
