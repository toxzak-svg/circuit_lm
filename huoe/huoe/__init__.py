"""Hierarchical Union-of-Experts (H-UoE).

Stateful, sparse Mixture-of-Experts transformer with:
- Hierarchical routing (group → experts) conditioned on GRU controller state.
- SET-style evolutionary sparse weights inside experts.
- Shared expert path per macro-block.
"""

from huoe.controller import GRUController
from huoe.sparse_linear import SparseLinear, rewire_sparse_layer
from huoe.experts import GroupExperts
from huoe.router import HierarchicalGroupRouter
from huoe.macro_block import MacroBlock
from huoe.model import HUoEModel

__all__ = [
    "GRUController",
    "SparseLinear",
    "rewire_sparse_layer",
    "GroupExperts",
    "HierarchicalGroupRouter",
    "MacroBlock",
    "HUoEModel",
]
