"""
This type stub file was generated by pyright.
"""

from ._svds import svds
from .arpack import *
from .lobpcg import *

"""
Sparse Eigenvalue Solvers
-------------------------

The submodules of sparse.linalg._eigen:
    1. lobpcg: Locally Optimal Block Preconditioned Conjugate Gradient Method

"""
__all__ = ['ArpackError', 'ArpackNoConvergence', 'eigs', 'eigsh', 'lobpcg', 'svds']
test = ...
