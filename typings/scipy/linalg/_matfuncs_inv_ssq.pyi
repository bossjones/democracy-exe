"""
This type stub file was generated by pyright.
"""

import numpy as np
from scipy.sparse.linalg._interface import LinearOperator

"""
Matrix functions that use Pade approximation with inverse scaling and squaring.

"""
class LogmRankWarning(UserWarning):
    ...


class LogmExactlySingularWarning(LogmRankWarning):
    ...


class LogmNearlySingularWarning(LogmRankWarning):
    ...


class LogmError(np.linalg.LinAlgError):
    ...


class FractionalMatrixPowerError(np.linalg.LinAlgError):
    ...


class _MatrixM1PowerOperator(LinearOperator):
    """
    A representation of the linear operator (A - I)^p.
    """
    def __init__(self, A, p) -> None:
        ...
