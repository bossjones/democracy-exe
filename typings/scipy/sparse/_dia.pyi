"""
This type stub file was generated by pyright.
"""

from ._matrix import spmatrix
from ._base import sparray
from ._data import _data_matrix

"""Sparse DIAgonal format"""
__docformat__ = ...
__all__ = ['dia_array', 'dia_matrix', 'isspmatrix_dia']
class _dia_base(_data_matrix):
    _format = ...
    def __init__(self, arg1, shape=..., dtype=..., copy=...) -> None:
        ...
    
    def __repr__(self): # -> str:
        ...
    
    def count_nonzero(self): # -> int:
        ...
    
    def sum(self, axis=..., dtype=..., out=...): # -> Any | matrix[Any, Any]:
        ...
    
    def todia(self, copy=...): # -> Self:
        ...
    
    def transpose(self, axes=..., copy=...): # -> dia_array:
        ...
    
    def diagonal(self, k=...): # -> NDArray[float_] | ndarray[Any, dtype[Any]]:
        ...
    
    def tocsc(self, copy=...): # -> csc_array:
        ...
    
    def tocoo(self, copy=...): # -> coo_array:
        ...
    
    def resize(self, *shape): # -> None:
        ...
    


def isspmatrix_dia(x): # -> bool:
    """Is `x` of dia_matrix type?

    Parameters
    ----------
    x
        object to check for being a dia matrix

    Returns
    -------
    bool
        True if `x` is a dia matrix, False otherwise

    Examples
    --------
    >>> from scipy.sparse import dia_array, dia_matrix, coo_matrix, isspmatrix_dia
    >>> isspmatrix_dia(dia_matrix([[5]]))
    True
    >>> isspmatrix_dia(dia_array([[5]]))
    False
    >>> isspmatrix_dia(coo_matrix([[5]]))
    False
    """
    ...

class dia_array(_dia_base, sparray):
    """
    Sparse array with DIAgonal storage.

    This can be instantiated in several ways:
        dia_array(D)
            where D is a 2-D ndarray

        dia_array(S)
            with another sparse array or matrix S (equivalent to S.todia())

        dia_array((M, N), [dtype])
            to construct an empty array with shape (M, N),
            dtype is optional, defaulting to dtype='d'.

        dia_array((data, offsets), shape=(M, N))
            where the ``data[k,:]`` stores the diagonal entries for
            diagonal ``offsets[k]`` (See example below)

    Attributes
    ----------
    dtype : dtype
        Data type of the array
    shape : 2-tuple
        Shape of the array
    ndim : int
        Number of dimensions (this is always 2)
    nnz
    size
    data
        DIA format data array of the array
    offsets
        DIA format offset array of the array
    T

    Notes
    -----

    Sparse arrays can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.

    Examples
    --------

    >>> import numpy as np
    >>> from scipy.sparse import dia_array
    >>> dia_array((3, 4), dtype=np.int8).toarray()
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=int8)

    >>> data = np.array([[1, 2, 3, 4]]).repeat(3, axis=0)
    >>> offsets = np.array([0, -1, 2])
    >>> dia_array((data, offsets), shape=(4, 4)).toarray()
    array([[1, 0, 3, 0],
           [1, 2, 0, 4],
           [0, 2, 3, 0],
           [0, 0, 3, 4]])

    >>> from scipy.sparse import dia_array
    >>> n = 10
    >>> ex = np.ones(n)
    >>> data = np.array([ex, 2 * ex, ex])
    >>> offsets = np.array([-1, 0, 1])
    >>> dia_array((data, offsets), shape=(n, n)).toarray()
    array([[2., 1., 0., ..., 0., 0., 0.],
           [1., 2., 1., ..., 0., 0., 0.],
           [0., 1., 2., ..., 0., 0., 0.],
           ...,
           [0., 0., 0., ..., 2., 1., 0.],
           [0., 0., 0., ..., 1., 2., 1.],
           [0., 0., 0., ..., 0., 1., 2.]])
    """
    ...


class dia_matrix(spmatrix, _dia_base):
    """
    Sparse matrix with DIAgonal storage.

    This can be instantiated in several ways:
        dia_matrix(D)
            where D is a 2-D ndarray

        dia_matrix(S)
            with another sparse array or matrix S (equivalent to S.todia())

        dia_matrix((M, N), [dtype])
            to construct an empty matrix with shape (M, N),
            dtype is optional, defaulting to dtype='d'.

        dia_matrix((data, offsets), shape=(M, N))
            where the ``data[k,:]`` stores the diagonal entries for
            diagonal ``offsets[k]`` (See example below)

    Attributes
    ----------
    dtype : dtype
        Data type of the matrix
    shape : 2-tuple
        Shape of the matrix
    ndim : int
        Number of dimensions (this is always 2)
    nnz
    size
    data
        DIA format data array of the matrix
    offsets
        DIA format offset array of the matrix
    T

    Notes
    -----

    Sparse matrices can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.

    Examples
    --------

    >>> import numpy as np
    >>> from scipy.sparse import dia_matrix
    >>> dia_matrix((3, 4), dtype=np.int8).toarray()
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=int8)

    >>> data = np.array([[1, 2, 3, 4]]).repeat(3, axis=0)
    >>> offsets = np.array([0, -1, 2])
    >>> dia_matrix((data, offsets), shape=(4, 4)).toarray()
    array([[1, 0, 3, 0],
           [1, 2, 0, 4],
           [0, 2, 3, 0],
           [0, 0, 3, 4]])

    >>> from scipy.sparse import dia_matrix
    >>> n = 10
    >>> ex = np.ones(n)
    >>> data = np.array([ex, 2 * ex, ex])
    >>> offsets = np.array([-1, 0, 1])
    >>> dia_matrix((data, offsets), shape=(n, n)).toarray()
    array([[2., 1., 0., ..., 0., 0., 0.],
           [1., 2., 1., ..., 0., 0., 0.],
           [0., 1., 2., ..., 0., 0., 0.],
           ...,
           [0., 0., 0., ..., 2., 1., 0.],
           [0., 0., 0., ..., 1., 2., 1.],
           [0., 0., 0., ..., 0., 1., 2.]])
    """
    ...


