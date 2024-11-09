"""
This type stub file was generated by pyright.
"""

from ._base import sparray
from ._compressed import _cs_matrix
from ._matrix import spmatrix

"""Compressed Sparse Column matrix format"""
__docformat__ = ...
__all__ = ['csc_array', 'csc_matrix', 'isspmatrix_csc']
class _csc_base(_cs_matrix):
    _format = ...
    def transpose(self, axes=..., copy=...): # -> csr_array:
        ...

    def __iter__(self): # -> Generator[csr_array, Any, None]:
        ...

    def tocsc(self, copy=...): # -> Self:
        ...

    def tocsr(self, copy=...): # -> csr_array:
        ...

    def nonzero(self): # -> tuple[Any, Any]:
        ...



def isspmatrix_csc(x): # -> bool:
    """Is `x` of csc_matrix type?

    Parameters
    ----------
    x
        object to check for being a csc matrix

    Returns
    -------
    bool
        True if `x` is a csc matrix, False otherwise

    Examples
    --------
    >>> from scipy.sparse import csc_array, csc_matrix, coo_matrix, isspmatrix_csc
    >>> isspmatrix_csc(csc_matrix([[5]]))
    True
    >>> isspmatrix_csc(csc_array([[5]]))
    False
    >>> isspmatrix_csc(coo_matrix([[5]]))
    False
    """
    ...

class csc_array(_csc_base, sparray):
    """
    Compressed Sparse Column array.

    This can be instantiated in several ways:
        csc_array(D)
            where D is a 2-D ndarray

        csc_array(S)
            with another sparse array or matrix S (equivalent to S.tocsc())

        csc_array((M, N), [dtype])
            to construct an empty array with shape (M, N)
            dtype is optional, defaulting to dtype='d'.

        csc_array((data, (row_ind, col_ind)), [shape=(M, N)])
            where ``data``, ``row_ind`` and ``col_ind`` satisfy the
            relationship ``a[row_ind[k], col_ind[k]] = data[k]``.

        csc_array((data, indices, indptr), [shape=(M, N)])
            is the standard CSC representation where the row indices for
            column i are stored in ``indices[indptr[i]:indptr[i+1]]``
            and their corresponding values are stored in
            ``data[indptr[i]:indptr[i+1]]``.  If the shape parameter is
            not supplied, the array dimensions are inferred from
            the index arrays.

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
        CSC format data array of the array
    indices
        CSC format index array of the array
    indptr
        CSC format index pointer array of the array
    has_sorted_indices
    has_canonical_format
    T

    Notes
    -----

    Sparse arrays can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.

    Advantages of the CSC format
        - efficient arithmetic operations CSC + CSC, CSC * CSC, etc.
        - efficient column slicing
        - fast matrix vector products (CSR, BSR may be faster)

    Disadvantages of the CSC format
      - slow row slicing operations (consider CSR)
      - changes to the sparsity structure are expensive (consider LIL or DOK)

    Canonical format
      - Within each column, indices are sorted by row.
      - There are no duplicate entries.

    Examples
    --------

    >>> import numpy as np
    >>> from scipy.sparse import csc_array
    >>> csc_array((3, 4), dtype=np.int8).toarray()
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=int8)

    >>> row = np.array([0, 2, 2, 0, 1, 2])
    >>> col = np.array([0, 0, 1, 2, 2, 2])
    >>> data = np.array([1, 2, 3, 4, 5, 6])
    >>> csc_array((data, (row, col)), shape=(3, 3)).toarray()
    array([[1, 0, 4],
           [0, 0, 5],
           [2, 3, 6]])

    >>> indptr = np.array([0, 2, 3, 6])
    >>> indices = np.array([0, 2, 2, 0, 1, 2])
    >>> data = np.array([1, 2, 3, 4, 5, 6])
    >>> csc_array((data, indices, indptr), shape=(3, 3)).toarray()
    array([[1, 0, 4],
           [0, 0, 5],
           [2, 3, 6]])

    """
    ...


class csc_matrix(spmatrix, _csc_base):
    """
    Compressed Sparse Column matrix.

    This can be instantiated in several ways:
        csc_matrix(D)
            where D is a 2-D ndarray

        csc_matrix(S)
            with another sparse array or matrix S (equivalent to S.tocsc())

        csc_matrix((M, N), [dtype])
            to construct an empty matrix with shape (M, N)
            dtype is optional, defaulting to dtype='d'.

        csc_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
            where ``data``, ``row_ind`` and ``col_ind`` satisfy the
            relationship ``a[row_ind[k], col_ind[k]] = data[k]``.

        csc_matrix((data, indices, indptr), [shape=(M, N)])
            is the standard CSC representation where the row indices for
            column i are stored in ``indices[indptr[i]:indptr[i+1]]``
            and their corresponding values are stored in
            ``data[indptr[i]:indptr[i+1]]``.  If the shape parameter is
            not supplied, the matrix dimensions are inferred from
            the index arrays.

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
        CSC format data array of the matrix
    indices
        CSC format index array of the matrix
    indptr
        CSC format index pointer array of the matrix
    has_sorted_indices
    has_canonical_format
    T

    Notes
    -----

    Sparse matrices can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.

    Advantages of the CSC format
        - efficient arithmetic operations CSC + CSC, CSC * CSC, etc.
        - efficient column slicing
        - fast matrix vector products (CSR, BSR may be faster)

    Disadvantages of the CSC format
      - slow row slicing operations (consider CSR)
      - changes to the sparsity structure are expensive (consider LIL or DOK)

    Canonical format
      - Within each column, indices are sorted by row.
      - There are no duplicate entries.

    Examples
    --------

    >>> import numpy as np
    >>> from scipy.sparse import csc_matrix
    >>> csc_matrix((3, 4), dtype=np.int8).toarray()
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=int8)

    >>> row = np.array([0, 2, 2, 0, 1, 2])
    >>> col = np.array([0, 0, 1, 2, 2, 2])
    >>> data = np.array([1, 2, 3, 4, 5, 6])
    >>> csc_matrix((data, (row, col)), shape=(3, 3)).toarray()
    array([[1, 0, 4],
           [0, 0, 5],
           [2, 3, 6]])

    >>> indptr = np.array([0, 2, 3, 6])
    >>> indices = np.array([0, 2, 2, 0, 1, 2])
    >>> data = np.array([1, 2, 3, 4, 5, 6])
    >>> csc_matrix((data, indices, indptr), shape=(3, 3)).toarray()
    array([[1, 0, 4],
           [0, 0, 5],
           [2, 3, 6]])

    """
    ...
