"""
This type stub file was generated by pyright.
"""

from ._base import _spbase

"""Base class for sparse matrice with a .data attribute

    subclasses must provide a _with_data() method that
    creates a new matrix with the same sparsity pattern
    as self but with a different data array

"""
__all__ = []
class _data_matrix(_spbase):
    def __init__(self, arg1) -> None:
        ...
    
    @property
    def dtype(self):
        ...
    
    @dtype.setter
    def dtype(self, newtype): # -> None:
        ...
    
    def __abs__(self):
        ...
    
    def __round__(self, ndigits=...):
        ...
    
    def __neg__(self):
        ...
    
    def __imul__(self, other): # -> Self | _NotImplementedType:
        ...
    
    def __itruediv__(self, other): # -> Self | _NotImplementedType:
        ...
    
    def astype(self, dtype, casting=..., copy=...): # -> Self:
        ...
    
    def conjugate(self, copy=...): # -> Self:
        ...
    
    def copy(self):
        ...
    
    def count_nonzero(self): # -> int:
        ...
    
    def power(self, n, dtype=...):
        """
        This function performs element-wise power.

        Parameters
        ----------
        n : scalar
            n is a non-zero scalar (nonzero avoids dense ones creation)
            If zero power is desired, special case it to use `np.ones`

        dtype : If dtype is not specified, the current dtype will be preserved.

        Raises
        ------
        NotImplementedError : if n is a zero scalar
            If zero power is desired, special case it to use
            `np.ones(A.shape, dtype=A.dtype)`
        """
        ...
    


class _minmax_mixin:
    """Mixin for min and max methods.

    These are not implemented for dia_matrix, hence the separate class.
    """
    def max(self, axis=..., out=...):
        """
        Return the maximum of the array/matrix or maximum along an axis.
        This takes all elements into account, not just the non-zero ones.

        Parameters
        ----------
        axis : {-2, -1, 0, 1, None} optional
            Axis along which the sum is computed. The default is to
            compute the maximum over all elements, returning
            a scalar (i.e., `axis` = `None`).

        out : None, optional
            This argument is in the signature *solely* for NumPy
            compatibility reasons. Do not pass in anything except
            for the default value, as this argument is not used.

        Returns
        -------
        amax : coo_matrix or scalar
            Maximum of `a`. If `axis` is None, the result is a scalar value.
            If `axis` is given, the result is a sparse.coo_matrix of dimension
            ``a.ndim - 1``.

        See Also
        --------
        min : The minimum value of a sparse array/matrix along a given axis.
        numpy.matrix.max : NumPy's implementation of 'max' for matrices

        """
        ...
    
    def min(self, axis=..., out=...):
        """
        Return the minimum of the array/matrix or maximum along an axis.
        This takes all elements into account, not just the non-zero ones.

        Parameters
        ----------
        axis : {-2, -1, 0, 1, None} optional
            Axis along which the sum is computed. The default is to
            compute the minimum over all elements, returning
            a scalar (i.e., `axis` = `None`).

        out : None, optional
            This argument is in the signature *solely* for NumPy
            compatibility reasons. Do not pass in anything except for
            the default value, as this argument is not used.

        Returns
        -------
        amin : coo_matrix or scalar
            Minimum of `a`. If `axis` is None, the result is a scalar value.
            If `axis` is given, the result is a sparse.coo_matrix of dimension
            ``a.ndim - 1``.

        See Also
        --------
        max : The maximum value of a sparse array/matrix along a given axis.
        numpy.matrix.min : NumPy's implementation of 'min' for matrices

        """
        ...
    
    def nanmax(self, axis=..., out=...):
        """
        Return the maximum of the array/matrix or maximum along an axis, ignoring any
        NaNs. This takes all elements into account, not just the non-zero
        ones.

        .. versionadded:: 1.11.0

        Parameters
        ----------
        axis : {-2, -1, 0, 1, None} optional
            Axis along which the maximum is computed. The default is to
            compute the maximum over all elements, returning
            a scalar (i.e., `axis` = `None`).

        out : None, optional
            This argument is in the signature *solely* for NumPy
            compatibility reasons. Do not pass in anything except
            for the default value, as this argument is not used.

        Returns
        -------
        amax : coo_matrix or scalar
            Maximum of `a`. If `axis` is None, the result is a scalar value.
            If `axis` is given, the result is a sparse.coo_matrix of dimension
            ``a.ndim - 1``.

        See Also
        --------
        nanmin : The minimum value of a sparse array/matrix along a given axis,
                 ignoring NaNs.
        max : The maximum value of a sparse array/matrix along a given axis,
              propagating NaNs.
        numpy.nanmax : NumPy's implementation of 'nanmax'.

        """
        ...
    
    def nanmin(self, axis=..., out=...):
        """
        Return the minimum of the array/matrix or minimum along an axis, ignoring any
        NaNs. This takes all elements into account, not just the non-zero
        ones.

        .. versionadded:: 1.11.0

        Parameters
        ----------
        axis : {-2, -1, 0, 1, None} optional
            Axis along which the minimum is computed. The default is to
            compute the minimum over all elements, returning
            a scalar (i.e., `axis` = `None`).

        out : None, optional
            This argument is in the signature *solely* for NumPy
            compatibility reasons. Do not pass in anything except for
            the default value, as this argument is not used.

        Returns
        -------
        amin : coo_matrix or scalar
            Minimum of `a`. If `axis` is None, the result is a scalar value.
            If `axis` is given, the result is a sparse.coo_matrix of dimension
            ``a.ndim - 1``.

        See Also
        --------
        nanmax : The maximum value of a sparse array/matrix along a given axis,
                 ignoring NaNs.
        min : The minimum value of a sparse array/matrix along a given axis,
              propagating NaNs.
        numpy.nanmin : NumPy's implementation of 'nanmin'.

        """
        ...
    
    def argmax(self, axis=..., out=...): # -> NDArray[Any] | int:
        """Return indices of maximum elements along an axis.

        Implicit zero elements are also taken into account. If there are
        several maximum values, the index of the first occurrence is returned.

        Parameters
        ----------
        axis : {-2, -1, 0, 1, None}, optional
            Axis along which the argmax is computed. If None (default), index
            of the maximum element in the flatten data is returned.
        out : None, optional
            This argument is in the signature *solely* for NumPy
            compatibility reasons. Do not pass in anything except for
            the default value, as this argument is not used.

        Returns
        -------
        ind : numpy.matrix or int
            Indices of maximum elements. If matrix, its size along `axis` is 1.
        """
        ...
    
    def argmin(self, axis=..., out=...): # -> NDArray[Any] | int:
        """Return indices of minimum elements along an axis.

        Implicit zero elements are also taken into account. If there are
        several minimum values, the index of the first occurrence is returned.

        Parameters
        ----------
        axis : {-2, -1, 0, 1, None}, optional
            Axis along which the argmin is computed. If None (default), index
            of the minimum element in the flatten data is returned.
        out : None, optional
            This argument is in the signature *solely* for NumPy
            compatibility reasons. Do not pass in anything except for
            the default value, as this argument is not used.

        Returns
        -------
         ind : numpy.matrix or int
            Indices of minimum elements. If matrix, its size along `axis` is 1.
        """
        ...
    


