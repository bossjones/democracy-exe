"""
This type stub file was generated by pyright.
"""

"""Cholesky decomposition functions."""
__all__ = ['cholesky', 'cho_factor', 'cho_solve', 'cholesky_banded', 'cho_solve_banded']
def cholesky(a, lower=..., overwrite_a=..., check_finite=...): # -> NDArray[Any]:
    """
    Compute the Cholesky decomposition of a matrix.

    Returns the Cholesky decomposition, :math:`A = L L^*` or
    :math:`A = U^* U` of a Hermitian positive-definite matrix A.

    Parameters
    ----------
    a : (M, M) array_like
        Matrix to be decomposed
    lower : bool, optional
        Whether to compute the upper- or lower-triangular Cholesky
        factorization.  Default is upper-triangular.
    overwrite_a : bool, optional
        Whether to overwrite data in `a` (may improve performance).
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    c : (M, M) ndarray
        Upper- or lower-triangular Cholesky factor of `a`.

    Raises
    ------
    LinAlgError : if decomposition fails.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import cholesky
    >>> a = np.array([[1,-2j],[2j,5]])
    >>> L = cholesky(a, lower=True)
    >>> L
    array([[ 1.+0.j,  0.+0.j],
           [ 0.+2.j,  1.+0.j]])
    >>> L @ L.T.conj()
    array([[ 1.+0.j,  0.-2.j],
           [ 0.+2.j,  5.+0.j]])

    """
    ...

def cho_factor(a, lower=..., overwrite_a=..., check_finite=...): # -> tuple[NDArray[Any] | Any, bool]:
    """
    Compute the Cholesky decomposition of a matrix, to use in cho_solve

    Returns a matrix containing the Cholesky decomposition,
    ``A = L L*`` or ``A = U* U`` of a Hermitian positive-definite matrix `a`.
    The return value can be directly used as the first parameter to cho_solve.

    .. warning::
        The returned matrix also contains random data in the entries not
        used by the Cholesky decomposition. If you need to zero these
        entries, use the function `cholesky` instead.

    Parameters
    ----------
    a : (M, M) array_like
        Matrix to be decomposed
    lower : bool, optional
        Whether to compute the upper or lower triangular Cholesky factorization
        (Default: upper-triangular)
    overwrite_a : bool, optional
        Whether to overwrite data in a (may improve performance)
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    c : (M, M) ndarray
        Matrix whose upper or lower triangle contains the Cholesky factor
        of `a`. Other parts of the matrix contain random data.
    lower : bool
        Flag indicating whether the factor is in the lower or upper triangle

    Raises
    ------
    LinAlgError
        Raised if decomposition fails.

    See Also
    --------
    cho_solve : Solve a linear set equations using the Cholesky factorization
                of a matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import cho_factor
    >>> A = np.array([[9, 3, 1, 5], [3, 7, 5, 1], [1, 5, 9, 2], [5, 1, 2, 6]])
    >>> c, low = cho_factor(A)
    >>> c
    array([[3.        , 1.        , 0.33333333, 1.66666667],
           [3.        , 2.44948974, 1.90515869, -0.27216553],
           [1.        , 5.        , 2.29330749, 0.8559528 ],
           [5.        , 1.        , 2.        , 1.55418563]])
    >>> np.allclose(np.triu(c).T @ np. triu(c) - A, np.zeros((4, 4)))
    True

    """
    ...

def cho_solve(c_and_lower, b, overwrite_b=..., check_finite=...): # -> NDArray[Any]:
    """Solve the linear equations A x = b, given the Cholesky factorization of A.

    Parameters
    ----------
    (c, lower) : tuple, (array, bool)
        Cholesky factorization of a, as given by cho_factor
    b : array
        Right-hand side
    overwrite_b : bool, optional
        Whether to overwrite data in b (may improve performance)
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    x : array
        The solution to the system A x = b

    See Also
    --------
    cho_factor : Cholesky factorization of a matrix

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import cho_factor, cho_solve
    >>> A = np.array([[9, 3, 1, 5], [3, 7, 5, 1], [1, 5, 9, 2], [5, 1, 2, 6]])
    >>> c, low = cho_factor(A)
    >>> x = cho_solve((c, low), [1, 1, 1, 1])
    >>> np.allclose(A @ x - [1, 1, 1, 1], np.zeros(4))
    True

    """
    ...

def cholesky_banded(ab, overwrite_ab=..., lower=..., check_finite=...): # -> NDArray[Any]:
    """
    Cholesky decompose a banded Hermitian positive-definite matrix

    The matrix a is stored in ab either in lower-diagonal or upper-
    diagonal ordered form::

        ab[u + i - j, j] == a[i,j]        (if upper form; i <= j)
        ab[    i - j, j] == a[i,j]        (if lower form; i >= j)

    Example of ab (shape of a is (6,6), u=2)::

        upper form:
        *   *   a02 a13 a24 a35
        *   a01 a12 a23 a34 a45
        a00 a11 a22 a33 a44 a55

        lower form:
        a00 a11 a22 a33 a44 a55
        a10 a21 a32 a43 a54 *
        a20 a31 a42 a53 *   *

    Parameters
    ----------
    ab : (u + 1, M) array_like
        Banded matrix
    overwrite_ab : bool, optional
        Discard data in ab (may enhance performance)
    lower : bool, optional
        Is the matrix in the lower form. (Default is upper form)
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    c : (u + 1, M) ndarray
        Cholesky factorization of a, in the same banded format as ab

    See Also
    --------
    cho_solve_banded :
        Solve a linear set equations, given the Cholesky factorization
        of a banded Hermitian.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import cholesky_banded
    >>> from numpy import allclose, zeros, diag
    >>> Ab = np.array([[0, 0, 1j, 2, 3j], [0, -1, -2, 3, 4], [9, 8, 7, 6, 9]])
    >>> A = np.diag(Ab[0,2:], k=2) + np.diag(Ab[1,1:], k=1)
    >>> A = A + A.conj().T + np.diag(Ab[2, :])
    >>> c = cholesky_banded(Ab)
    >>> C = np.diag(c[0, 2:], k=2) + np.diag(c[1, 1:], k=1) + np.diag(c[2, :])
    >>> np.allclose(C.conj().T @ C - A, np.zeros((5, 5)))
    True

    """
    ...

def cho_solve_banded(cb_and_lower, b, overwrite_b=..., check_finite=...): # -> NDArray[Any]:
    """
    Solve the linear equations ``A x = b``, given the Cholesky factorization of
    the banded Hermitian ``A``.

    Parameters
    ----------
    (cb, lower) : tuple, (ndarray, bool)
        `cb` is the Cholesky factorization of A, as given by cholesky_banded.
        `lower` must be the same value that was given to cholesky_banded.
    b : array_like
        Right-hand side
    overwrite_b : bool, optional
        If True, the function will overwrite the values in `b`.
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    x : array
        The solution to the system A x = b

    See Also
    --------
    cholesky_banded : Cholesky factorization of a banded matrix

    Notes
    -----

    .. versionadded:: 0.8.0

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import cholesky_banded, cho_solve_banded
    >>> Ab = np.array([[0, 0, 1j, 2, 3j], [0, -1, -2, 3, 4], [9, 8, 7, 6, 9]])
    >>> A = np.diag(Ab[0,2:], k=2) + np.diag(Ab[1,1:], k=1)
    >>> A = A + A.conj().T + np.diag(Ab[2, :])
    >>> c = cholesky_banded(Ab)
    >>> x = cho_solve_banded((c, False), np.ones(5))
    >>> np.allclose(A @ x - np.ones(5), np.zeros(5))
    True

    """
    ...

