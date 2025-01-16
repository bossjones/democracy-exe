"""
This type stub file was generated by pyright.
"""

from scipy._lib.deprecation import _deprecate_positional_args

__all__ = ['bicg', 'bicgstab', 'cg', 'cgs', 'gmres', 'qmr']
@_deprecate_positional_args(version="1.14")
def bicg(A, b, x0=..., *, tol=..., maxiter=..., M=..., callback=..., atol=..., rtol=...): # -> tuple[ndarray[Any, dtype[Any]], Literal[0]] | tuple[ndarray[Any, dtype[Any]], int | Any]:
    """Use BIConjugate Gradient iteration to solve ``Ax = b``.

    Parameters
    ----------
    A : {sparse matrix, ndarray, LinearOperator}
        The real or complex N-by-N matrix of the linear system.
        Alternatively, ``A`` can be a linear operator which can
        produce ``Ax`` and ``A^T x`` using, e.g.,
        ``scipy.sparse.linalg.LinearOperator``.
    b : ndarray
        Right hand side of the linear system. Has shape (N,) or (N,1).
    x0 : ndarray
        Starting guess for the solution.
    rtol, atol : float, optional
        Parameters for the convergence test. For convergence,
        ``norm(b - A @ x) <= max(rtol*norm(b), atol)`` should be satisfied.
        The default is ``atol=0.`` and ``rtol=1e-5``.
    maxiter : integer
        Maximum number of iterations.  Iteration will stop after maxiter
        steps even if the specified tolerance has not been achieved.
    M : {sparse matrix, ndarray, LinearOperator}
        Preconditioner for A.  The preconditioner should approximate the
        inverse of A.  Effective preconditioning dramatically improves the
        rate of convergence, which implies that fewer iterations are needed
        to reach a given error tolerance.
    callback : function
        User-supplied function to call after each iteration.  It is called
        as callback(xk), where xk is the current solution vector.
    tol : float, optional, deprecated

        .. deprecated:: 1.12.0
           `bicg` keyword argument ``tol`` is deprecated in favor of ``rtol``
           and will be removed in SciPy 1.14.0.

    Returns
    -------
    x : ndarray
        The converged solution.
    info : integer
        Provides convergence information:
            0  : successful exit
            >0 : convergence to tolerance not achieved, number of iterations
            <0 : parameter breakdown

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csc_matrix
    >>> from scipy.sparse.linalg import bicg
    >>> A = csc_matrix([[3, 2, 0], [1, -1, 0], [0, 5, 1.]])
    >>> b = np.array([2., 4., -1.])
    >>> x, exitCode = bicg(A, b, atol=1e-5)
    >>> print(exitCode)  # 0 indicates successful convergence
    0
    >>> np.allclose(A.dot(x), b)
    True

    """
    ...

@_deprecate_positional_args(version="1.14")
def bicgstab(A, b, *, x0=..., tol=..., maxiter=..., M=..., callback=..., atol=..., rtol=...): # -> tuple[ndarray[Any, dtype[Any]], Literal[0]] | tuple[ndarray[Any, dtype[Any]], Literal[-10]] | tuple[ndarray[Any, dtype[Any]], Literal[-11]] | tuple[ndarray[Any, dtype[Any]], int | Any]:
    """Use BIConjugate Gradient STABilized iteration to solve ``Ax = b``.

    Parameters
    ----------
    A : {sparse matrix, ndarray, LinearOperator}
        The real or complex N-by-N matrix of the linear system.
        Alternatively, ``A`` can be a linear operator which can
        produce ``Ax`` and ``A^T x`` using, e.g.,
        ``scipy.sparse.linalg.LinearOperator``.
    b : ndarray
        Right hand side of the linear system. Has shape (N,) or (N,1).
    x0 : ndarray
        Starting guess for the solution.
    rtol, atol : float, optional
        Parameters for the convergence test. For convergence,
        ``norm(b - A @ x) <= max(rtol*norm(b), atol)`` should be satisfied.
        The default is ``atol=0.`` and ``rtol=1e-5``.
    maxiter : integer
        Maximum number of iterations.  Iteration will stop after maxiter
        steps even if the specified tolerance has not been achieved.
    M : {sparse matrix, ndarray, LinearOperator}
        Preconditioner for A.  The preconditioner should approximate the
        inverse of A.  Effective preconditioning dramatically improves the
        rate of convergence, which implies that fewer iterations are needed
        to reach a given error tolerance.
    callback : function
        User-supplied function to call after each iteration.  It is called
        as callback(xk), where xk is the current solution vector.
    tol : float, optional, deprecated

        .. deprecated:: 1.12.0
           `bicgstab` keyword argument ``tol`` is deprecated in favor of
           ``rtol`` and will be removed in SciPy 1.14.0.

    Returns
    -------
    x : ndarray
        The converged solution.
    info : integer
        Provides convergence information:
            0  : successful exit
            >0 : convergence to tolerance not achieved, number of iterations
            <0 : parameter breakdown

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csc_matrix
    >>> from scipy.sparse.linalg import bicgstab
    >>> R = np.array([[4, 2, 0, 1],
    ...               [3, 0, 0, 2],
    ...               [0, 1, 1, 1],
    ...               [0, 2, 1, 0]])
    >>> A = csc_matrix(R)
    >>> b = np.array([-1, -0.5, -1, 2])
    >>> x, exit_code = bicgstab(A, b, atol=1e-5)
    >>> print(exit_code)  # 0 indicates successful convergence
    0
    >>> np.allclose(A.dot(x), b)
    True

    """
    ...

@_deprecate_positional_args(version="1.14")
def cg(A, b, x0=..., *, tol=..., maxiter=..., M=..., callback=..., atol=..., rtol=...): # -> tuple[ndarray[Any, dtype[Any]], Literal[0]] | tuple[NDArray[Any] | Any, Literal[0]] | tuple[NDArray[Any] | Any, int | Any]:
    """Use Conjugate Gradient iteration to solve ``Ax = b``.

    Parameters
    ----------
    A : {sparse matrix, ndarray, LinearOperator}
        The real or complex N-by-N matrix of the linear system.
        ``A`` must represent a hermitian, positive definite matrix.
        Alternatively, ``A`` can be a linear operator which can
        produce ``Ax`` using, e.g.,
        ``scipy.sparse.linalg.LinearOperator``.
    b : ndarray
        Right hand side of the linear system. Has shape (N,) or (N,1).
    x0 : ndarray
        Starting guess for the solution.
    rtol, atol : float, optional
        Parameters for the convergence test. For convergence,
        ``norm(b - A @ x) <= max(rtol*norm(b), atol)`` should be satisfied.
        The default is ``atol=0.`` and ``rtol=1e-5``.
    maxiter : integer
        Maximum number of iterations.  Iteration will stop after maxiter
        steps even if the specified tolerance has not been achieved.
    M : {sparse matrix, ndarray, LinearOperator}
        Preconditioner for A.  The preconditioner should approximate the
        inverse of A.  Effective preconditioning dramatically improves the
        rate of convergence, which implies that fewer iterations are needed
        to reach a given error tolerance.
    callback : function
        User-supplied function to call after each iteration.  It is called
        as callback(xk), where xk is the current solution vector.
    tol : float, optional, deprecated

        .. deprecated:: 1.12.0
           `cg` keyword argument ``tol`` is deprecated in favor of ``rtol`` and
           will be removed in SciPy 1.14.0.

    Returns
    -------
    x : ndarray
        The converged solution.
    info : integer
        Provides convergence information:
            0  : successful exit
            >0 : convergence to tolerance not achieved, number of iterations

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csc_matrix
    >>> from scipy.sparse.linalg import cg
    >>> P = np.array([[4, 0, 1, 0],
    ...               [0, 5, 0, 0],
    ...               [1, 0, 3, 2],
    ...               [0, 0, 2, 4]])
    >>> A = csc_matrix(P)
    >>> b = np.array([-1, -0.5, -1, 2])
    >>> x, exit_code = cg(A, b, atol=1e-5)
    >>> print(exit_code)    # 0 indicates successful convergence
    0
    >>> np.allclose(A.dot(x), b)
    True

    """
    ...

@_deprecate_positional_args(version="1.14")
def cgs(A, b, x0=..., *, tol=..., maxiter=..., M=..., callback=..., atol=..., rtol=...): # -> tuple[ndarray[Any, dtype[Any]], Literal[0]] | tuple[NDArray[Any] | Any, Literal[0]] | tuple[Callable[..., Any], Literal[-10]] | tuple[NDArray[Any] | Any, int | Any]:
    """Use Conjugate Gradient Squared iteration to solve ``Ax = b``.

    Parameters
    ----------
    A : {sparse matrix, ndarray, LinearOperator}
        The real-valued N-by-N matrix of the linear system.
        Alternatively, ``A`` can be a linear operator which can
        produce ``Ax`` using, e.g.,
        ``scipy.sparse.linalg.LinearOperator``.
    b : ndarray
        Right hand side of the linear system. Has shape (N,) or (N,1).
    x0 : ndarray
        Starting guess for the solution.
    rtol, atol : float, optional
        Parameters for the convergence test. For convergence,
        ``norm(b - A @ x) <= max(rtol*norm(b), atol)`` should be satisfied.
        The default is ``atol=0.`` and ``rtol=1e-5``.
    maxiter : integer
        Maximum number of iterations.  Iteration will stop after maxiter
        steps even if the specified tolerance has not been achieved.
    M : {sparse matrix, ndarray, LinearOperator}
        Preconditioner for A.  The preconditioner should approximate the
        inverse of A.  Effective preconditioning dramatically improves the
        rate of convergence, which implies that fewer iterations are needed
        to reach a given error tolerance.
    callback : function
        User-supplied function to call after each iteration.  It is called
        as callback(xk), where xk is the current solution vector.
    tol : float, optional, deprecated

        .. deprecated:: 1.12.0
           `cgs` keyword argument ``tol`` is deprecated in favor of ``rtol``
           and will be removed in SciPy 1.14.0.

    Returns
    -------
    x : ndarray
        The converged solution.
    info : integer
        Provides convergence information:
            0  : successful exit
            >0 : convergence to tolerance not achieved, number of iterations
            <0 : parameter breakdown

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csc_matrix
    >>> from scipy.sparse.linalg import cgs
    >>> R = np.array([[4, 2, 0, 1],
    ...               [3, 0, 0, 2],
    ...               [0, 1, 1, 1],
    ...               [0, 2, 1, 0]])
    >>> A = csc_matrix(R)
    >>> b = np.array([-1, -0.5, -1, 2])
    >>> x, exit_code = cgs(A, b)
    >>> print(exit_code)  # 0 indicates successful convergence
    0
    >>> np.allclose(A.dot(x), b)
    True

    """
    ...

@_deprecate_positional_args(version="1.14")
def gmres(A, b, x0=..., *, tol=..., restart=..., maxiter=..., M=..., callback=..., restrt=..., atol=..., callback_type=..., rtol=...):
    """
    Use Generalized Minimal RESidual iteration to solve ``Ax = b``.

    Parameters
    ----------
    A : {sparse matrix, ndarray, LinearOperator}
        The real or complex N-by-N matrix of the linear system.
        Alternatively, ``A`` can be a linear operator which can
        produce ``Ax`` using, e.g.,
        ``scipy.sparse.linalg.LinearOperator``.
    b : ndarray
        Right hand side of the linear system. Has shape (N,) or (N,1).
    x0 : ndarray
        Starting guess for the solution (a vector of zeros by default).
    atol, rtol : float
        Parameters for the convergence test. For convergence,
        ``norm(b - A @ x) <= max(rtol*norm(b), atol)`` should be satisfied.
        The default is ``atol=0.`` and ``rtol=1e-5``.
    restart : int, optional
        Number of iterations between restarts. Larger values increase
        iteration cost, but may be necessary for convergence.
        If omitted, ``min(20, n)`` is used.
    maxiter : int, optional
        Maximum number of iterations (restart cycles).  Iteration will stop
        after maxiter steps even if the specified tolerance has not been
        achieved. See `callback_type`.
    M : {sparse matrix, ndarray, LinearOperator}
        Inverse of the preconditioner of A.  M should approximate the
        inverse of A and be easy to solve for (see Notes).  Effective
        preconditioning dramatically improves the rate of convergence,
        which implies that fewer iterations are needed to reach a given
        error tolerance.  By default, no preconditioner is used.
        In this implementation, left preconditioning is used,
        and the preconditioned residual is minimized. However, the final
        convergence is tested with respect to the ``b - A @ x`` residual.
    callback : function
        User-supplied function to call after each iteration.  It is called
        as `callback(args)`, where `args` are selected by `callback_type`.
    callback_type : {'x', 'pr_norm', 'legacy'}, optional
        Callback function argument requested:
          - ``x``: current iterate (ndarray), called on every restart
          - ``pr_norm``: relative (preconditioned) residual norm (float),
            called on every inner iteration
          - ``legacy`` (default): same as ``pr_norm``, but also changes the
            meaning of `maxiter` to count inner iterations instead of restart
            cycles.

        This keyword has no effect if `callback` is not set.
    restrt : int, optional, deprecated

        .. deprecated:: 0.11.0
           `gmres` keyword argument ``restrt`` is deprecated in favor of
           ``restart`` and will be removed in SciPy 1.14.0.
    tol : float, optional, deprecated

        .. deprecated:: 1.12.0
           `gmres` keyword argument ``tol`` is deprecated in favor of ``rtol``
           and will be removed in SciPy 1.14.0

    Returns
    -------
    x : ndarray
        The converged solution.
    info : int
        Provides convergence information:
            0  : successful exit
            >0 : convergence to tolerance not achieved, number of iterations

    See Also
    --------
    LinearOperator

    Notes
    -----
    A preconditioner, P, is chosen such that P is close to A but easy to solve
    for. The preconditioner parameter required by this routine is
    ``M = P^-1``. The inverse should preferably not be calculated
    explicitly.  Rather, use the following template to produce M::

      # Construct a linear operator that computes P^-1 @ x.
      import scipy.sparse.linalg as spla
      M_x = lambda x: spla.spsolve(P, x)
      M = spla.LinearOperator((n, n), M_x)

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csc_matrix
    >>> from scipy.sparse.linalg import gmres
    >>> A = csc_matrix([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=float)
    >>> b = np.array([2, 4, -1], dtype=float)
    >>> x, exitCode = gmres(A, b, atol=1e-5)
    >>> print(exitCode)            # 0 indicates successful convergence
    0
    >>> np.allclose(A.dot(x), b)
    True
    """
    ...

@_deprecate_positional_args(version="1.14")
def qmr(A, b, x0=..., *, tol=..., maxiter=..., M1=..., M2=..., callback=..., atol=..., rtol=...): # -> tuple[ndarray[Any, dtype[Any]], Literal[0]] | tuple[NDArray[Any] | Any, Literal[0]] | tuple[NDArray[Any] | Any, Literal[-10]] | tuple[NDArray[Any] | Any, Literal[-15]] | tuple[NDArray[Any] | Any, Literal[-13]] | tuple[NDArray[Any] | Any, int | Any]:
    """Use Quasi-Minimal Residual iteration to solve ``Ax = b``.

    Parameters
    ----------
    A : {sparse matrix, ndarray, LinearOperator}
        The real-valued N-by-N matrix of the linear system.
        Alternatively, ``A`` can be a linear operator which can
        produce ``Ax`` and ``A^T x`` using, e.g.,
        ``scipy.sparse.linalg.LinearOperator``.
    b : ndarray
        Right hand side of the linear system. Has shape (N,) or (N,1).
    x0 : ndarray
        Starting guess for the solution.
    atol, rtol : float, optional
        Parameters for the convergence test. For convergence,
        ``norm(b - A @ x) <= max(rtol*norm(b), atol)`` should be satisfied.
        The default is ``atol=0.`` and ``rtol=1e-5``.
    maxiter : integer
        Maximum number of iterations.  Iteration will stop after maxiter
        steps even if the specified tolerance has not been achieved.
    M1 : {sparse matrix, ndarray, LinearOperator}
        Left preconditioner for A.
    M2 : {sparse matrix, ndarray, LinearOperator}
        Right preconditioner for A. Used together with the left
        preconditioner M1.  The matrix M1@A@M2 should have better
        conditioned than A alone.
    callback : function
        User-supplied function to call after each iteration.  It is called
        as callback(xk), where xk is the current solution vector.
    tol : float, optional, deprecated

        .. deprecated:: 1.12.0
           `qmr` keyword argument ``tol`` is deprecated in favor of ``rtol``
           and will be removed in SciPy 1.14.0.

    Returns
    -------
    x : ndarray
        The converged solution.
    info : integer
        Provides convergence information:
            0  : successful exit
            >0 : convergence to tolerance not achieved, number of iterations
            <0 : parameter breakdown

    See Also
    --------
    LinearOperator

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csc_matrix
    >>> from scipy.sparse.linalg import qmr
    >>> A = csc_matrix([[3., 2., 0.], [1., -1., 0.], [0., 5., 1.]])
    >>> b = np.array([2., 4., -1.])
    >>> x, exitCode = qmr(A, b, atol=1e-5)
    >>> print(exitCode)            # 0 indicates successful convergence
    0
    >>> np.allclose(A.dot(x), b)
    True
    """
    ...

