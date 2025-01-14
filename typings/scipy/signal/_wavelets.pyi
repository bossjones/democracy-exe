"""
This type stub file was generated by pyright.
"""

__all__ = ['daub', 'qmf', 'cascade', 'morlet', 'ricker', 'morlet2', 'cwt']
_msg = ...
def daub(p): # -> NDArray[Any] | Any:
    """
    The coefficients for the FIR low-pass filter producing Daubechies wavelets.

    .. deprecated:: 1.12.0

        scipy.signal.daub is deprecated in SciPy 1.12 and will be removed
        in SciPy 1.15. We recommend using PyWavelets instead.

    p>=1 gives the order of the zero at f=1/2.
    There are 2p filter coefficients.

    Parameters
    ----------
    p : int
        Order of the zero at f=1/2, can have values from 1 to 34.

    Returns
    -------
    daub : ndarray
        Return

    """
    ...

def qmf(hk):
    """
    Return high-pass qmf filter from low-pass

    .. deprecated:: 1.12.0

        scipy.signal.qmf is deprecated in SciPy 1.12 and will be removed
        in SciPy 1.15. We recommend using PyWavelets instead.

    Parameters
    ----------
    hk : array_like
        Coefficients of high-pass filter.

    Returns
    -------
    array_like
        High-pass filter coefficients.

    """
    ...

def cascade(hk, J=...): # -> tuple[NDArray[floating[Any]], NDArray[floating[Any]], NDArray[floating[Any]]]:
    """
    Return (x, phi, psi) at dyadic points ``K/2**J`` from filter coefficients.

    .. deprecated:: 1.12.0

        scipy.signal.cascade is deprecated in SciPy 1.12 and will be removed
        in SciPy 1.15. We recommend using PyWavelets instead.

    Parameters
    ----------
    hk : array_like
        Coefficients of low-pass filter.
    J : int, optional
        Values will be computed at grid points ``K/2**J``. Default is 7.

    Returns
    -------
    x : ndarray
        The dyadic points ``K/2**J`` for ``K=0...N * (2**J)-1`` where
        ``len(hk) = len(gk) = N+1``.
    phi : ndarray
        The scaling function ``phi(x)`` at `x`:
        ``phi(x) = sum(hk * phi(2x-k))``, where k is from 0 to N.
    psi : ndarray, optional
        The wavelet function ``psi(x)`` at `x`:
        ``phi(x) = sum(gk * phi(2x-k))``, where k is from 0 to N.
        `psi` is only returned if `gk` is not None.

    Notes
    -----
    The algorithm uses the vector cascade algorithm described by Strang and
    Nguyen in "Wavelets and Filter Banks".  It builds a dictionary of values
    and slices for quick reuse.  Then inserts vectors into final vector at the
    end.

    """
    ...

def morlet(M, w=..., s=..., complete=...): # -> Any:
    """
    Complex Morlet wavelet.

    .. deprecated:: 1.12.0

        scipy.signal.morlet is deprecated in SciPy 1.12 and will be removed
        in SciPy 1.15. We recommend using PyWavelets instead.

    Parameters
    ----------
    M : int
        Length of the wavelet.
    w : float, optional
        Omega0. Default is 5
    s : float, optional
        Scaling factor, windowed from ``-s*2*pi`` to ``+s*2*pi``. Default is 1.
    complete : bool, optional
        Whether to use the complete or the standard version.

    Returns
    -------
    morlet : (M,) ndarray

    See Also
    --------
    morlet2 : Implementation of Morlet wavelet, compatible with `cwt`.
    scipy.signal.gausspulse

    Notes
    -----
    The standard version::

        pi**-0.25 * exp(1j*w*x) * exp(-0.5*(x**2))

    This commonly used wavelet is often referred to simply as the
    Morlet wavelet.  Note that this simplified version can cause
    admissibility problems at low values of `w`.

    The complete version::

        pi**-0.25 * (exp(1j*w*x) - exp(-0.5*(w**2))) * exp(-0.5*(x**2))

    This version has a correction
    term to improve admissibility. For `w` greater than 5, the
    correction term is negligible.

    Note that the energy of the return wavelet is not normalised
    according to `s`.

    The fundamental frequency of this wavelet in Hz is given
    by ``f = 2*s*w*r / M`` where `r` is the sampling rate.

    Note: This function was created before `cwt` and is not compatible
    with it.

    Examples
    --------
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> M = 100
    >>> s = 4.0
    >>> w = 2.0
    >>> wavelet = signal.morlet(M, s, w)
    >>> plt.plot(wavelet.real, label="real")
    >>> plt.plot(wavelet.imag, label="imag")
    >>> plt.legend()
    >>> plt.show()

    """
    ...

def ricker(points, a):
    """
    Return a Ricker wavelet, also known as the "Mexican hat wavelet".

    .. deprecated:: 1.12.0

        scipy.signal.ricker is deprecated in SciPy 1.12 and will be removed
        in SciPy 1.15. We recommend using PyWavelets instead.

    It models the function:

        ``A * (1 - (x/a)**2) * exp(-0.5*(x/a)**2)``,

    where ``A = 2/(sqrt(3*a)*(pi**0.25))``.

    Parameters
    ----------
    points : int
        Number of points in `vector`.
        Will be centered around 0.
    a : scalar
        Width parameter of the wavelet.

    Returns
    -------
    vector : (N,) ndarray
        Array of length `points` in shape of ricker curve.

    Examples
    --------
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> points = 100
    >>> a = 4.0
    >>> vec2 = signal.ricker(points, a)
    >>> print(len(vec2))
    100
    >>> plt.plot(vec2)
    >>> plt.show()

    """
    ...

def morlet2(M, s, w=...): # -> Any:
    """
    Complex Morlet wavelet, designed to work with `cwt`.

    .. deprecated:: 1.12.0

        scipy.signal.morlet2 is deprecated in SciPy 1.12 and will be removed
        in SciPy 1.15. We recommend using PyWavelets instead.

    Returns the complete version of morlet wavelet, normalised
    according to `s`::

        exp(1j*w*x/s) * exp(-0.5*(x/s)**2) * pi**(-0.25) * sqrt(1/s)

    Parameters
    ----------
    M : int
        Length of the wavelet.
    s : float
        Width parameter of the wavelet.
    w : float, optional
        Omega0. Default is 5

    Returns
    -------
    morlet : (M,) ndarray

    See Also
    --------
    morlet : Implementation of Morlet wavelet, incompatible with `cwt`

    Notes
    -----

    .. versionadded:: 1.4.0

    This function was designed to work with `cwt`. Because `morlet2`
    returns an array of complex numbers, the `dtype` argument of `cwt`
    should be set to `complex128` for best results.

    Note the difference in implementation with `morlet`.
    The fundamental frequency of this wavelet in Hz is given by::

        f = w*fs / (2*s*np.pi)

    where ``fs`` is the sampling rate and `s` is the wavelet width parameter.
    Similarly we can get the wavelet width parameter at ``f``::

        s = w*fs / (2*f*np.pi)

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt

    >>> M = 100
    >>> s = 4.0
    >>> w = 2.0
    >>> wavelet = signal.morlet2(M, s, w)
    >>> plt.plot(abs(wavelet))
    >>> plt.show()

    This example shows basic use of `morlet2` with `cwt` in time-frequency
    analysis:

    >>> t, dt = np.linspace(0, 1, 200, retstep=True)
    >>> fs = 1/dt
    >>> w = 6.
    >>> sig = np.cos(2*np.pi*(50 + 10*t)*t) + np.sin(40*np.pi*t)
    >>> freq = np.linspace(1, fs/2, 100)
    >>> widths = w*fs / (2*freq*np.pi)
    >>> cwtm = signal.cwt(sig, signal.morlet2, widths, w=w)
    >>> plt.pcolormesh(t, freq, np.abs(cwtm), cmap='viridis', shading='gouraud')
    >>> plt.show()

    """
    ...

def cwt(data, wavelet, widths, dtype=..., **kwargs): # -> NDArray[complexfloating[_64Bit, _64Bit] | floating[_64Bit]]:
    """
    Continuous wavelet transform.

    .. deprecated:: 1.12.0

        scipy.signal.cwt is deprecated in SciPy 1.12 and will be removed
        in SciPy 1.15. We recommend using PyWavelets instead.

    Performs a continuous wavelet transform on `data`,
    using the `wavelet` function. A CWT performs a convolution
    with `data` using the `wavelet` function, which is characterized
    by a width parameter and length parameter. The `wavelet` function
    is allowed to be complex.

    Parameters
    ----------
    data : (N,) ndarray
        data on which to perform the transform.
    wavelet : function
        Wavelet function, which should take 2 arguments.
        The first argument is the number of points that the returned vector
        will have (len(wavelet(length,width)) == length).
        The second is a width parameter, defining the size of the wavelet
        (e.g. standard deviation of a gaussian). See `ricker`, which
        satisfies these requirements.
    widths : (M,) sequence
        Widths to use for transform.
    dtype : data-type, optional
        The desired data type of output. Defaults to ``float64`` if the
        output of `wavelet` is real and ``complex128`` if it is complex.

        .. versionadded:: 1.4.0

    kwargs
        Keyword arguments passed to wavelet function.

        .. versionadded:: 1.4.0

    Returns
    -------
    cwt: (M, N) ndarray
        Will have shape of (len(widths), len(data)).

    Notes
    -----

    .. versionadded:: 1.4.0

    For non-symmetric, complex-valued wavelets, the input signal is convolved
    with the time-reversed complex-conjugate of the wavelet data [1].

    ::

        length = min(10 * width[ii], len(data))
        cwt[ii,:] = signal.convolve(data, np.conj(wavelet(length, width[ii],
                                        **kwargs))[::-1], mode='same')

    References
    ----------
    .. [1] S. Mallat, "A Wavelet Tour of Signal Processing (3rd Edition)",
        Academic Press, 2009.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> t = np.linspace(-1, 1, 200, endpoint=False)
    >>> sig  = np.cos(2 * np.pi * 7 * t) + signal.gausspulse(t - 0.4, fc=2)
    >>> widths = np.arange(1, 31)
    >>> cwtmatr = signal.cwt(sig, signal.ricker, widths)

    .. note:: For cwt matrix plotting it is advisable to flip the y-axis

    >>> cwtmatr_yflip = np.flipud(cwtmatr)
    >>> plt.imshow(cwtmatr_yflip, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
    ...            vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
    >>> plt.show()
    """
    ...

