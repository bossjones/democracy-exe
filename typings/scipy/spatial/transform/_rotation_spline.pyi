"""
This type stub file was generated by pyright.
"""

class RotationSpline:
    """Interpolate rotations with continuous angular rate and acceleration.

    The rotation vectors between each consecutive orientation are cubic
    functions of time and it is guaranteed that angular rate and acceleration
    are continuous. Such interpolation are analogous to cubic spline
    interpolation.

    Refer to [1]_ for math and implementation details.

    Parameters
    ----------
    times : array_like, shape (N,)
        Times of the known rotations. At least 2 times must be specified.
    rotations : `Rotation` instance
        Rotations to perform the interpolation between. Must contain N
        rotations.

    Methods
    -------
    __call__

    References
    ----------
    .. [1] `Smooth Attitude Interpolation
            <https://github.com/scipy/scipy/files/2932755/attitude_interpolation.pdf>`_

    Examples
    --------
    >>> from scipy.spatial.transform import Rotation, RotationSpline
    >>> import numpy as np

    Define the sequence of times and rotations from the Euler angles:

    >>> times = [0, 10, 20, 40]
    >>> angles = [[-10, 20, 30], [0, 15, 40], [-30, 45, 30], [20, 45, 90]]
    >>> rotations = Rotation.from_euler('XYZ', angles, degrees=True)

    Create the interpolator object:

    >>> spline = RotationSpline(times, rotations)

    Interpolate the Euler angles, angular rate and acceleration:

    >>> angular_rate = np.rad2deg(spline(times, 1))
    >>> angular_acceleration = np.rad2deg(spline(times, 2))
    >>> times_plot = np.linspace(times[0], times[-1], 100)
    >>> angles_plot = spline(times_plot).as_euler('XYZ', degrees=True)
    >>> angular_rate_plot = np.rad2deg(spline(times_plot, 1))
    >>> angular_acceleration_plot = np.rad2deg(spline(times_plot, 2))

    On this plot you see that Euler angles are continuous and smooth:

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(times_plot, angles_plot)
    >>> plt.plot(times, angles, 'x')
    >>> plt.title("Euler angles")
    >>> plt.show()

    The angular rate is also smooth:

    >>> plt.plot(times_plot, angular_rate_plot)
    >>> plt.plot(times, angular_rate, 'x')
    >>> plt.title("Angular rate")
    >>> plt.show()

    The angular acceleration is continuous, but not smooth. Also note that
    the angular acceleration is not a piecewise-linear function, because
    it is different from the second derivative of the rotation vector (which
    is a piecewise-linear function as in the cubic spline).

    >>> plt.plot(times_plot, angular_acceleration_plot)
    >>> plt.plot(times, angular_acceleration, 'x')
    >>> plt.title("Angular acceleration")
    >>> plt.show()
    """
    MAX_ITER = ...
    TOL = ...
    def __init__(self, times, rotations) -> None:
        ...

    def __call__(self, times, order=...): # -> Any:
        """Compute interpolated values.

        Parameters
        ----------
        times : float or array_like
            Times of interest.
        order : {0, 1, 2}, optional
            Order of differentiation:

                * 0 (default) : return Rotation
                * 1 : return the angular rate in rad/sec
                * 2 : return the angular acceleration in rad/sec/sec

        Returns
        -------
        Interpolated Rotation, angular rate or acceleration.
        """
        ...
