"""
Equivalent Layer interpolators for harmonic functions
"""
import numpy as np
from numba import jit
from sklearn.utils.validation import check_is_fitted
from verde import get_region, median_distance
from verde.base import BaseGridder, check_fit_input, least_squares, n_1d_arrays

from ..forward.utils import distance_cartesian


class EQLHarmonic(BaseGridder):
    r"""
    3D Equivalent Layer interpolator for harmonic fields using Green's functions

    This gridder assumes Cartesian coordinates.

    Predict values of an harmonic function. It uses point sources to build the
    Equivalent Layer, fitting the coefficients that correspond to each point source in
    order to fit the data values. Uses as Green's functions the inverse distance between
    the grid coordinates and the point source:

    .. math::

        \phi_m(P, Q) = \frac{1}{|P - Q|}

    where :math:`P` and :math:`Q` are the coordinates of the observation point and the
    source, respectively.

    Parameters
    ----------
    damping : None or float
        The positive damping regularization parameter. Controls how much smoothness is
        imposed on the estimated coefficients. If None, no regularization is used.
    points : None or list of arrays (optional)
        List containing the coordinates of the point sources used as Equivalent Layer
        in the following order: (``easting``, ``northing``, ``upward``). If None,
        a default set of points will be created putting a single point source bellow
        each observation point at a relative depth proportional to the mean distance to
        the nearest k observation points [Cooper2000]_. Default None.
    depth_factor : float (optional)
        Adimensional factor to set the depth of each point source.
        If ``points`` is None, a default set of point will be created putting
        a single point source bellow each obervation point at a relative depth given by
        the product of the ``depth_factor`` and the mean distance to the nearest
        ``k_nearest`` obervation points. A greater ``depth_factor`` will increase the
        depth of the point source. This parameter is ignored if ``points`` is not None.
        Default 3 (following [Cooper2000]_).
    k_nearest : int (optional)
        Number of observation points used to compute the median distance to its nearest
        neighbours. This argument is passed to :func:`verde.mean_distance`. It's ignored
        if ``points`` is not None. Default 1.

    Attributes
    ----------
    points_ : 2d-array
        Coordinates of the point sources used to build the Equivalent Layer.
    coefs_ : array
        Estimated coefficients of every point source.
    region_ : tuple
        The boundaries (``[W, E, S, N]``) of the data used to fit the interpolator.
        Used as the default region for the :meth:`~harmonica.HarmonicEQL.grid` and
        :meth:`~harmonica.HarmonicEQL.scatter` methods.
    """

    def __init__(self, damping=None, points=None, depth_factor=3, k_nearest=1):
        self.damping = damping
        self.points = points
        self.depth_factor = depth_factor
        self.k_nearest = k_nearest

    def fit(self, coordinates, data, weights=None):
        """
        Fit the coefficients of the Equivalent Layer.

        The data region is captured and used as default for the
        :meth:`~harmonica.HarmonicEQL.grid` and :meth:`~harmonica.HarmonicEQL.scatter`
        methods.

        All input arrays must have the same shape.

        Parameters
        ----------
        coordinates : tuple of arrays
            Arrays with the coordinates of each data point. Should be in the
            following order: (easting, northing, upward, ...). Only easting
            and northing will be used, all subsequent coordinates will be
            ignored.
        data : array
            The data values of each data point.
        weights : None or array
            If not None, then the weights assigned to each data point.
            Typically, this should be 1 over the data uncertainty squared.

        Returns
        -------
        self
            Returns this estimator instance for chaining operations.
        """
        coordinates, data, weights = check_fit_input(coordinates, data, weights)
        # Capture the data region to use as a default when gridding.
        self.region_ = get_region(coordinates[:2])
        coordinates = n_1d_arrays(coordinates, 3)
        if self.points is None:
            # Put a single point source bellow each observation point at a depth three
            # times the median distance to the nearest k observation points.
            point_east, point_north, point_upward = tuple(i.copy() for i in coordinates)
            point_upward += self.depth_factor * median_distance(
                coordinates, k_nearest=self.k_nearest
            )
            self.points_ = (point_east, point_north, point_upward)
        else:
            self.points_ = n_1d_arrays(self.points, 3)
        jacobian = self.jacobian(coordinates, self.points_)
        self.coefs_ = least_squares(jacobian, data, weights, self.damping)
        return self

    def predict(self, coordinates):
        """
        Evaluate the estimated spline on the given set of points.

        Requires a fitted estimator (see :meth:`~harmonica.HarmonicEQL.fit`).

        Parameters
        ----------
        coordinates : tuple of arrays
            Arrays with the coordinates of each data point. Should be in the
            following order: (``easting``, ``northing``, ``upward``, ...). Only
            ``easting``, ``northing`` and ``upward`` will be used, all subsequent
            coordinates will be ignored.

        Returns
        -------
        data : array
            The data values evaluated on the given points.
        """
        # We know the gridder has been fitted if it has the coefs_
        check_is_fitted(self, ["coefs_"])
        shape = np.broadcast(*coordinates[:3]).shape
        size = np.broadcast(*coordinates[:3]).size
        dtype = coordinates[0].dtype
        coordinates = tuple(np.atleast_1d(i).ravel() for i in coordinates[:3])
        data = np.zeros(size, dtype=dtype)
        predict_numba(coordinates, self.points_, self.coefs_, data)
        return data.reshape(shape)

    def jacobian(
        self, coordinates, points, dtype="float64"
    ):  # pylint: disable=no-self-use
        """
        Make the Jacobian matrix for the Equivalent Layer.

        Each column of the Jacobian is the Green's function for a single point source
        evaluated on all observation points [Sandwell1987]_.

        Parameters
        ----------
        coordinates : tuple of arrays
            Arrays with the coordinates of each data point. Should be in the
            following order: (``easting``, ``northing``, ``upward``, ...). Only
            ``easting``, ``northing`` and ``upward`` will be used, all subsequent
            coordinates will be ignored.
        points : tuple of arrays
            Tuple of arrays containing the coordinates of the point sources used as
            Equivalent Layer in the following order: (``easting``, ``northing``,
            ``upward``).
        dtype : str or numpy dtype
            The type of the Jacobian array.

        Returns
        -------
        jacobian : 2D array
            The (n_data, n_points) Jacobian matrix.
        """
        n_data = coordinates[0].size
        n_points = points[0].size
        jac = np.zeros((n_data, n_points), dtype=dtype)
        jacobian_numba(coordinates, points, jac)
        return jac


@jit(nopython=True)
def predict_numba(coordinates, points, coeffs, result):
    """
    Calculate the predicted data using numba for speeding things up.
    """
    east, north, upward = coordinates[:]
    point_east, point_north, point_upward = points[:]
    for i in range(east.size):
        for j in range(point_east.size):
            result[i] += coeffs[j] * greens_func(
                east[i],
                north[i],
                upward[i],
                point_east[j],
                point_north[j],
                point_upward[j],
            )


@jit(nopython=True)
def greens_func(east, north, upward, point_east, point_north, point_upward):
    """
    Calculate the Green's function for the Equivalent Layer using Numba.
    """
    distance = distance_cartesian(
        (east, north, upward), (point_east, point_north, point_upward)
    )
    return 1 / distance


@jit(nopython=True)
def jacobian_numba(coordinates, points, jac):
    """
    Calculate the Jacobian matrix using numba to speed things up.
    """
    east, north, upward = coordinates[:]
    point_east, point_north, point_upward = points[:]
    for i in range(east.size):
        for j in range(point_east.size):
            jac[i, j] = greens_func(
                east[i],
                north[i],
                upward[i],
                point_east[j],
                point_north[j],
                point_upward[j],
            )
