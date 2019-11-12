"""
Equivalent layer for generic harmonic functions
"""
import numpy as np
from numba import jit
from sklearn.utils.validation import check_is_fitted
import verde as vd
import verde.base as vdb

from ..forward.utils import distance_cartesian


class EQLHarmonic(vdb.BaseGridder):
    r"""
    Equivalent-layer for generic harmonic functions (gravity, magnetics, etc).

    This equivalent layer can be used for:

    * Cartesian coordinates (geographic coordinates must be project before use)
    * Gravity and magnetic data (including derivatives)
    * Single data types
    * Interpolation
    * Upward continuation
    * Finite-difference based derivative calculations

    It cannot be used for:

    * Joint inversion of multiple data types (e.g., gravity + gravity
      gradients)
    * Reduction to the pole of magnetic total field anomaly data
    * Analytical derivative calculations

    Point sources are located beneath the observed potential-field measurement
    points by default [Cooper2000]_. Custom source locations can be used by
    specifying the *points* argument. Coefficients associated with each point
    source are estimated through linear least-squares with damping (Tikhonov
    0th order) regularization.

    The Green's function for point mass effects used is the inverse Cartesian
    distance between the grid coordinates and the point source:

    .. math::

        \phi(\bar{x}, \bar{x}') = \frac{1}{||\bar{x} - \bar{x}'||}

    where :math:`\bar{x}` and :math:`\bar{x}'` are the coordinate vectors of
    the observation point and the source, respectively.

    Parameters
    ----------
    damping : None or float
        The positive damping regularization parameter. Controls how much
        smoothness is imposed on the estimated coefficients.
        If None, no regularization is used.
    points : None or list of arrays (optional)
        List containing the coordinates of the point sources used as the
        equivalent layer. Coordinates are assumed to be in the following order:
        (``easting``, ``northing``, ``upward``).
        If None, will place one point source bellow each observation point at
        a fixed relative depth bellow the observation point [Cooper2000]_.
        Defaults to None.
    relative_depth : float
        Relative depth at which the point sources are placed beneath the
        observation points. Each source point will be set beneath each data
        point at a depth calculated as the elevation of the data point minus
        this constant *relative_depth*. Use positive numbers (negative numbers
        would mean point sources are above the data points). Ignored if
        *points* is specified.

    Attributes
    ----------
    points_ : 2d-array
        Coordinates of the point sources used to build the equivalent layer.
    coefs_ : array
        Estimated coefficients of every point source.
    region_ : tuple
        The boundaries (``[W, E, S, N]``) of the data used to fit the
        interpolator. Used as the default region for the
        :meth:`~harmonica.HarmonicEQL.grid` and
        :meth:`~harmonica.HarmonicEQL.scatter` methods.
    """

    def __init__(self, damping=None, points=None, relative_depth=500):
        self.damping = damping
        self.points = points
        self.relative_depth = relative_depth

    def fit(self, coordinates, data, weights=None):
        """
        Fit the coefficients of the equivalent layer.

        The data region is captured and used as default for the
        :meth:`~harmonica.HarmonicEQL.grid` and
        :meth:`~harmonica.HarmonicEQL.scatter` methods.

        All input arrays must have the same shape.

        Parameters
        ----------
        coordinates : tuple of arrays
            Arrays with the coordinates of each data point. Should be in the
            following order: (easting, northing, upward, ...).
            Only easting, northing, and upward will be used, all subsequent
            coordinates will be ignored.
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
        coordinates, data, weights = vdb.check_fit_input(coordinates, data, weights)
        # Capture the data region to use as a default when gridding.
        self.region_ = vd.get_region(coordinates[:2])
        coordinates = vdb.n_1d_arrays(coordinates, 3)
        if self.points is None:
            self.points_ = (
                coordinates[0],
                coordinates[1],
                coordinates[2] - self.relative_depth,
            )
        else:
            self.points_ = vdb.n_1d_arrays(self.points, 3)
        jacobian = self.jacobian(coordinates, self.points_)
        self.coefs_ = vdb.least_squares(jacobian, data, weights, self.damping)
        return self

    def predict(self, coordinates):
        """
        Evaluate the estimated equivalent layer on the given set of points.

        Requires a fitted estimator (see :meth:`~harmonica.HarmonicEQL.fit`).

        Parameters
        ----------
        coordinates : tuple of arrays
            Arrays with the coordinates of each data point. Should be in the
            following order: (``easting``, ``northing``, ``upward``, ...). Only
            ``easting``, ``northing`` and ``upward`` will be used, all
            subsequent coordinates will be ignored.

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
        Make the Jacobian matrix for the equivalent layer.

        Each column of the Jacobian is the Green's function for a single point
        source evaluated on all observation points.

        Parameters
        ----------
        coordinates : tuple of arrays
            Arrays with the coordinates of each data point. Should be in the
            following order: (``easting``, ``northing``, ``upward``, ...).
            Only ``easting``, ``northing`` and ``upward`` will be used, all
            subsequent coordinates will be ignored.
        points : tuple of arrays
            Tuple of arrays containing the coordinates of the point sources
            used as equivalent layer in the following order:
            (``easting``, ``northing``, ``upward``).
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
    Calculate the Green's function for the equivalent layer using Numba.
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
