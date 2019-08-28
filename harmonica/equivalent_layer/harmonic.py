"""
Equivalent Layer interpolators for harmonic functions
"""
import numpy as np
from numba import jit
from scipy.sparse import csr_matrix
from sklearn.utils.validation import check_is_fitted
from verde import get_region, median_distance, distance_mask
from verde.utils import kdtree
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

    def __init__(
        self,
        damping=None,
        points=None,
        depth_factor=3,
        k_nearest=1,
        distance_threshold=None,
    ):
        self.damping = damping
        self.points = points
        self.depth_factor = depth_factor
        self.k_nearest = k_nearest
        self.distance_threshold = distance_threshold

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
            # Put a single point source bellow each observation point at a depth
            # proportional to the median distance to the nearest k observation points.
            point_east, point_north, point_upward = tuple(i.copy() for i in coordinates)
            point_upward -= self.depth_factor * median_distance(
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
        if self.distance_threshold is None:
            jac = np.zeros((n_data, n_points), dtype=dtype)
            jacobian_numba(coordinates, points, jac)
        else:
            # Use cKDTree to get the indices of the observation points that are within
            # the distance threshold to the source points
            points_tree = kdtree(points, use_pykdtree=False)
            coords_tree = kdtree(coordinates, use_pykdtree=False)
            # Get the indices of the coordinates points that are close to each source
            # points
            col_indices = points_tree.query_ball_tree(
                coords_tree, self.distance_threshold
            )
            # Build the indices for the source points
            row_indices = tuple(
                np.full_like(indices, fill_value=i)
                for i, indices in enumerate(col_indices)
            )
            col_indices = tuple(np.atleast_1d(indices_i) for indices_i in col_indices)
            # Stack all the arrays inside the row and col indices
            row_indices, col_indices = np.hstack(row_indices), np.hstack(col_indices)
            # Compute the non-zero elements of the Jacobian matrix
            jac_values = np.zeros_like(row_indices)
            sparse_jacobian_elements(
                coordinates, points, col_indices, row_indices, jac_values
            )
            # Build the sparse Jacobian matrix
            jac = csr_matrix(
                (jac_values, (row_indices, col_indices)),
                shape=(n_data, n_points),
                dtype=dtype,
            )
        return jac


@jit(nopython=True)
def sparse_jacobian_elements(coordinates, points, col_indices, row_indices, jac_values):
    """
    Compute elements of the sparse Jacobian matrix
    """
    counter = 0
    east, north, upward = coordinates[:]
    point_east, point_north, point_upward = points[:]
    for i, j in zip(col_indices, row_indices):
        jac_values[counter] = greens_func(
            east[i], north[i], upward[i], point_east[j], point_north[j], point_upward[j]
        )
        counter += 1


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
