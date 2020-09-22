"""
Equivalent layer for generic harmonic functions in spherical coordinates
"""
from numba import jit

from .harmonic import EQLHarmonic
from ..forward.utils import distance_spherical


# EQLHarmonicSpherical is a subclass of EQLHarmonic.
# EQLHarmonicSpherical overrides some methods of its parent to rewrite
# docstrings according to spherical coordiantes. Besides, the projection
# argument has been removed from the grid and scatter methods because
# projections would only work on Cartesians gridders to get objects in geodetic
# coordinates. We will disable pylint arguments-differ error because we intend
# to make these methods different from the ones that are being inherited.


class EQLHarmonicSpherical(EQLHarmonic):
    r"""
    Equivalent-layer for generic harmonic functions in spherical coordinates

    This equivalent layer can be used for:

    * Spherical coordinates (geographic coordinates must be converted before
      use)
    * Regional or global data where Earth's curvature must be taken into
      account
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

    The Green's function for point mass effects used is the inverse Euclidean
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
        (``longitude``, ``latitude``, ``radius``). Both ``longitude`` and
        ``latitude`` must be in degrees and ``radius`` in meters.
        If None, will place one point source bellow each observation point at
        a fixed relative depth bellow the observation point [Cooper2000]_.
        Defaults to None.
    relative_depth : float
        Relative depth at which the point sources are placed beneath the
        observation points. Each source point will be set beneath each data
        point at a depth calculated as the radius of the data point minus
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
        :meth:`~harmonica.EQLHarmonicSpherical.grid` and
        :meth:`~harmonica.EQLHarmonicSpherical.scatter` methods.
    """

    # Set the default dimension names for generated outputs
    # (pd.DataFrame, xr.Dataset, etc)
    dims = ("spherical_latitude", "longitude")

    # Overwrite the defalt name for the upward coordinate.
    extra_coords_name = "radius"

    def __init__(
        self,
        damping=None,
        points=None,
        relative_depth=500,
    ):
        super().__init__(damping=damping, points=points, relative_depth=relative_depth)
        # Define Green's function for spherical coordinates
        self.greens_function = greens_func_spherical

    def fit(self, coordinates, data, weights=None):
        """
        Fit the coefficients of the equivalent layer.

        The data region is captured and used as default for the
        :meth:`~harmonica.EQLHarmonicSpherical.grid` and
        :meth:`~harmonica.EQLHarmonicSpherical.scatter` methods.

        All input arrays must have the same shape.

        Parameters
        ----------
        coordinates : tuple of arrays
            Arrays with the coordinates of each data point. Should be in the
            following order: (``longitude``, ``latitude``, ``radius``, ...).
            Only ``longitude``, ``latitude``, and ``radius`` will be used, all
            subsequent coordinates will be ignored.
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
        # Overwrite method just to change the docstring
        super().fit(coordinates, data, weights=weights)
        return self

    def predict(self, coordinates):
        """
        Evaluate the estimated equivalent layer on the given set of points.

        Requires a fitted estimator (see :meth:`~harmonica.EQLHarmonic.fit`).

        Parameters
        ----------
        coordinates : tuple of arrays
            Arrays with the coordinates of each data point. Should be in the
            following order: (``longitude``, ``latitude``, ``radius``, ...).
            Only ``longitude``, ``latitude`` and ``radius`` will be used, all
            subsequent coordinates will be ignored.

        Returns
        -------
        data : array
            The data values evaluated on the given points.
        """
        # Overwrite method just to change the docstring
        data = super().predict(coordinates)
        return data

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
            following order: (``longitude``, ``latitude``, ``radius``, ...).
            Only ``longitude``, ``latitude`` and ``radius`` will be used, all
            subsequent coordinates will be ignored.
        points : tuple of arrays
            Tuple of arrays containing the coordinates of the point sources
            used as equivalent layer in the following order:
            (``longitude``, ``latitude``, ``radius``).
        dtype : str or numpy dtype
            The type of the Jacobian array.

        Returns
        -------
        jacobian : 2D array
            The (n_data, n_points) Jacobian matrix.
        """
        # Overwrite method just to change the docstring
        jacobian = super().jacobian(coordinates, points, dtype=dtype)
        return jacobian

    def grid(
        self,
        upward,
        region=None,
        shape=None,
        spacing=None,
        dims=None,
        data_names=None,
        **kwargs
    ):  # pylint: disable=arguments-differ
        """
        Interpolate the data onto a regular grid.

        The grid can be specified by either the number of points in each
        dimension (the *shape*) or by the grid node spacing. See
        :func:`verde.grid_coordinates` for details. All grid points will be
        located at the same `upward` coordinate. Other arguments for
        :func:`verde.grid_coordinates` can be passed as extra keyword arguments
        (``kwargs``) to this method.

        If the interpolator collected the input data region, then it will be
        used if ``region=None``. Otherwise, you must specify the grid region.
        Use the *dims* and *data_names* arguments to set custom names for the
        dimensions and the data field(s) in the output :class:`xarray.Dataset`.
        Default names will be provided if none are given.

        Parameters
        ----------
        upward : float
            Upward coordinate of the grid points.
        region : list = [W, E, S, N]
            The west, east, south, and north boundaries of a given region.
        shape : tuple = (n_north, n_east) or None
            The number of points in the South-North and West-East directions,
            respectively.
        spacing : tuple = (s_north, s_east) or None
            The grid spacing in the South-North and West-East directions,
            respectively.
        dims : list or None
            The names of the northing and easting data dimensions,
            respectively, in the output grid. Default is determined from the
            ``dims`` attribute of the class. Must be defined in the following
            order: northing dimension, easting dimension.
            **NOTE: This is an exception to the "easting" then
            "northing" pattern but is required for compatibility with xarray.**
        data_names : list of None
            The name(s) of the data variables in the output grid. Defaults to
            ``['scalars']``.

        Returns
        -------
        grid : xarray.Dataset
            The interpolated grid. Metadata about the interpolator is written
            to the ``attrs`` attribute.

        """
        # Overwrite method to ditch projection argument
        grid = super().grid(
            upward=upward,
            region=region,
            shape=shape,
            spacing=spacing,
            dims=dims,
            data_names=data_names,
            projection=None,
            **kwargs,
        )
        return grid

    def scatter(
        self,
        upward,
        region=None,
        size=300,
        random_state=0,
        dims=None,
        data_names=None,
        **kwargs
    ):  # pylint: disable=arguments-differ
        """
        Interpolate values onto a random scatter of points.

        Point coordinates are generated by :func:`verde.scatter_points`. Other
        arguments for this function can be passed as extra keyword arguments
        (``kwargs``) to this method.

        If the interpolator collected the input data region, then it will be
        used if ``region=None``. Otherwise, you must specify the grid region.

        Use the *dims* and *data_names* arguments to set custom names for the
        dimensions and the data field(s) in the output
        :class:`pandas.DataFrame`. Default names are provided.

        Parameters
        ----------
        upward: float
            Upward coordinate of the grid points.
        region : list = [W, E, S, N]
            The west, east, south, and north boundaries of a given region.
        size : int
            The number of points to generate.
        random_state : numpy.random.RandomState or an int seed
            A random number generator used to define the state of the random
            permutations. Use a fixed seed to make sure computations are
            reproducible. Use ``None`` to choose a seed automatically
            (resulting in different numbers with each run).
        dims : list or None
            The names of the northing and easting data dimensions,
            respectively, in the output dataframe. Default is determined from
            the ``dims`` attribute of the class. Must be defined in the
            following order: northing dimension, easting dimension.
            **NOTE: This is an exception to the "easting" then
            "northing" pattern but is required for compatibility with xarray.**
        data_names : list of None
            The name(s) of the data variables in the output dataframe. Defaults
            to ``['scalars']``.

        Returns
        -------
        table : pandas.DataFrame
            The interpolated values on a random set of points.

        """
        # Overwrite method to ditch projection argument
        table = super().scatter(
            upward=upward,
            region=region,
            size=size,
            random_state=random_state,
            dims=dims,
            data_names=data_names,
            projection=None,
            **kwargs,
        )
        return table

    def profile(
        self,
        point1,
        point2,
        upward,
        size,
        dims=None,
        data_names=None,
        projection=None,
        **kwargs
    ):
        """
        .. warning ::

            Not implemented method. The profile on spherical coordinates should
            be done using great-circle distances through the Haversine formula.

        """
        raise NotImplementedError


@jit(nopython=True)
def greens_func_spherical(
    longitude, latitude, radius, point_longitude, point_latitude, point_radius
):
    """
    Green's function for the equivalent layer in spherical coordinates

    Uses Numba to speed up things.
    """
    distance = distance_spherical(
        (longitude, latitude, radius), (point_longitude, point_latitude, point_radius)
    )
    return 1 / distance
