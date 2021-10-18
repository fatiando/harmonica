# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Equivalent sources for generic harmonic functions in Cartesian coordinates
"""
import warnings
import numpy as np
from numba import jit
from sklearn.utils.validation import check_is_fitted
import verde as vd
import verde.base as vdb

from .utils import (
    pop_extra_coords,
    predict_numba_serial,
    predict_numba_parallel,
    jacobian_numba_serial,
    jacobian_numba_parallel,
)
from ..forward.utils import distance_cartesian


class EquivalentSources(vdb.BaseGridder):
    r"""
    Equivalent sources for generic harmonic functions (gravity, magnetics).

    These equivalent sources can be used for:

    * Cartesian coordinates (geographic coordinates must be project before use)
    * Gravity and magnetic data (including derivatives)
    * Single data types
    * Interpolation
    * Upward continuation
    * Finite-difference based derivative calculations

    They cannot be used for:

    * Regional or global data where Earth's curvature must be taken into
      account
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
        List containing the coordinates of the equivalent point sources.
        Coordinates are assumed to be in the following order:
        (``easting``, ``northing``, ``upward``).
        If None, will place one point source below each observation point at
        a fixed relative depth below the observation point [Cooper2000]_.
        Defaults to None.
    depth : float
        Parameter used to control the depth at which the point sources will be
        located.
        If ``depth_type`` is equal to ``"relative"``, the ``depth`` specifies
        the relative depth at which the point sources are placed beneath the
        observation points. Each source point will be set beneath each data
        point at a depth calculated as the elevation of the data point minus
        this *depth*. Use positive numbers (negative numbers would mean point
        sources are above the data points).
        If ``depth_type`` is equal to ``"constant"``, the ``depth`` specifies
        the constant depth at which the point sources are placed beneath the
        observation points. Every source point will be located at this *depth*.
        Use positive numbers (negative numbers would mean point sources are
        located above the zeroth level).
        This parameter is ignored if *points* is specified.
        Defaults to 500.
    depth_type : str
        Strategy used for setting the depth of the point sources.
        The two available strategies are ``"constant"`` and ``"relative"``.
        This parameter is ignored if *points* is specified.
        Defaults to ``"relative"``.
    parallel : bool
        If True any predictions and Jacobian building is carried out in
        parallel through Numba's ``jit.prange``, reducing the computation time.
        If False, these tasks will be run on a single CPU. Default to True.

    Attributes
    ----------
    points_ : 2d-array
        Coordinates of the equivalent point sources.
    coefs_ : array
        Estimated coefficients of every point source.
    region_ : tuple
        The boundaries (``[W, E, S, N]``) of the data used to fit the
        interpolator. Used as the default region for the
        :meth:`~harmonica.EQLHarmonic.grid` method.
    """

    # Set the default dimension names for generated outputs
    # as xr.Dataset.
    dims = ("northing", "easting")

    # Overwrite the defalt name for the upward coordinate.
    extra_coords_name = "upward"

    # Define dispatcher for Numba functions with or without parallelization
    _predict_kernel = {False: predict_numba_serial, True: predict_numba_parallel}
    _jacobian_kernel = {False: jacobian_numba_serial, True: jacobian_numba_parallel}

    def __init__(
        self,
        damping=None,
        points=None,
        depth=500,
        depth_type="relative",
        parallel=True,
        **kwargs,
    ):
        self.damping = damping
        self.points = points
        self.depth = depth
        self.depth_type = depth_type
        self.parallel = parallel
        # Define Green's function for Cartesian coordinates
        self.greens_function = greens_func_cartesian
        # Check if depth_type is valid
        if depth_type not in ("constant", "relative"):
            raise ValueError(
                f"Invalid depth type '{depth_type}'. Should be either be 'constant' or 'relative'."
            )
        # Check if relative_depth has been passed (will be deprecated)
        if "relative_depth" in kwargs:
            warnings.warn(
                "The 'relative_depth' parameter is deprecated, please use "
                + "the 'depth' paramter and set 'depth_type' to 'relative_depth' instead. ",
                FutureWarning,
            )
            # Override depth and depth_type
            self.depth, self.depth_type = kwargs["relative_depth"], "relative"

    def fit(self, coordinates, data, weights=None):
        """
        Fit the coefficients of the equivalent sources.

        The data region is captured and used as default for the
        :meth:`~harmonica.EQLHarmonic.grid` method.

        All input arrays must have the same shape.

        Parameters
        ----------
        coordinates : tuple of arrays
            Arrays with the coordinates of each data point. Should be in the
            following order: (``easting``, ``northing``, ``upward``, ...).
            Only ``easting``, ``northing``, and ``upward`` will be used, all
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
        coordinates, data, weights = vdb.check_fit_input(coordinates, data, weights)
        # Capture the data region to use as a default when gridding.
        self.region_ = vd.get_region(coordinates[:2])
        coordinates = vdb.n_1d_arrays(coordinates, 3)
        if self.points is None:
            self.points_ = self._build_points(coordinates)
        else:
            self.points_ = vdb.n_1d_arrays(self.points, 3)
        jacobian = self.jacobian(coordinates, self.points_)
        self.coefs_ = vdb.least_squares(jacobian, data, weights, self.damping)
        return self

    def _build_points(self, coordinates):
        """
        Generate coordinates of point sources based on the data points

        Locate the point sources following the chosen ``depth_type`` strategy.
        If ``depth_type`` is equal to ``"relative"``, the point sources will be
        placed beneath the observation points at a depth calculated as the
        elevation of the data point minus the ``depth``.
        If ``depth_type`` is equal to ``"constant"``, the point sources will be
        placed beneath the observation points at the same height equal to minus
        ``depth``.

        Parameters
        ----------
        coordinates : tuple of arrays
            Arrays with the coordinates of each data point. Should be in the
            following order: (``easting``, ``northing``, ``upward``, ...).
            Only ``easting``, ``northing``, and ``upward`` will be used, all
            subsequent coordinates will be ignored.

        Returns
        -------
        points : tuple of arrays
            Tuple containing the coordinates of the equivalent point sources,
            in the following order: (``easting``, ``northing``, ``upward``).
        """
        if self.depth_type == "relative":
            return (
                coordinates[0],
                coordinates[1],
                coordinates[2] - self.depth,
            )
        if self.depth_type == "constant":
            return (
                coordinates[0],
                coordinates[1],
                -self.depth * np.ones_like(coordinates[0]),
            )
        return None

    def predict(self, coordinates):
        """
        Evaluate the estimated equivalent sources on the given set of points.

        Requires a fitted estimator (see :meth:`~harmonica.EQLHarmonic.fit`).

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
        self._predict_kernel[self.parallel](
            coordinates, self.points_, self.coefs_, data, self.greens_function
        )
        return data.reshape(shape)

    def jacobian(
        self, coordinates, points, dtype="float64"
    ):  # pylint: disable=no-self-use
        """
        Make the Jacobian matrix for the equivalent sources.

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
            Tuple of arrays containing the coordinates of the equivalent point
            sources in the following order:
            (``easting``, ``northing``, ``upward``).
        dtype : str or numpy dtype
            The type of the Jacobian array.

        Returns
        -------
        jacobian : 2D array
            The (n_data, n_points) Jacobian matrix.
        """
        # Compute Jacobian matrix
        n_data = coordinates[0].size
        n_points = points[0].size
        jac = np.zeros((n_data, n_points), dtype=dtype)
        self._jacobian_kernel[self.parallel](
            coordinates, points, jac, self.greens_function
        )
        return jac

    def grid(
        self,
        upward,
        region=None,
        shape=None,
        spacing=None,
        dims=None,
        data_names=None,
        projection=None,
        **kwargs,
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
        projection : callable or None
            If not None, then should be a callable object
            ``projection(easting, northing) -> (proj_easting, proj_northing)``
            that takes in easting and northing coordinate arrays and returns
            projected northing and easting coordinate arrays. This function
            will be used to project the generated grid coordinates before
            passing them into ``predict``. For example, you can use this to
            generate a geographic grid from a Cartesian gridder.

        Returns
        -------
        grid : xarray.Dataset
            The interpolated grid. Metadata about the interpolator is written
            to the ``attrs`` attribute.

        """
        # We override the grid method from BaseGridder so it takes the upward
        # coordinate as a positional argument. We disable pylint
        # arguments-differ error because we intend to make this method
        # different from the inherited one.

        # Ignore extra_coords if passed
        pop_extra_coords(kwargs)
        # Grid data
        grid = super().grid(
            region=region,
            shape=shape,
            spacing=spacing,
            dims=dims,
            data_names=data_names,
            projection=projection,
            extra_coords=upward,
            **kwargs,
        )
        return grid

    def scatter(
        self,
        region=None,
        size=300,
        random_state=0,
        dims=None,
        data_names=None,
        projection=None,
        **kwargs,
    ):
        """
        .. warning ::

            Not implemented method. The scatter method will be deprecated on
            Verde v2.0.0.

        """
        raise NotImplementedError

    def profile(
        self,
        point1,
        point2,
        upward,
        size,
        dims=None,
        data_names=None,
        projection=None,
        **kwargs,
    ):  # pylint: disable=arguments-differ
        """
        Interpolate data along a profile between two points.

        Generates the profile along a straight line assuming Cartesian
        distances and the same upward coordinate for all points. Point
        coordinates are generated by :func:`verde.profile_coordinates`. Other
        arguments for this function can be passed as extra keyword arguments
        (``kwargs``) to this method.

        Use the *dims* and *data_names* arguments to set custom names for the
        dimensions and the data field(s) in the output
        :class:`pandas.DataFrame`. Default names are provided.

        Includes the calculated Cartesian distance from *point1* for each data
        point in the profile.

        To specify *point1* and *point2* in a coordinate system that would
        require projection to Cartesian (geographic longitude and latitude, for
        example), use the ``projection`` argument. With this option, the input
        points will be projected using the given projection function prior to
        computations. The generated Cartesian profile coordinates will be
        projected back to the original coordinate system. **Note that the
        profile points are evenly spaced in projected coordinates, not the
        original system (e.g., geographic)**.

        Parameters
        ----------
        point1 : tuple
            The easting and northing coordinates, respectively, of the first
            point.
        point2 : tuple
            The easting and northing coordinates, respectively, of the second
            point.
        upward : float
            Upward coordinate of the profile points.
        size : int
            The number of points to generate.
        dims : list or None
            The names of the northing and easting data dimensions,
            respectively, in the output dataframe. Default is determined from
            the ``dims`` attribute of the class. Must be defined in the
            following order: northing dimension, easting dimension.
            **NOTE: This is an exception to the "easting" then
            "northing" pattern but is required for compatibility with xarray.**
        data_names : list of None
            The name(s) of the data variables in the output dataframe. Defaults
            to ``['scalars']`` for scalar data,
            ``['east_component', 'north_component']`` for 2D vector data, and
            ``['east_component', 'north_component', 'vertical_component']`` for
            3D vector data.
        projection : callable or None
            If not None, then should be a callable object ``projection(easting,
            northing, inverse=False) -> (proj_easting, proj_northing)`` that
            takes in easting and northing coordinate arrays and returns
            projected northing and easting coordinate arrays. Should also take
            an optional keyword argument ``inverse`` (default to False) that if
            True will calculate the inverse transform instead. This function
            will be used to project the profile end points before generating
            coordinates and passing them into ``predict``. It will also be used
            to undo the projection of the coordinates before returning the
            results.

        Returns
        -------
        table : pandas.DataFrame
            The interpolated values along the profile.

        """
        # We override the profile method from BaseGridder so it takes the
        # upward coordinate as a positional argument. We disable pylint
        # arguments-differ error because we intend to make this method
        # different from the inherited one.

        # Ignore extra_coords if passed
        pop_extra_coords(kwargs)
        # Create profile points and predict
        table = super().profile(
            point1,
            point2,
            size,
            dims=dims,
            data_names=data_names,
            projection=projection,
            extra_coords=upward,
            **kwargs,
        )
        return table


class EQLHarmonic(EquivalentSources):
    """
    DEPRECATED, use ``harmonica.EquivalentSources`` instead.

    This class exists to support backward compatibility until next release.
    """

    def __init__(
        self,
        damping=None,
        points=None,
        depth=500,
        depth_type="relative",
        parallel=True,
        **kwargs,
    ):
        warnings.warn(
            "The 'EQLHarmonic' class has been renamed to 'EquivalentSources' "
            + "and will be deprecated on the next release, "
            + "please use 'EquivalentSources' instead.",
            FutureWarning,
        )
        super().__init__(
            damping=damping,
            points=points,
            depth=depth,
            depth_type=depth_type,
            parallel=parallel,
            **kwargs,
        )


@jit(nopython=True)
def greens_func_cartesian(east, north, upward, point_east, point_north, point_upward):
    """
    Green's function for the equivalent sources in Cartesian coordinates

    Uses Numba to speed up things.
    """
    distance = distance_cartesian(
        (east, north, upward), (point_east, point_north, point_upward)
    )
    return 1 / distance
