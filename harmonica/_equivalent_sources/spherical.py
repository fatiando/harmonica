# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Equivalent sources for generic harmonic functions in spherical coordinates
"""
import warnings

import numpy as np
import verde as vd
import verde.base as vdb
from numba import jit
from sklearn.utils.validation import check_is_fitted

from .._forward.utils import distance_spherical
from .utils import (
    jacobian_numba_parallel,
    jacobian_numba_serial,
    predict_numba_parallel,
    predict_numba_serial,
)


class EquivalentSourcesSph(vdb.BaseGridder):
    r"""
    Equivalent sources for generic harmonic functions in spherical coordinates

    These equivalent sources can be used for:

    * Spherical coordinates (geographic coordinates must be converted before
      use)
    * Regional or global data where Earth's curvature must be taken into
      account
    * Gravity and magnetic data (including derivatives)
    * Single data types
    * Interpolation
    * Upward continuation
    * Finite-difference based derivative calculations

    They cannot be used for:

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
        (``longitude``, ``latitude``, ``radius``). Both ``longitude`` and
        ``latitude`` must be in degrees and ``radius`` in meters.
        If None, will place one point source below each observation point at
        a fixed relative depth below the observation point [Cooper2000]_.
        Defaults to None.
    relative_depth : float
        Relative depth at which the point sources are placed beneath the
        observation points. Each source point will be set beneath each data
        point at a depth calculated as the radius of the data point minus
        this constant *relative_depth*. Use positive numbers (negative numbers
        would mean point sources are above the data points). Ignored if
        *points* is specified.
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
        :meth:`~harmonica.EquivalentSources.grid` method.
    """

    # Set the default dimension names for generated outputs
    # as xr.Dataset.
    dims = ("spherical_latitude", "longitude")

    # Overwrite the defalt name for the upward coordinate.
    extra_coords_name = "radius"

    # Define dispatcher for Numba functions with or without parallelization
    _predict_kernel = {False: predict_numba_serial, True: predict_numba_parallel}
    _jacobian_kernel = {False: jacobian_numba_serial, True: jacobian_numba_parallel}

    def __init__(
        self,
        damping=None,
        points=None,
        relative_depth=500,
        parallel=True,
    ):
        self.damping = damping
        self.points = points
        self.relative_depth = relative_depth
        self.parallel = parallel
        # Define Green's function for spherical coordinates
        self.greens_function = greens_func_spherical

    def fit(self, coordinates, data, weights=None):
        """
        Fit the coefficients of the equivalent sources.

        The data region is captured and used as default for the
        :meth:`~harmonica.EquivalentSourcesSph.grid` method.

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
        Evaluate the estimated equivalent sources on the given set of points.

        Requires a fitted estimator
        (see :meth:`~harmonica.EquivalentSourcesSph.fit`).

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

    def jacobian(self, coordinates, points, dtype="float64"):
        """
        Make the Jacobian matrix for the equivalent sources.

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
            Tuple of arrays containing the coordinates of the equivalent point
            sources in the following order:
            (``longitude``, ``latitude``, ``radius``).
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
        coordinates,
        dims=None,
        data_names=None,
        **kwargs,
    ):
        """
        Interpolate the data onto a regular grid.

        The coordinates of the regular grid must be passed through the
        ``coordinates`` argument as a tuple containing three arrays in the
        following order: ``(longitude, latitude, radius)``. They can be easily
        created through the :func:`verde.grid_coordinates` function. If the
        grid points must be all at the same radius, it can be specified in the
        ``extra_coords`` argument of :func:`verde.grid_coordinates`.

        Use the *dims* and *data_names* arguments to set custom names for the
        dimensions and the data field(s) in the output :class:`xarray.Dataset`.
        Default names will be provided if none are given.

        Parameters
        ----------
        coordinates : tuple of arrays
            Tuple of arrays containing the coordinates of the grid in the
            following order: (longitude, latitude, radius).
            The longitude and latitude arrays could be 1d or 2d arrays, if
            they are 2d they must be part of a meshgrid.
            The radius array should be a 2d array with the same shape of
            longitude and latitude (if they are 2d arrays) or with a shape of
            ``(latitude.size, longitude.size)`` (if they are 1d arrays).
        dims : list or None
            The names of the latitude and longitude data dimensions,
            respectively, in the output grid. Default is determined from the
            ``dims`` attribute of the class. Must be defined in the following
            order: latitude dimension, longitude dimension.
            **NOTE: This is an exception to the "longitude" then
            "latitude" pattern but is required for compatibility with xarray.**
        data_names : list of None
            The name(s) of the data variables in the output grid. Defaults to
            ``['scalars']``.

        Returns
        -------
        grid : xarray.Dataset
            The interpolated grid. Metadata about the interpolator is written
            to the ``attrs`` attribute.

        """
        # We override the grid method from BaseGridder to change the docstring
        # and to make it work only with the `coordinates` argument (no region,
        # shape or spacing)

        # Raise ValueError if any deprecated argument has been passed
        deprecated_args = (
            "upward" in kwargs,
            "shape" in kwargs,
            "region" in kwargs,
            "spacing" in kwargs,
        )
        if any(deprecated_args):
            raise ValueError(
                "The 'upward', 'region', 'shape' and 'spacing' arguments have been "
                + "deprecated. "
                + "Please pass the coordinates of the target grid through the "
                + "'coordinates' argument."
            )

        # Raise warning if any kwargs has been passed
        if kwargs:
            args = "'" + "', '".join(list(kwargs.keys())) + "'"
            warnings.warn(
                f"The {args} arguments are being ignored. The 'grid' method "
                + "will not take any keyword arguments in the next Harmonica release",
                FutureWarning,
            )

        # Grid data
        # We always pass projection=None because that argument it's intended to
        # be used only with Cartesian gridders.
        grid = super().grid(
            coordinates=coordinates,
            dims=dims,
            data_names=data_names,
            projection=None,
        )
        return grid

    def scatter(
        self,
        region=None,  # noqa: U100
        size=None,  # noqa: U100
        random_state=None,  # noqa: U100
        dims=None,  # noqa: U100
        data_names=None,  # noqa: U100
        projection=None,  # noqa: U100
        **kwargs,  # noqa: U100
    ):
        """
        .. warning ::

            Not implemented method. The scatter method will be deprecated on
            Verde v2.0.0.

        """
        raise NotImplementedError

    def profile(
        self,
        point1,  # noqa: U100
        point2,  # noqa: U100
        size,  # noqa: U100
        dims=None,  # noqa: U100
        data_names=None,  # noqa: U100
        projection=None,  # noqa: U100
        **kwargs,  # noqa: U100
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
    Green's function for the equivalent sources in spherical coordinates

    Uses Numba to speed up things.
    """
    distance = distance_spherical(
        (longitude, latitude, radius), (point_longitude, point_latitude, point_radius)
    )
    return 1 / distance
