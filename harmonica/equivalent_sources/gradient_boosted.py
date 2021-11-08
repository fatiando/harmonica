# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Gradient-boosted equivalent sources in Cartesian coordinates
"""
import numpy as np
from sklearn.utils import shuffle
from verde import get_region, rolling_window
import verde.base as vdb

from .cartesian import EquivalentSources
from .utils import (
    predict_numba_parallel,
)


class EquivalentSourcesGB(EquivalentSources):
    r"""
    Gradient-boosted equivalent sources for generic harmonic functions.

    Gradient-boosted version of the :class:`harmonica.EquivalentSources`,
    introduced in [Soler2021]_. These equivalent sources are intended to be
    used to fit very large datasets, where the Jacobian matrices generated by
    regular equivalent sources (like :class:`harmonica.EquivalentSources`) are
    larger than the available memory. They fit the sources coefficients
    iteratively using overlapping windows of equal size, greatly reducing the
    memory requirements.

    Smaller windows lower the memory requirements. Using very small windows may
    impact the accuracy of the interpolations. We recommend using the larger
    windows that generate Jacobian matrices that fit in the available memory.

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
        If ``depth_type`` is ``"constant"``, each source is located at the same
        depth specified through the ``depth`` argument.
        If ``depth_type`` is ``"relative"``, each source is located beneath
        each data point (or block-averaged location) at a depth equal to its
        elevation minus the ``depth`` value.
        This parameter is ignored if *points* is specified.
        Defaults to 500.
    depth_type : str
        Strategy used for setting the depth of the point sources.
        The two available strategies are ``"constant"`` and ``"relative"``.
        This parameter is ignored if *points* is specified.
        Defaults to ``"relative"``.
    block_size: float, tuple = (s_north, s_east) or None
        Size of the blocks used on block-averaged equivalent sources.
        If a single value is passed, the blocks will have a square shape.
        Alternatively, the dimensions of the blocks in the South-North and
        West-East directions can be specified by passing a tuple.
        If None, no block-averaging is applied.
        This parameter is ignored if *points* are specified.
        Default to None.
    window_size : float
        Size of overlapping windows used during the gradient-boosting
        algorithm. Smaller windows reduce the memory requirements of the
        source coefficients fitting process. Very small windows may impact on
        Defaults to 5000.
        the accuracy of the interpolations.
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

    References
    ----------
    [Soler2021]_
    """

    # Define amount of overlapping between adjacent windows to 50%.
    overlapping = 0.5

    def __init__(
        self,
        damping=None,
        points=None,
        depth=500,
        depth_type="relative",
        block_size=None,
        window_size=5e3,
        parallel=True,
        random_state=None,
    ):
        super().__init__(
            damping=damping,
            points=points,
            depth=depth,
            depth_type=depth_type,
            block_size=block_size,
            parallel=parallel,
        )
        self.random_state = random_state
        self.window_size = window_size

    def fit(self, coordinates, data, weights=None):
        """
        Fit the coefficients of the equivalent sources.

        The fitting process is carried out through the gradient-boosting
        algorithm.
        The data region is captured and used as default for the
        :meth:`~harmonica.EquivalentSourcesGB.grid` method.

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
        self.region_ = get_region(coordinates[:2])
        # Ravel coordinates, data and weights to 1d-arrays
        coordinates = vdb.n_1d_arrays(coordinates, 3)
        data = data.ravel()
        if weights is not None:
            weights = weights.ravel()
        # Build point sources
        if self.points is None:
            self.points_ = self._build_points(coordinates)
        else:
            self.points_ = vdb.n_1d_arrays(self.points, 3)
        # Initialize coefficients
        self.coefs_ = np.zeros_like(self.points_[0])
        # Fit coefficients through gradient boosting
        self._gradient_boosting(coordinates, data, weights)
        return self

    def _gradient_boosting(self, coordinates, data, weights):
        """
        Fit source coefficients through gradient boosting
        """
        # Create rolling windows
        point_windows, data_windows = self._create_windows(coordinates)
        # Get number of windows
        n_windows = len(point_windows)
        # Initialize RMSE array
        errors = [np.sqrt(np.mean(data ** 2))]
        # Set weights_chunk to None (will be changed unless weights is None)
        weights_chunk = None
        # Initialized the predicted and residue arrays
        predicted = np.empty_like(data)
        residue = data.copy()
        # Iterate over the windows
        for window in range(n_windows):
            # Get source and data points indices for current window
            point_window, data_window = point_windows[window], data_windows[window]
            # Choose source and data points that fall inside the window
            points_chunk = tuple(p[point_window] for p in self.points_)
            coords_chunk = tuple(c[data_window] for c in coordinates)
            # Choose weights for data points inside the window (if not None)
            if weights is not None:
                weights_chunk = weights[data_window]
            # Compute Jacobian (for sources and data points in current window)
            jacobian = self.jacobian(coords_chunk, points_chunk)
            # Fit coefficients of sources with residue points inside window
            coeffs_chunk = vdb.least_squares(
                jacobian,
                residue[data_window],
                weights_chunk,
                self.damping,
            )
            # Predict field of the sources in the window on every data point
            predicted[:] = 0
            predict_numba_parallel(
                coordinates,
                points_chunk,
                coeffs_chunk,
                predicted,
                self.greens_function,
            )
            # Update the residue
            residue -= predicted
            # Add RMS of the residue to the RMSE
            errors.append(np.sqrt(np.mean(residue ** 2)))
            # Update source coefficients
            self.coefs_[point_window] += coeffs_chunk
        self.errors_ = np.array(errors)

    def _create_windows(self, coordinates, shuffle_windows=True):
        """
        Create indices of sources and data points for each overlapping window

        Parameters
        ----------
        coordinates : tuple
            Arrays with the coordinates of each data point. Should be in the
            following order: (``easting``, ``northing``, ``upward``).
        shuffle_windows : bool
            Enable or disable the random shuffling of windows order. It's is
            highly recommended to enable shuffling for better fitting results.
            This argument is mainly included for testing purposes. Default to
            True.

        Returns
        -------
        source_windows_nonempty : list
            List containing arrays with the indices of the sources that fall
            under each window. The order of the windows is randomly shuffled if
            ``shuffle_windows`` is True, although the order of the windows is
            the same as the one in ``data_windows_nonempty``.
        data_windows_nonempty : list
            List containing arrays with the indices of the data points that
            under each window. The order of the windows is randomly shuffled if
            ``shuffle_windows`` is True, although the order of the windows is
            the same as the one in ``source_windows_nonempty``.
        """
        # Compute window spacing based on overlapping
        window_spacing = self.window_size * (1 - self.overlapping)
        # Get the region that contains every data point and every source
        region = _get_region_data_sources(coordinates, self.points_)
        # The windows for sources and data points are the same, but the
        # verde.rolling_window function creates indices for the given
        # coordinates. That's why we need to create two set of window indices:
        # one for the sources and one for the data points.
        # We pass the same region, size and spacing to be sure that both set of
        # windows are the same.
        kwargs = dict(region=region, size=self.window_size, spacing=window_spacing)
        _, source_windows = rolling_window(self.points_, **kwargs)
        _, data_windows = rolling_window(coordinates, **kwargs)
        # Ravel the indices
        source_windows = [i[0] for i in source_windows.ravel()]
        data_windows = [i[0] for i in data_windows.ravel()]
        # Shuffle windows
        if shuffle_windows:
            source_windows, data_windows = shuffle(
                source_windows, data_windows, random_state=self.random_state
            )
        # Remove empty windows
        source_windows_nonempty = []
        data_windows_nonempty = []
        for src, data in zip(source_windows, data_windows):
            if src.size > 0 and data.size > 0:
                source_windows_nonempty.append(src)
                data_windows_nonempty.append(data)
        return source_windows_nonempty, data_windows_nonempty


def _get_region_data_sources(coordinates, points):
    """
    Return the region that contains every observation and every source

    Parameters
    ----------
    coordinates : tuple
    points : tuple

    Returns
    -------
    region : tuple
    """
    data_region = get_region(coordinates)
    sources_region = get_region(points)
    region = (
        min(data_region[0], sources_region[0]),
        max(data_region[1], sources_region[1]),
        min(data_region[2], sources_region[2]),
        max(data_region[3], sources_region[3]),
    )
    return region
