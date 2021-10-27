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
    r""" """

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
            block_size=None,
            parallel=parallel,
        )
        self.random_state = random_state
        self.window_size = window_size

    def fit(self, coordinates, data, weights=None):
        """
        Fit the coefficients of the equivalent sources.

        The fitting process is carried out through the gradient-boosting
        algoritm.
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
        shuffle_windows : bool

        Returns
        -------
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
