# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Gradient-boosted equivalent sources in Cartesian coordinates
"""
import warnings

import numpy as np
import verde.base as vdb
from sklearn import utils
from verde import get_region, rolling_window

from .cartesian import EquivalentSources
from .utils import cast_fit_input, predict_numba_parallel


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
        Each source is located beneath each data point (or block-averaged
        location) at a depth equal to its elevation minus the ``depth`` value.
        This parameter is ignored if *points* is specified.
        Defaults to 500.
    block_size: float, tuple = (s_north, s_east) or None
        Size of the blocks used on block-averaged equivalent sources.
        If a single value is passed, the blocks will have a square shape.
        Alternatively, the dimensions of the blocks in the South-North and
        West-East directions can be specified by passing a tuple.
        If None, no block-averaging is applied.
        This parameter is ignored if *points* are specified.
        Default to None.
    window_size : float or "default"
        Size of overlapping windows used during the gradient-boosting
        algorithm. Smaller windows reduce the memory requirements of the source
        coefficients fitting process. Very small windows may impact on the
        accuracy of the interpolations.
        Defaults to estimating a window size such that approximately 5000 data
        points are in each window.
    parallel : bool
        If True any predictions and Jacobian building is carried out in
        parallel through Numba's ``jit.prange``, reducing the computation time.
        If False, these tasks will be run on a single CPU. Default to True.
    dtype : data-type
        The desired data-type for the predictions and the Jacobian matrix.
        Default to ``"float64"``.

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
    window_size_ : float or None
        Size of the overlapping windows used in gradient-boosting equivalent
        point sources. It will be set to None if ``window_size = "default"``
        and less than 5000 data points were used to fit the sources; a single
        window will be used in such case.

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
        block_size=None,
        window_size="default",
        parallel=True,
        random_state=None,
        dtype="float64",
    ):
        if isinstance(window_size, str) and window_size != "default":
            raise ValueError(
                f"Found invalid 'window_size' value equal to '{window_size}'."
                "It should be 'default' or a numeric value."
            )
        super().__init__(
            damping=damping,
            points=points,
            depth=depth,
            block_size=block_size,
            parallel=parallel,
            dtype=dtype,
        )
        self.random_state = random_state
        self.window_size = window_size

    def estimate_required_memory(self, coordinates):
        """
        Estimate the memory required for storing the largest Jacobian matrix

        Parameters
        ----------
        coordinates : tuple of arrays
            Arrays with the coordinates of each data point. Should be in the
            following order: (``easting``, ``northing``, ``upward``, ...).
            Only ``easting``, ``northing``, and ``upward`` will be used, all
            subsequent coordinates will be ignored.

        Returns
        -------
        memory_required : float
            Amount of memory required to store the largest Jacobian matrix in
            bytes.

        Examples
        --------

        >>> import verde as vd
        >>> coordinates = vd.scatter_points(
        ...     region=(-1e3, 3e3, 2e3, 5e3),
        ...     size=100,
        ...     extra_coords=100,
        ...     random_state=42,
        ... )
        >>> eqs = EquivalentSourcesGB(window_size=2e3)
        >>> eqs.estimate_required_memory(coordinates)
        9800
        """
        # Build the sources and assign the points_ attribute
        coordinates = vdb.n_1d_arrays(coordinates, 3)
        points = self._build_points(coordinates)
        self.points_ = points
        # Build the windows and get the indices
        source_windows, data_windows = self._create_windows(coordinates)
        # Get the number of sources and data for each window
        source_sizes = np.array([w.size for w in source_windows])
        data_sizes = np.array([w.size for w in data_windows])
        # Compute the size of the Jacobian matrix for each window
        jacobian_sizes = source_sizes * data_sizes
        # Estimate size of a single element of the Jacobian matrix in bytes
        return jacobian_sizes.max() * np.dtype(self.dtype).itemsize

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
        coordinates, data, weights = cast_fit_input(
            coordinates, data, weights, self.dtype
        )
        # Capture the data region to use as a default when gridding.
        self.region_ = get_region(coordinates[:2])
        # Ravel coordinates, data and weights to 1d-arrays
        coordinates = vdb.n_1d_arrays(coordinates, 3)
        data = data.ravel()
        if weights is not None:
            weights = weights.ravel()
        # Build point sources
        if self.points is None:
            self.points_ = tuple(
                p.astype(self.dtype) for p in self._build_points(coordinates)
            )
        else:
            self.points_ = tuple(
                p.astype(self.dtype) for p in vdb.n_1d_arrays(self.points, 3)
            )
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
        errors = [np.sqrt(np.mean(data**2))]
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
            errors.append(np.sqrt(np.mean(residue**2)))
            # Update source coefficients
            self.coefs_[point_window] += coeffs_chunk
        self.rmse_per_iteration_ = np.array(errors)

    def _create_windows(self, coordinates, shuffle=True):
        """
        Create indices of sources and data points for each overlapping window

        Parameters
        ----------
        coordinates : tuple
            Arrays with the coordinates of each data point. Should be in the
            following order: (``easting``, ``northing``, ``upward``).
        shuffle : bool
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
            fall under each window. The order of the windows is randomly
            shuffled if ``shuffle_windows`` is True, although the order of the
            windows is the same as the one in ``source_windows_nonempty``.
        """

        # Get the region that contains every data point and every source
        region = _get_region_data_sources(coordinates, self.points_)
        # Calculate the window size such that there are approximately 5000 data
        # points in each window. Otherwise use the given window size.
        if self.window_size == "default":
            area = (region[1] - region[0]) * (region[3] - region[2])
            ndata = coordinates[0].size
            if ndata <= 5e3:
                warnings.warn(
                    f"Found {ndata} number of coordinates (<= 5e3). "
                    "Only one window will be used.",
                    stacklevel=1,
                )
                source_windows_nonempty = [np.arange(self.points_[0].size)]
                data_windows_nonempty = [np.arange(ndata)]
                self.window_size_ = None
                return source_windows_nonempty, data_windows_nonempty
            points_per_m2 = ndata / area
            window_area = 5e3 / points_per_m2
            self.window_size_ = np.sqrt(window_area)
        else:
            self.window_size_ = self.window_size
        # Compute window spacing based on overlapping
        window_spacing = self.window_size_ * (1 - self.overlapping)
        # The windows for sources and data points are the same, but the
        # verde.rolling_window function creates indices for the given
        # coordinates. That's why we need to create two set of window indices:
        # one for the sources and one for the data points.
        # We pass the same region, size and spacing to be sure that both set of
        # windows are the same.
        kwargs = dict(region=region, size=self.window_size_, spacing=window_spacing)
        _, source_windows = rolling_window(self.points_, **kwargs)
        _, data_windows = rolling_window(coordinates, **kwargs)
        # Ravel the indices
        source_windows = [i[0] for i in source_windows.ravel()]
        data_windows = [i[0] for i in data_windows.ravel()]
        # Shuffle windows
        if shuffle:
            source_windows, data_windows = utils.shuffle(
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
