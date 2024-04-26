# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
import numpy as np
import scipy as sp


class EulerDeconvolution:
    r"""
    Euler deconvolution for estimating source location and base level from
    potential field measurements.

    Implements Euler deconvolution to estimate subsurface source locations
    and a base level constant using potential field measurements and their
    directional derivatives. The approach employs linear least-squares to solve
    Euler's homogeneity equation.

    Parameters
    ----------
    structural_index : float
        Defines the degree of the field's rate of change with distance from
        the source, influencing the decay rate of the field and the formulation
        of Euler's homogeneity equation.

    Attributes
    ----------
    location_ : numpy.ndarray
        Estimated (x, y, z) coordinates of the source after model fitting.
    base_level_ : float
        Estimated base level constant of the anomaly after model fitting.

    References
    ----------
    [Reid1990]_
    """

    def __init__(self, structural_index):
        self.structural_index = structural_index
        # The estimated parameters. Start them with None
        self.location_ = None
        self.base_level_ = None

    def fit(self, coordinates, field, east_deriv, north_deriv, up_deriv):
        """
        Fit the model using potential field measurements and their derivatives.

        Solves Euler's homogeneity equation to estimate the source location
        and base level by utilizing field values and their spatial derivatives
        in east, north, and upward directions.

        Parameters
        ----------
        coordinates : tuple of 1d-arrays
            Arrays with the coordinates of each data point, in the order of
            (x, y, z), representing easting, nothing, and upward directions,
            respectively.
        field : 1d-array
            Field measurements at each data point.
        east_deriv, north_deriv, up_deriv : 1d-array
            Partial derivatives of the field with respect to east, north, and
            upward directions, respectively.

        Returns
        -------
        self
            The instance itself, updated with the estimated `location_`
            and `base_level_`.

        Examples
        --------
        >>> import numpy as np
        >>> import verde as vd
        >>> import harmonica as hm
        >>> # Generate the dipole position and magnetic moment
        >>> dipole_coordinates = [[50e3], [50e3], [-10e3]]
        >>> dipole_moments = hm.magnetic_angles_to_vec(1.0e14, 0, 0)
        >>> # Set a regional field
        >>> inc, dec = -40, 15
        >>> fe, fn, fu = hm.magnetic_angles_to_vec(1, inc, dec)
        >>> # Generate the coordinates
        >>> region = [0, 100e3, 0, 80e3]
        >>> coordinates = vd.grid_coordinates(region, spacing=500,
        ...                                   extra_coords=500)
        >>> # Calculate the dipole signal and total anomaly
        >>> be, bn, bu = hm.dipole_magnetic(coordinates, dipole_coordinates,
        ...                                 dipole_moments, field='b')
        >>> anomaly = fe * be + fn * bn + fu * bu
        >>> grid = vd.make_xarray_grid(coordinates, anomaly, data_names="tfa",
        ...                            extra_coords_names="upward")
        >>> # Caclulate the derivatives
        >>> grid["d_east"] = hm.derivative_easting(grid.tfa)
        >>> grid["d_north"] = hm.derivative_northing(grid.tfa)
        >>> grid["d_up"] = hm.derivative_upward(grid.tfa)
        >>> # Slice the window data
        >>> window = [30e3, 70e3, 30e3, 80e3]
        >>> grid_sliced = grid.sel(northing=slice(window[0], window[1]),
        ...                        easting=slice(window[2], window[3]))
        >>> grid_table = vd.grid_to_table(grid_sliced)
        >>> euler = EulerDeconvolution(structural_index=3)
        >>> coordinates=(grid_table.easting, grid_table.northing,
        ...              grid_table.upward)
        >>> euler.fit(
        ...     (grid_table.easting, grid_table.northing, grid_table.upward),
        ...     grid_table.tfa, grid_table.d_east, grid_table.d_north,
        ...     grid_table.d_up)
        >>> print(f'{euler.location_[0]:.2e}')
        5.00e+04
        >>> print(f'{euler.location_[1]:.2e}')
        5.00e+04
        >>> print(f'{euler.location_[2]:.2e}')
        -1.00e+04
        >>> print(f'{euler.base_level_:.1f}')
        -7.2
        """
        n_data = field.shape[0]
        matrix = np.empty((n_data, 4))
        matrix[:, 0] = east_deriv
        matrix[:, 1] = north_deriv
        matrix[:, 2] = up_deriv
        matrix[:, 3] = self.structural_index
        data_vector = (
            coordinates[0] * east_deriv
            + coordinates[1] * north_deriv
            + coordinates[2] * up_deriv
            + self.structural_index * field
        )
        estimate = sp.linalg.solve(matrix.T @ matrix, matrix.T @ data_vector)

        self.location_ = estimate[:3]
        self.base_level_ = estimate[3]
