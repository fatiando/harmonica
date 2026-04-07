# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Classes for Euler Deconvolution of potential field data.
"""

import numpy as np
import scipy as sp
import verde.base as vdb


class EulerDeconvolution:
    r"""
    Estimate source location and base level using Euler Deconvolution.

    Implements Euler Deconvolution [Reid1990]_ to estimate subsurface source
    location and a base level constant from potential field data and their
    directional derivatives. The approach employs linear least-squares to solve
    Euler's homogeneity equation. **Assumes a single data window** and provides
    a single estimate.

    .. hint::

        Please read the paper [Reid2014]_ to avoid doing **horrible things**
        with Euler deconvolution. [Uieda2014]_ offer a practical tutorial using
        `legacy Fatiando a Terra <https://legacy.fatiando.org/>`__ code to show
        some common misinterpretations.

    .. note::

        Does not yet support structural index 0.

    Parameters
    ----------
    structural_index : int
        Defines the nature of the source of the potential field data. It's the
        degree of the field's rate of change with distance from the source,
        influencing the decay rate of the field and the formulation of Euler's
        homogeneity equation. **Correlated with the depth estimate**, so larger
        structural index will lead to larger depths. **Choose based on known
        source geometry**. See table below.

    Attributes
    ----------
    location_ : 1d-array
        Estimated (easting, northing, upward) coordinates of the source after
        model fitting.
    base_level_ : float
        Estimated base level constant of the anomaly after model fitting.
    covariance_ : 2d-array
        The 4 x 4 estimated covariance matrix of the solution. Parameters are
        in the order: easting, northing, upward, base level. **This is not an
        uncertainty of the position** but a rough estimate of their variance
        with regard to the data.

    Notes
    -----
    Works on any potential field that satisfies Euler's homogeneity equation
    (like gravity, magnetic, and their gradients caused by **simple sources**):

    .. math::

        (e_i - e_0)\dfrac{\partial f_i}{\partial e} +
        (n_i - n_0)\dfrac{\partial f_i}{\partial n} +
        (u_i - u_0)\dfrac{\partial f_i}{\partial u} =
        \eta (b - f_i),

    in which :math:`f_i` is the given potential field observation at point
    :math:`(e_i, n_i, u_i)`, :math:`b` is the base level (a constant shift of
    the field, like a regional field), :math:`\eta` is the structural index,
    and :math:`(e_0, n_0, u_0)` are the coordinates of a point on the source
    (for a sphere, this is the center point).

    The Euler deconvolution estimates :math:`(e_0, n_0, u_0)` and :math:`b`
    given a potential field and its easting, northing, and upward derivatives
    and the structural index. However, **this assumes that the sources are
    ideal** (see the table below). We recommend reading [ReidThurston2014]_ for
    a discussion on what the structural index means and what it does not mean.

    After [ReidThurston2014]_, values of the structural index (SI) can be:

    ===================================== ======== =========
    Source type                           SI (Mag) SI (Grav)
    ===================================== ======== =========
    Point, sphere                            3         2
    Line, cylinder, thin bed fault           2         1
    Thin sheet edge, thin sill, thin dyke    1         0
    ===================================== ======== =========

    """

    def __init__(self, structural_index):
        self.structural_index = structural_index
        # The estimated parameters. Start them with None
        self.location_ = None
        self.base_level_ = None
        self.covariance_ = None

    def fit(self, coordinates, data):
        """
        Fit the model using potential field measurements and their derivatives.

        Solves Euler's homogeneity equation to estimate the source location
        and base level by utilizing field values and their spatial derivatives
        in easting, northing, and upward directions.

        .. tip::

            Data does not need to be gridded for this to work.

        Parameters
        ----------
        coordinates : tuple of arrays
            Tuple of 3 with the coordinates of each data point. Should be in
            the following order: ``(easting, northing, upward)``.
            Arrays can be n-dimensional but must all have the same shape.
        data : tuple of arrays
            Tuple of 4 arrays with the observed data in the following order:
            ``(potential_field, derivative_easting, derivative_northing,
            derivative_upward)``. Arrays can be n-dimensional but must all have
            the same shape as the coordinates. Derivatives must be in data
            units over coordinates units, for example nT/m or mGal/m.

        Returns
        -------
        self
            The instance itself, updated with the estimated `location_`
            and `base_level_`.
        """
        coordinates, data, _ = vdb.check_fit_input(coordinates, data, weights=None)
        field, east_deriv, north_deriv, up_deriv = vdb.n_1d_arrays(data, 4)
        easting, northing, upward = vdb.n_1d_arrays(coordinates, 3)
        n_data = field.size
        jacobian = np.empty((n_data, 4))
        jacobian[:, 0] = east_deriv
        jacobian[:, 1] = north_deriv
        jacobian[:, 2] = up_deriv
        jacobian[:, 3] = self.structural_index
        pseudo_data = (
            easting * east_deriv
            + northing * north_deriv
            + upward * up_deriv
            + self.structural_index * field
        )
        hessian = jacobian.T @ jacobian
        # Invert the Hessian instead of solving the system because this is a
        # 4x4 system and it won't cost much more, plus we need the inverse
        # anyway to estimate the covariance matrix (used as a filtering
        # criterion in windowed implementations)
        hessian_inv = sp.linalg.inv(hessian)
        estimate = hessian_inv @ jacobian.T @ pseudo_data
        pseudo_residuals = pseudo_data - jacobian @ estimate
        chi_squared = np.sum(pseudo_residuals**2) / (n_data - 4)
        self.covariance_ = chi_squared * hessian_inv
        self.location_ = estimate[:3]
        self.base_level_ = estimate[3]
        return self
