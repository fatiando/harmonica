# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Classes for Euler-based source location estimation.
"""

import warnings

import numpy as np
import scipy as sp
import verde.base as vdb


class EulerDeconvolution:
    r"""
    Estimate source location and base level using Euler Deconvolution.

    Implements Euler Deconvolution [Reid1990]_ to estimate subsurface source
    location from potential field data and their directional derivatives. Also
    estimates any constant shifts or biases of the data (called the base level).
    The approach employs linear least-squares to solve Euler's homogeneity
    equation.

    **Assumes a single data window** and provides a single estimate.

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
        Defines the nature of the source of the potential field data. Should
        be an integer between 1 and 3. It's the degree of the field's rate of
        change with distance from the source, influencing the decay rate of the
        field and the formulation of Euler's homogeneity equation. **Correlated
        with the depth estimate**, so larger structural index will lead to
        larger depths. **Choose based on known source geometry**. See table
        below.

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
    and the structural index.

    **This assumes that the sources are ideal** (see the table below).
    We recommend reading [ReidThurston2014]_ for a discussion on what the
    structural index means and what it does not mean. After [ReidThurston2014]_,
    values of the structural index (SI) can be:

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

    def fit(self, coordinates, data):
        """
        Fit the model using potential field measurements and their derivatives.

        Solves Euler's homogeneity equation to estimate the source location and
        base level by utilizing field values and their spatial derivatives in
        easting, northing, and upward directions. Creates a pseudo-parametric
        model that assumes derivatives are free of error.

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
            The instance itself, updated with the estimated ``location_``
            and ``base_level_``.
        """
        coordinates, data, _ = vdb.check_fit_input(coordinates, data, weights=None)
        field, east_deriv, north_deriv, up_deriv = vdb.n_1d_arrays(data, 4)
        easting, northing, upward = vdb.n_1d_arrays(coordinates, 3)
        n_data = field.size
        n_params = 4
        jacobian = np.empty((n_data, n_params))
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
        # Invert the Hessian instead of solving the system because this is a 4 x 4 or
        # 3 x 3 system and it won't cost much more. Plus we need the inverse anyway
        # to estimate the covariance matrix (used as a filtering criterion in windowed
        # implementations)
        hessian_inv = sp.linalg.inv(hessian)
        estimate = hessian_inv @ jacobian.T @ pseudo_data
        pseudo_residuals = pseudo_data - jacobian @ estimate
        chi_squared = np.sum(pseudo_residuals**2) / (n_data - n_params)
        self.covariance_ = chi_squared * hessian_inv
        self.location_ = estimate[:3]
        self.base_level_ = estimate[-1]
        return self


class EulerInversion:
    r"""
    Estimate source location, base level, and SI using Euler Inversion.

    Implements Euler Inversion [Uieda2025]_ to estimate subsurface source
    location from potential field data and their directional derivatives. Also
    estimates any constant shifts or biases of the data (called the base level),
    as well as the structural index (SI; a parameter reflecting the source
    geometry; see below). The approach employs a non-linear total-least-squares
    approach to solve the inverse problem of Euler's homogeneity equation.

    **Assumes a single data window** and provides a single estimate.

    .. hint::

        **Euler Inversion is much more stable than Euler Deconvolution.** It's
        less sensitive to noise in the field derivatives and to interfering
        sources within the data window. It can also estimate integer-valued
        structural indices (SI).

    .. note::

        Does not yet support structural index 0.

    Parameters
    ----------
    structural_index : int
        Defines the nature of the source of the potential field data. Should be an
        integer between 1 and 3. It's the degree of the field's rate of change with
        distance from the source, influencing the decay rate of the field and the
        formulation of Euler's homogeneity equation. **Correlated with the depth
        estimate**, so larger structural index will lead to larger depths. **Choose
        based on known source geometry**. See table below.
    max_iterations : int
        The maximum number of iterations allowed in the non-linear Gauss-Newton
        inversion. If the value is too small, there is a risk of exiting the
        inversion without the solution converging to the minimum of the goal
        function. Larger values won't necessarily lead to longer computation
        times since the inversion will stop if convergence is reached.
    tol : float
        The tolerance in decimal percentage that is needed to continue the
        iterations. If the change in the merit function (see below) is less than
        ``tol`` times the current merit function value, the iterations will be
        terminated. Use smaller values to allow for longer inversions.
    euler_misfit_balance : float
        The trade-off parameter :math:`\nu` between fitting the data and obeying
        Euler's homogeneity equation (see below).

    Attributes
    ----------
    location_ : 1d-array
        Estimated (easting, northing, upward) coordinates of the source after
        model fitting.
    base_level_ : float
        Estimated base level constant of the anomaly after model fitting.
    covariance_ : 2d-array
        The 4 x 4 estimated covariance matrix of the solution. Parameters are in the
        order: easting, northing, upward, base level. **This is not an uncertainty of
        the position** but a rough estimate of their variance with regard to the data.
    structural_index_  : int
        The estimated structural index.

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

    The Euler Inversion estimates :math:`(e_0, n_0, u_0)` and :math:`b` given
    a potential field and its easting, northing, and upward derivatives.
    If the structural index is not given, it can estimate an integer valued
    :math:`\eta` by running the inversion multiple times and choosing the
    :math:`\eta` that produces the best fit to the data. This is a big advantage
    of the Euler Inversion approach over the Deconvolution since the latter
    is unable to calculate predicted data and thus cannot evaluate true data
    misfit.

    The convergence of the solution is measured through a *merti function*

    .. math::

        \mathcal{M}(\mathbf{p}, \mathbf{d}) =
        \sqrt{\mathbf{r}^T\mathbf{W}\mathbf{r}} +
        \nu\sqrt{\mathbf{e}^T\mathbf{e}}

    in which :math:`\mathbf{p}` is the parameter vector, :math:`\mathbf{d}`
    is the predicted data vector, :math:`\mathbf{W}` is the weight matrix,
    :math:`\mathbf{r}` is the residual vector, :math:`\mathbf{e}` is the
    evaluation of Euler's equation using the current data and parameters, and
    :math:`\nu` is trade-off parameter that balances fitting the data with
    obeying Euler's equation.

    As with Euler Deconvolution, Euler Inversion **still assumes that
    the sources are ideal** (see the table below). We recommend reading
    [ReidThurston2014]_ for a discussion on what the structural index means and
    what it does not mean. After [ReidThurston2014]_, values of the structural
    index (SI) can be:

    ===================================== ======== =========
    Source type                           SI (Mag) SI (Grav)
    ===================================== ======== =========
    Point, sphere                            3         2
    Line, cylinder, thin bed fault           2         1
    Thin sheet edge, thin sill, thin dyke    1         0
    ===================================== ======== =========

    """

    def __init__(
        self,
        structural_index=None,
        max_iterations=20,
        tol=0.1,
        euler_misfit_balance=0.1,
    ):
        self.structural_index = structural_index
        self.max_iterations = max_iterations
        self.tol = tol
        self.euler_misfit_balance = euler_misfit_balance

    def fit(self, coordinates, data, weights=(1, 0.1, 0.1, 0.025)):
        """
        Fit the model using potential field measurements and their derivatives.

        Solves Euler's homogeneity equation to estimate the source location
        and base level by utilizing field values and their spatial derivatives
        in easting, northing, and upward directions. Constructs an implicit
        mathematical model and estimates both the parameters and the predicted
        data (field and its derivatives). Will also estimate the structural
        index if a value was not provided.

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
        weights : tuple, list, 1d-array, optional
            Weights assigned to each of the four data types (field and
            its derivatives) in the inversion. Reducing the weights of the
            derivatives helps reduce the influence of random noise in the
            results. By default, weights are 1 for the field, 0.1 for its
            eastward and northward derivatives, and 0.025 for its upward
            derivative. The upward derivative has a smaller weight because it
            usually contains more errors arising from FFT-based processing.

        Returns
        -------
        self
            The instance itself, updated with the estimated ``location_``,
            ``base_level_``, and ``structural_index_``.
        """
        coordinates, data, _ = vdb.check_fit_input(coordinates, data, weights=None)
        data = vdb.n_1d_arrays(data, 4)
        coordinates = vdb.n_1d_arrays(coordinates, 3)
        if self.structural_index is None:
            candidates = []
            for si in (1, 2, 3):
                euler = EulerInversion(
                    structural_index=si,
                    max_iterations=self.max_iterations,
                    tol=self.tol,
                    euler_misfit_balance=self.euler_misfit_balance,
                )
                euler.fit(coordinates, data, weights)
                candidates.append(euler)
            best = candidates[np.argmin([e.data_misfit_ for e in candidates])]
            self.structural_index_ = best.structural_index
            self.location_ = best.location_
            self.base_level_ = best.base_level_
            self.covariance_ = best.covariance_
        else:
            self._fit(coordinates, data, weights)
            self.structural_index_ = self.structural_index
        return self

    def _fit(self, coordinates, data, weights):
        """
        Run the Euler Inversion when there is a specified structural index.
        """
        n_data = data[0].size
        # The data are organized into a single vector because of the maths
        data_observed = np.concatenate(data)
        data_predicted = 0.9 * data_observed
        parameters = np.empty(4)
        # Make an initial estimate for the parameters using Euler Deconvolution
        euler_deconv = EulerDeconvolution(structural_index=self.structural_index)
        euler_deconv.fit(coordinates, data)
        parameters[:3] = euler_deconv.location_
        parameters[3] = euler_deconv.base_level_
        # Create the data weights vector
        data_weights = np.empty_like(data_predicted)
        data_weights[:n_data] = weights[0]
        data_weights[n_data : 2 * n_data] = weights[1]
        data_weights[2 * n_data : 3 * n_data] = weights[2]
        data_weights[3 * n_data : 4 * n_data] = weights[3]
        # Store the inverse of each section of the data weights matrix for use in later
        # computations since we can take advantage of its block nature.
        Wd_inv = [np.full(n_data, 1 / w) for w in weights]
        # Keep track of the way these metrics vary with iteration
        euler = self._eulers_equation(coordinates, data_predicted, parameters)
        residuals = data_observed - data_predicted
        euler_misfit = np.linalg.norm(euler)
        data_misfit = np.linalg.norm(residuals * data_weights)
        merit = data_misfit + self.euler_misfit_balance * euler_misfit
        for _ in range(self.max_iterations):
            parameter_step, data_step, cofactor = self._newton_step(
                coordinates,
                data_observed,
                data_predicted,
                parameters,
                euler,
                Wd_inv,
            )
            parameters += parameter_step
            data_predicted += data_step
            # Update metrics
            euler = self._eulers_equation(coordinates, data_predicted, parameters)
            residuals = data_observed - data_predicted
            euler_misfit = np.linalg.norm(euler)
            data_misfit = np.linalg.norm(residuals * data_weights)
            new_merit = data_misfit + self.euler_misfit_balance * euler_misfit
            merit_change = abs((merit - new_merit) / merit)
            merit = new_merit
            # Check for convergence
            if merit_change < self.tol:
                break
        else:
            message = (
                "Euler Inversion exited because maximum number of iterations was reached"
                " and not because the algorithm converged. Consider increasing the"
                " maximum number of iterations."
            )
            warnings.warn(message, stacklevel=2)
        # Save output attributes
        self.location_ = parameters[:3]
        self.base_level_ = parameters[3]
        chi_squared = np.sum(residuals**2) / (residuals.size - parameters.size)
        self.covariance_ = chi_squared * cofactor
        self.data_misfit_ = data_misfit
        return self

    def _newton_step(
        self, coordinates, data_observed, data_predicted, parameters, euler, Wd_inv
    ):
        """
        Calculate the step in parameters and data in the Gauss-Newton iteration.
        """
        deriv_east, deriv_north, deriv_up = np.split(data_predicted, 4)[1:]
        A = self._parameter_jacobian(deriv_east, deriv_north, deriv_up)
        B_diags = self._data_jacobian_diagonals(coordinates, parameters[:3])
        B = sp.sparse.hstack([sp.sparse.diags(b) for b in B_diags])
        WBT = sp.sparse.hstack(
            [sp.sparse.diags(w * b) for b, w in zip(B_diags, Wd_inv, strict=True)]
        ).T
        residuals = data_observed - data_predicted
        # Q = B @ Wd_inv @ B.T
        Q_inv = sp.sparse.diags(
            1 / sum([b**2 * w for b, w in zip(B_diags, Wd_inv, strict=True)])
        )
        ATQ = A.T @ Q_inv
        BTQ = WBT @ Q_inv
        Br = B @ residuals
        cofactor = sp.linalg.inv(ATQ @ A)
        parameter_step = -cofactor @ ATQ @ (euler + Br)
        data_step = residuals - BTQ @ (Br + euler + A @ parameter_step)
        return parameter_step, data_step, cofactor

    def _parameter_jacobian(
        self,
        deriv_east,
        deriv_north,
        deriv_up,
    ):
        """
        Calculate the model parameter Jacobian for Euler Inversion.
        """
        jacobian = np.empty((deriv_east.size, 4), dtype="float64")
        jacobian[:, 0] = -deriv_east
        jacobian[:, 1] = -deriv_north
        jacobian[:, 2] = -deriv_up
        jacobian[:, 3] = -self.structural_index
        return jacobian

    def _data_jacobian_diagonals(self, coordinates, source_location):
        """
        Calculate the data Jacobian for Euler Inversion.
        """
        east, north, up = coordinates
        east_s, north_s, up_s = source_location
        nequations = east.size
        diagonals = [
            np.full(nequations, self.structural_index, dtype="float64"),
            east - east_s,
            north - north_s,
            up - up_s,
        ]
        return diagonals

    def _eulers_equation(self, coordinates, data, parameters):
        """
        Evaluate Euler's homogeneity equation.
        """
        east, north, up = coordinates
        field, deriv_east, deriv_north, deriv_up = np.split(data, 4)
        east_s, north_s, up_s = parameters[:3]
        base_level = parameters[-1]
        euler = (
            (east - east_s) * deriv_east
            + (north - north_s) * deriv_north
            + (up - up_s) * deriv_up
            + self.structural_index * (field - base_level)
        )
        return euler
