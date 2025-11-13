# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Forward modelling magnetic fields produced by ellipsoidal bodies.
"""

import warnings
from collections.abc import Iterable
from numbers import Real

import numpy as np
from scipy.constants import mu_0
from scipy.special import ellipeinc, ellipkinc

from .._utils import magnetic_angles_to_vec
from ..errors import NoPhysicalPropertyWarning
from ..typing import Coordinates, Ellipsoid
from .utils_ellipsoids import (
    calculate_lambda,
    get_derivatives_of_elliptical_integrals,
    get_elliptical_integrals,
)


def ellipsoid_magnetic(
    coordinates: Coordinates,
    ellipsoids: Iterable[Ellipsoid] | Ellipsoid,
    external_field: tuple[float, float, float],
):
    """
    Forward model magnetic fields of ellipsoids.

    Compute the magnetic field components for an ellipsoidal body at specified
    observation points.

    .. important::

        The magnetic field components are returned in nT.

    Parameters
    ----------
    coordinates : list of array
        List of arrays containing the ``easting``, ``northing`` and ``upward``
        coordinates of the computation points defined on a Cartesian coordinate
        system. All coordinates should be in meters.
    ellipsoid : ellipsoid or list of ellipsoids
        Ellipsoidal body represented by an instance of
        :class:`harmonica.TriaxialEllipsoid`, :class:`harmonica.ProlateEllipsoid`, or
        :class:`harmonica.OblateEllipsoid`, or a list of them.
    external_field : tuple
        The uniform magnetic field (B) as and array with values of
        (magnitude, inclination, declination). The magnitude should be in nT,
        and the angles in degrees.

    Returns
    -------
    be, bn, bu : array
        Easting, northing and upward magnetic field components in nT.

    References
    ----------
    [Clark1986]_
    [Takahashi2018]_
    """
    # Sanity checks for ellipsoids
    if not isinstance(ellipsoids, Iterable):
        ellipsoids = [ellipsoids]

    # Flatten coordinates
    cast = np.broadcast(*coordinates)
    easting, northing, upward = tuple(np.atleast_1d(c).ravel() for c in coordinates)

    # Allocate output arrays
    be, bn, bu = tuple(np.zeros_like(easting, dtype=np.float64) for _ in range(3))

    # Compute the inducing H0 field
    magnitude, inclination, declination = external_field
    b0_field = np.array(magnetic_angles_to_vec(magnitude, inclination, declination))
    b0_field *= 1e-9  # convert to SI units
    h0_field = b0_field / mu_0

    # Forward model the magnetic field of ellipsoids
    for ellipsoid in ellipsoids:
        # Skip ellipsoid without susceptibility nor remanent mag
        if ellipsoid.susceptibility is None and ellipsoid.remanent_mag is None:
            msg = (
                f"Ellipsoid {ellipsoid} doesn't have a susceptibility nor a "
                "remanent_mag value. It will be skipped."
            )
            warnings.warn(msg, NoPhysicalPropertyWarning, stacklevel=2)
            continue

        b_field = _single_ellipsoid_magnetic(
            (easting, northing, upward),
            ellipsoid,
            ellipsoid.susceptibility,
            ellipsoid.remanent_mag,
            h0_field,
        )
        be += b_field[0]
        bn += b_field[1]
        bu += b_field[2]

    # Reshape and convert to nT
    be, bn, bu = tuple(1e9 * b.reshape(cast.shape) for b in (be, bn, bu))
    return be, bn, bu


def _single_ellipsoid_magnetic(
    coordinates, ellipsoid, susceptibility, remanent_mag, h0_field
):
    """
    Forward model the magnetic field of a single ellipsoid.

    Parameters
    ----------
    coordinates : tuple of (n) arrays
        Easting, northing and upward coordinates of the observation points. They should
        be 1d arrays.
    ellipsoid : object
        Ellipsoid object for which the magnetic field will be computed.
    susceptibility : float or (3, 3) array
        Susceptibility scalar or tensor of the ellipsoid.
    remanent_mag : (3) array
        Remanent magnetization vector of the ellipsoid in SI units.
        The components of the vector should be in the easting-northing-upward coordinate
        system.
    h0_field : (3) array
        Array with the components of the external H field in SI units.
        The components of the vector should be in the easting-northing-upward coordinate
        system.

    Returns
    -------
    be, bn, bu : (n) arrays
        Arrays with the magnetic field components of the ellipsoid on the observation
        points.
    """
    # Shift coordinates to the center of the ellipsoid
    origin_e, origin_n, origin_u = ellipsoid.center
    easting, northing, upward = coordinates
    coords_shifted = (easting - origin_e, northing - origin_n, upward - origin_u)

    # Rotate observation points
    r_matrix = ellipsoid.rotation_matrix
    x, y, z = r_matrix.T @ np.vstack(coords_shifted)

    # Calculate lambda for each observation point
    lambda_ = calculate_lambda(x, y, z, ellipsoid.a, ellipsoid.b, ellipsoid.c)

    # Build internal demagnetization tensor
    n_tensor_internal = get_demagnetization_tensor_internal(
        ellipsoid.a, ellipsoid.b, ellipsoid.c
    )

    # Cast susceptibility and remanent magnetization into matrix and array, respectively
    susceptibility = cast_susceptibility(susceptibility)
    remanent_mag = cast_remanent_magnetization(remanent_mag)

    # Rotate the external field and the remanent magnetization
    h0_field_rotated = r_matrix.T @ h0_field
    remnant_mag_rotated = r_matrix.T @ remanent_mag

    # Get magnetization of the ellipsoid
    magnetization = get_magnetisation(
        ellipsoid.a,
        ellipsoid.b,
        ellipsoid.c,
        susceptibility,
        h0_field_rotated,
        remnant_mag_rotated,
        n_tensor=n_tensor_internal,
    )

    # Compute magnetic field on observation points
    be, bn, bu = tuple(np.zeros_like(easting) for _ in range(3))
    for i, (x_i, y_i, z_i, lambda_i) in enumerate(zip(x, y, z, lambda_, strict=True)):
        internal = _is_internal(x_i, y_i, z_i, ellipsoid)
        if internal:
            h_field = -n_tensor_internal @ magnetization
            b_field = mu_0 * (h_field + magnetization)
        else:
            n_tensor = get_demagnetization_tensor_external(
                x_i, y_i, z_i, ellipsoid.a, ellipsoid.b, ellipsoid.c, lmbda=lambda_i
            )
            h_field = -n_tensor @ magnetization
            b_field = mu_0 * h_field

        # Rotate the B field back into the global coordinate system
        b_field = r_matrix @ b_field
        be[i], bn[i], bu[i] = b_field

    return be, bn, bu


def _is_internal(x, y, z, ellipsoid):
    """
    Check if a given point(s) is internal or external to the ellipsoid.
    """
    a, b, c = ellipsoid.a, ellipsoid.b, ellipsoid.c
    return ((x**2) / (a**2) + (y**2) / (b**2) + (z**2) / (c**2)) < 1


def get_magnetisation(a, b, c, susceptibility, h0_field, remnant_mag, n_tensor=None):
    r"""
    Get magnetization vector for an ellipsoid.

    Get the magnetization vector of and ellipsoid considering induced and remanent
    magnetization. This function takes into account demagnetization effects.

    .. important::

        This function works in the local x, y, z coordinate system defined for the
        ellipsoid. The external field and the remanent magnetization vector should be
        provided in the x-y-z coordinate system. The generated magnetization vector will
        be defined in the same coordinate system.

    Parameters
    ----------
    a, b, c : floats
        Semi-axes lengths of the ellipsoid.
    susceptibility : (3, 3) array
        Susceptibility tensor.
    h0_field: array
        The rotated background field (in local coordinates).
    remnant_mag : (3) array
        Remnant magnetisation vector (in local coordinates).
    n_tensor : (3, 3) array, optional
        Demagnetization tensor inside the ellipsoid. If None, the demagnetization tensor
        will be calculated by the function itself. Pass an array if it was already
        precomputed, in order to save computation time.
        Default to None.

    Returns
    -------
    m (magnetisation): (3) array
        The magnetisation vector for the defined body in local coordinates.

    Notes
    -----
    Considering an ellipsoid with susceptibility :math:`\chi` (scalar or
    tensor) in a uniform background field :math:`\mathbf{H}_0`, and with remanent
    magnetization :math:`\mathbf{M}_r`, compute the magnetization vector
    :math:`\mathbf{M}` of the ellipsoid accounting for demagnetization effects as:

    .. math::

        \mathbf{M} =
        \[left \mathbf{I} + \chi \mathbf{N}^\text{int} \right]^{-1}
        \[left \chi \mathbf{H}_0 + \mathbf{M}_r \right],

    where :math:`\mathbf{N}^\text{int}` is the internal demagnetization tensor,
    defined as:

    .. math::

        \mathbf{H}(\mathbf{r}) = \mathbf{H}_0 - \mathbf{N}(\mathbf{r})
        \mathbf{M}.
    """
    if n_tensor is None:
        n_tensor = get_demagnetization_tensor_internal(a, b, c)
    eye = np.identity(3)
    lhs = eye + n_tensor @ susceptibility
    rhs = remnant_mag + susceptibility @ h0_field
    m = np.linalg.solve(lhs, rhs)
    return m


def cast_susceptibility(susceptibility):
    """
    Cast susceptibility into a susceptibility tensor.

    Check whether user has input a k value with anisotropy.

    Parameters
    ----------
    susceptibility : float or (3) array or (3, 3) array or None
        Susceptibility value. A single value or list of single values assumes
        isotropy in the body/bodies. An array or list of arrays should be a 3x3
        matrix with the given susceptibility components, suggesting an
        anisotropic susceptibility.

    Returns
    -------
    susceptibility: (3, 3) array
        Susceptibility tensor.
    """
    # TODO: update the docstrings
    if susceptibility is None:
        return np.zeros((3, 3), dtype=np.float64)
    if isinstance(susceptibility, Real):
        susceptibility = susceptibility * np.identity(3)
    elif isinstance(susceptibility, Iterable):
        susceptibility = np.asarray(susceptibility)
        if susceptibility.shape != (3, 3):
            msg = f"Susceptibility matrix must be 3x3, got shape {susceptibility.shape}"
            raise ValueError(msg)
    else:
        msg = f"Unrecognized susceptibility type: {type(susceptibility)}"
        raise TypeError(msg)
    return susceptibility


def cast_remanent_magnetization(remnant_mag):
    """
    Cast remanent magnetization to an array of three elements.

    Check if remanent magnetization has the right shape. If ``remnant_mag`` is None,
    then an array full of zeros will be returned.

    Parameters
    ----------
    remnant_mag : array-like or None
        Remanent magnetization. Pass an array, a list, or tuple of three elements, or
        pass None.

    Returns
    -------
    remnant_mag : (3) array
        Remanent magnetization as an array with 3 elements.

    Raises
    ------
    ValueError
        If the passed array doesn't have the right shape and dimensions.
    """
    # TODO: update docstring
    if remnant_mag is None:
        return np.zeros(3, dtype=np.float64)
    remnant_mag = np.asarray(remnant_mag)
    return remnant_mag


def get_demagnetization_tensor_internal(a, b, c):
    r"""
    Construct the demagnetization tensor N on external points.

    Parameters
    ----------
    a, b, c : floats
        Semi-axes lengths of the given ellipsoid.

    Returns
    -------
    N : matrix
        Demagnetization tensor for the given ellipsoid on internal points.

    Notes
    -----
    The elements of the demagnetization tensor are defined following the sign convention
    of Clark et al. (1986), in which the internal demagnetization tensor
    :math:`N_\text{int}` and the demagnetizing field :math:`\Delta \mathbf{H}` are
    related as follows:

    .. math::

        \Delta \mathbf{H}(\mathbf{r}) = - N_\text{int} \mathbf{M}

    where :math:`\mathbf{M}` is the magnetization vector of the ellipsoid.
    """
    if a > b > c:
        n_diagonal = _demag_tensor_triaxial_internal(a, b, c)
    elif a > b and b == c:
        n_diagonal = _demag_tensor_prolate_internal(a, b)
    elif a < b and b == c:
        n_diagonal = _demag_tensor_oblate_internal(a, b)
    else:
        msg = "Could not determine ellipsoid type for values given."
        raise ValueError(msg)

    n = np.diag(n_diagonal)
    return n


def _demag_tensor_triaxial_internal(a, b, c):
    """
    Calculate the internal demagnetization tensor (N(r)) for the triaxial case.

    Parameters
    ----------
    a, b, c : floats
        Semi-axes lengths of the triaxial ellipsoid (a ≥ b ≥ c).

    Returns
    -------
    nxx, nyy, nzz : floats
        individual diagonal components of the x, y, z matrix.
    """
    # Cache values of E(theta, k) and F(theta, k) so we compute them only once
    phi = np.arccos(c / a)
    k = (a**2 - b**2) / (a**2 - c**2)
    ellipk = ellipkinc(phi, k)
    ellipe = ellipeinc(phi, k)

    nxx = (a * b * c) / (np.sqrt(a**2 - c**2) * (a**2 - b**2)) * (ellipk - ellipe)
    nyy = (
        -1 * nxx
        + ((a * b * c) / (np.sqrt(a**2 - c**2) * (b**2 - c**2))) * ellipe
        - c**2 / (b**2 - c**2)
    )
    nzz = -1 * (
        (a * b * c) / (np.sqrt(a**2 - c**2) * (b**2 - c**2))
    ) * ellipe + b**2 / (b**2 - c**2)

    return nxx, nyy, nzz


def _demag_tensor_prolate_internal(a, b):
    """
    Calculate internal demagnetization factors for prolate case.

    Parameters
    ----------
    a, b: floats
        Semi-axes lengths of the prolate ellipsoid (a > b = c).

    Returns
    -------
    nxx, nyy, nzz : floats
        individual diagonal components of the x, y, z matrix.
    """
    m = a / b
    if not m > 1:
        msg = f"Invalid aspect ratio for prolate ellipsoid: a={a}, b={b}, a/b={m}"
        raise ValueError(msg)
    nxx = (1 / (m**2 - 1)) * (
        ((m / np.sqrt(m**2 - 1)) * np.log(m + np.sqrt(m**2 - 1))) - 1
    )
    nyy = nzz = 0.5 * (1 - nxx)
    return nxx, nyy, nzz


def _demag_tensor_oblate_internal(a, b):
    """
    Calculate internal demagnetization factors for oblate case.

    Parameters
    ----------
    a, b: floats
        Semi-axes lengths of the oblate ellipsoid (a < b = c).

    Returns
    -------
    nxx, nyy, nzz : floats
        individual diagonal components of the x, y, z matrix.
    """
    m = a / b
    if not 0 < m < 1:
        msg = f"Invalid aspect ratio for oblate ellipsoid: a={a}, b={b}, a/b={m}"
        raise ValueError(msg)
    nxx = 1 / (1 - m**2) * (1 - (m / np.sqrt(1 - m**2)) * np.arccos(m))
    nyy = nzz = 0.5 * (1 - nxx)
    return nxx, nyy, nzz


def get_demagnetization_tensor_external(x, y, z, a, b, c, lmbda):
    r"""
    Construct the demagnetization tensor N on external points.

    Parameters
    ----------
    x, y, z : floats
        Coordinates of the observation point in the local coordinate system.
    a, b, c : floats
        Semi-axes lengths of the given ellipsoid.
    lmbda : float
        The lambda value for the observation point.

    Returns
    -------
    N : matrix
        External points' demagnetization tensor for the given point.

    Notes
    -----
    The elements of the demagnetization tensor are defined following the sign convention
    of Clark et al. (1986), in which the tensor :math:`N` and the
    demagnetizing field :math:`\Delta \mathbf{H}` are related as follows:

    .. math::

        \Delta \mathbf{H}(\mathbf{r}) = - N(\mathbf{r}) \mathbf{M}

    where :math:`\mathbf{M}` is the magnetization vector of the ellipsoid.

    The components of the demagnetization tensor for any ellipsoid are given by:

    .. math::

        n_{ii} =
            \frac{abc}{2}
            \left[
                    \frac{\partial \lambda}{\partial r_i}
                    \frac{\text{d} F_i}{\text{d} \lambda}
                    r_i
                    +
                    F_i(\lambda)
            \right],
        \quad
        \forall i \in \{x, y, z\}

    and

    .. math::

        n_{ij} =
            \frac{abc}{2}
                \frac{\partial \lambda}{\partial r_i}
                \frac{\text{d} F_j}{\text{d} \lambda}
                r_j,
        \quad
        \forall i,j \in \{x, y, z\}, \, i \ne j

    where :math:`F_x(\lambda) = A(\lambda)`, :math:`F_y(\lambda) = B(\lambda)`,
    :math:`F_z(\lambda) = C(\lambda)`, and the :math:`r_i` are the :math:`x`, :math:`y`,
    and :math:`z` coordinates.

    Note the sign difference with Takahashi et al. (2018) equations 34 and 35.
    """
    n = np.empty((3, 3), dtype=np.float64)

    coords = (x, y, z)
    ellip_integrals = get_elliptical_integrals(a, b, c, lmbda)
    deriv_ellip_integrals = get_derivatives_of_elliptical_integrals(a, b, c, lmbda)
    derivs_lmbda = _spatial_deriv_lambda(x, y, z, a, b, c, lmbda)

    for i in range(len(n)):
        for j in range(len(n[0])):
            if i == j:
                n[i][j] = ((a * b * c) / 2) * (
                    derivs_lmbda[i] * deriv_ellip_integrals[i] * coords[i]
                    + ellip_integrals[i]
                )
            else:
                n[i][j] = ((a * b * c) / 2) * (
                    derivs_lmbda[i] * deriv_ellip_integrals[j] * coords[j]
                )

    return n


def _spatial_deriv_lambda(x, y, z, a, b, c, lmbda):
    """
    Get the spatial derivatives of lambda with respect to x, y, and z.

    Parameters
    ----------
    x, y, z : floats
        Coordinates of the observation point in the local coordinate system.
    a, b, c : floats
        Semi-axes lengths of the given ellipsoid.
    lmbda : float
        The given lambda value for the point we are considering with this matrix.

    Returns
    -------
    derivatives : tuple of floats
        The spatial derivatives of lambda for the given observation point.

    """
    denom = (
        (x / (a**2 + lmbda)) ** 2
        + (y / (b**2 + lmbda)) ** 2
        + (z / (c**2 + lmbda)) ** 2
    )

    dlambda_dx = (2 * x) / (a**2 + lmbda) / denom
    dlambda_dy = (2 * y) / (b**2 + lmbda) / denom
    dlambda_dz = (2 * z) / (c**2 + lmbda) / denom

    return dlambda_dx, dlambda_dy, dlambda_dz
