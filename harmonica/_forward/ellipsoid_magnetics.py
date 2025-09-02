# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Forward modelling for the magnetic field anomaly produced by ellipsoidal
bodies.
"""

from collections.abc import Iterable

import numpy as np
from scipy.constants import mu_0
from scipy.special import ellipeinc, ellipkinc

import harmonica as hm

from .ellipsoid_gravity import _get_g_values
from .utils_ellipsoids import _calculate_lambda, _get_v_as_euler


# internal field N matrix functions
def ellipsoid_magnetics(
    coordinates,
    ellipsoids,
    susceptibility,
    external_field,
    remnant_mag=None,
    field="b",
):
    """
    Produces the components for the magnetic field components (be, bn, bu):

        - Unpacks ellipsoid instance parameters (a, b, c, yaw, pitch, roll,
          origin)
        - Constructs Euler rotation matrix
        - Rotates observation points (e, n, u) into local ellipsoid system
          (x, y, z)
        - constructs susceptability matrix and depolarisation matrix for
          internal
          and external points of observation
        - Calculates the magentic field due to the magnetised ellipsoid (H())
        - Converts this to magentic induction (B()) in nT.

    Parameters
    ----------
    coordinates: tuple of easting (e), northing (n), upward (u) points
        e : ndarray
            Easting coordinates, in the form:
                - A scalar value (float or int)
                - A 1D array of shape (N,)
                - A 2D array (meshgrid) of shape (M, N)

        n : ndarray
            Northing coordinates, same shape and rules as 'e'.

        u : ndarray
            Upward coordinates, e.g. the surface height desired to compute the
            gravity value. Same shape and rules as 'e'.

    ellipsoid* : value, or list of values
        instance(s) of TriaxialEllipsoid, ProlateEllipsoid,
                 OblateEllipsoid
        Geometric description of the ellipsoid:
            - Semiaxes : a, b, c**
            - Orientation : yaw, pitch, roll**
            - Origin : centre point (x, y, z)

    susceptibility : list of floats or arrays
        Susceptibilty value. A single value or list of single values assumes
        isotropy in the body/bodies. An array or list of arrays should be a 3x3
        matrix with the given susceptibilty components, suggesting an
        anisotropic susceptibility.

    external_field : ndarray
        The uniform magnetic field (B) as and array with values of
        (magnitude, inclination, declination). The magnitude should be in nT,
        and the angles in degrees.

    remnant_mag:  (optional) array
        Remnent magnetisation vector of the body. Default is None.

    field : (optional) str, one of either "e", "n", "u".
        if no input is given, the function will return all three components of
        magentic induction.

    Returns
    -------
    be: ndarray
        Easting component of the magnetic field.

    bn ndarray
        Northing component of the magnetic field.

    bu: ndarray
        Upward component of the magnetic field.

    References
    ----------
    Clark, S. A., et al. (1986), "Magnetic and gravity anomalies of a trixial
    ellipsoid"
    Takahashi, Y., et al. (2018), "Magentic modelling of ellipsoidal bodies"
    For derivations of the equations and methods used in this code.
    """
    # check inputs are of the correct type
    if not isinstance(ellipsoids, Iterable):
        ellipsoids = [ellipsoids]

    if not isinstance(susceptibility, Iterable):
        susceptibility = [susceptibility]

    if remnant_mag is not None:
        mr = np.asarray(remnant_mag, dtype=float)

        if mr.ndim == 1 and mr.size == 3:
            mr = np.tile(mr, (len(ellipsoids), 1))

        if mr.shape != (len(ellipsoids), 3):
            msg = (
                f"Remanent magnetisation must have shape "
                f"({len(ellipsoids)}, 3); got {mr.shape}."
            )
            raise ValueError(msg)
    if remnant_mag is None:
        mr = np.zeros((len(ellipsoids), 3))

    if not isinstance(external_field, Iterable) and len(external_field) != 3:
        msg = (
            "External field  must contain three values (M, I, D):"
            f" instead got {external_field}."
        )
        raise ValueError(msg)

    # unpack coordinates, set up arrays to hold results
    e, n, u = [np.atleast_1d(np.asarray(coordinates[i])) for i in range(3)]

    broadcast = (np.broadcast(e, n, u)).shape
    be, bn, bu = (
        np.zeros(e.shape).ravel(),
        np.zeros(e.shape).ravel(),
        np.zeros(e.shape).ravel(),
    )

    # unpack external field, change to vector
    magnitude, inclination, declination = external_field
    b0 = np.array(hm.magnetic_angles_to_vec(magnitude, inclination, declination))
    h0 = b0 * 1e-9 / mu_0

    # loop over each given ellipsoid
    for ellipsoid, susceptibility, m_r in zip(
        ellipsoids, susceptibility, mr, strict=True
    ):
        k_matrix = check_susceptibility(susceptibility)

        a, b, c = ellipsoid.a, ellipsoid.b, ellipsoid.c
        ox, oy, oz = ellipsoid.centre
        yaw, pitch, roll = ellipsoid.yaw, ellipsoid.pitch, ellipsoid.roll

        cast = np.broadcast(e, n, u)
        obs_points = np.vstack(((e - ox).ravel(), (n - oy).ravel(), (u - oz).ravel()))
        r = _get_v_as_euler(yaw, pitch, roll)
        rotated = r.T @ obs_points
        x, y, z = [axis.reshape(cast.shape).ravel() for axis in rotated]

        lmbda = _calculate_lambda(x, y, z, a, b, c).ravel()
        internal_mask = ((x**2) / (a**2) + (y**2) / (b**2) + (z**2) / (c**2)) < 1

        h0_rot = r.T @ h0

        m = _get_magnetisation_with_rem(a, b, c, k_matrix, h0_rot, m_r)

        n_cross = _construct_n_matrix_internal(a, b, c)

        # create N matricies for each given point
        for idx in range(len(lmbda)):
            lam = lmbda[idx]
            xi, yi, zi = x[idx], y[idx], z[idx]
            is_internal = internal_mask[idx]

            if is_internal:
                hr = (-n_cross + np.identity(3)) @ m

                hr = r @ hr
                be[idx] += 1e9 * mu_0 * hr[0]
                bn[idx] += 1e9 * mu_0 * hr[1]
                bu[idx] += 1e9 * mu_0 * hr[2]

            else:
                nr = _construct_n_matrix_external(xi, yi, zi, a, b, c, lam)

                hr = nr @ m
                # print('H_ext', hr)
                # hr = r @ h_ext

                be[idx] += 1e9 * mu_0 * hr[0]
                bn[idx] += 1e9 * mu_0 * hr[1]
                bu[idx] += 1e9 * mu_0 * hr[2]

    be = be.reshape(broadcast)
    bn = bn.reshape(broadcast)
    bu = bu.reshape(broadcast)
    # return according to user
    return {"e": be, "n": bn, "u": bu}.get(field, (be, bn, bu))


def _get_magnetisation_with_rem(a, b, c, k, h0, mr):
    r"""
    Get the magnetization vector from the ellipsoid parameters and the rotated
    external field.

    Parameters
    ----------
    a, b, c : floats
        Semi-axes lengths of the ellipsoid.
    k : (3, 3) array
        Susceptibility tensor.
    h0: array
        The rotated background field (in local coordinates).

    Returns
    -------
    m (magentisation): array
        The magnetisation vector for the defined body.

    Notes
    -----
    Considering an ellipsoid with susceptibility :math:`\chi` (scalar or
                                                               tensor) in
    a uniform background field :math:`\mathbf{H}_0`, compute the magnetization
    vector
    :math:`\mathbf{M}` of the ellipsoid accounting for demagnetization effects
    as:

    .. math::

        \mathbf{M} =
        \chi \[left \mathbf{I} + \mathbf{N}^\text{int} \chi \right]^{-1}
        \mathbf{H}_0,

    where :math:`\mathbf{N}^\text{int}` is the internal demagnetization tensor,
    defined as:

    .. math::

        \mathbf{H}(\mathbf{r}) = \mathbf{H}_0 - \mathbf{N}(\mathbf{r})
        \mathbf{M}.
    """
    n_cross = _construct_n_matrix_internal(a, b, c)
    eye = np.identity(3)
    lhs = eye + n_cross @ k
    rhs = mr + k @ h0
    m = np.linalg.solve(lhs, rhs)
    return m


def check_susceptibility(susceptibility):
    """
    Check whether user has input a k value with anisotropy.

    Parameters
    ----------
    susceptibility : list of floats or arrays
        Susceptibilty value. A single value or list of single values assumes
        isotropy in the body/bodies. An array or list of arrays should be a 3x3
        matrix with the given susceptibilty components, suggesting an
        anisotropic susceptibility.

    Returns
    -------
    k_matrix: array
        the matrix for k (isotropic or anisotropic)

    """
    if isinstance(susceptibility, (int, float)):
        k_matrix = susceptibility * np.identity(3)
    elif isinstance(susceptibility, (list, tuple, np.ndarray)):
        k_array = np.asarray(susceptibility)
        if k_array.shape != (3, 3):
            msg = f"Susceptibility matrix must be 3x3, got shape {k_array.shape}"
            raise ValueError(msg)
        k_matrix = k_array
    else:
        msg = f"Unrecognized susceptibility type: {type(susceptibility)}"
        raise ValueError(msg)

    return k_matrix


def _depol_triaxial_int(a, b, c):
    """
    Calculate the internal depolarisation tensor (N(r)) for the triaxial case.

    Parameters
    ----------
    a, b, c : floats
        Semiaxis lengths of the triaxial ellipsoid (a ≥ b ≥ c).

    Returns
    -------
    nxx, nyy, nzz : floats
        individual diagonal components of the x, y, z matrix.


    """
    phi = np.arccos(c / a)
    k = (a**2 - b**2) / (a**2 - c**2)

    nxx = (
        (a * b * c)
        / (np.sqrt(a**2 - c**2) * (a**2 - b**2))
        * (ellipkinc(phi, k) - ellipeinc(phi, k))
    )
    nyy = (
        -1 * nxx
        + ((a * b * c) / (np.sqrt(a**2 - c**2) * (b**2 - c**2))) * ellipeinc(phi, k)
        - c**2 / (b**2 - c**2)
    )
    nzz = -1 * ((a * b * c) / (np.sqrt(a**2 - c**2) * (b**2 - c**2))) * ellipeinc(
        phi, k
    ) + b**2 / (b**2 - c**2)

    np.testing.assert_allclose((nxx + nyy + nzz), 1, rtol=1e-4)
    return nxx, nyy, nzz


def _depol_prolate_int(a, b):
    """
    Calcualte internal depolarisation factors for prolate case.

    Parameters
    ----------
    a, b: floats
        Semiaxis lengths of the prolate ellipsoid (a > b = c).

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

    if (m + np.sqrt(m**2 - 1)) < 0 or (m**2 - 1) < 0:
        msg = (
            "Values in the internal N matrix calculation"
            " are less than 0 - errors may occur."
        )
        raise RuntimeWarning(msg)
    nyy = nzz = 0.5 * (1 - nxx)

    return nxx, nyy, nzz


def _depol_oblate_int(a, b):
    """
    Calcualte internal depolarisation factors for oblate case.

    Parameters
    ----------
    a, b: floats
        Semiaxis lengths of the oblate ellipsoid (a < b = c).

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


def _construct_n_matrix_internal(a, b, c):
    """
    Construct the N matrix for the internal field using the above functions.

    Parameters
    ----------
    a, b, c : floats
        Semiaxis lengths of the given ellipsoid.

    Returns
    -------
    N : matrix
        depolarisation matrix (diagonal-only values) for the given ellipsoid.

    """
    # only diagonal elements
    # Nii corresponds to the above functions
    if a > b > c:
        func = _depol_triaxial_int(a, b, c)
    elif a > b and b == c:
        func = _depol_prolate_int(a, b, c)
    elif a < b and b == c:
        func = _depol_oblate_int(a, b, c)
    else:
        msg = "Could not determine ellipsoid type for values given."
        raise ValueError(msg)
    # construct identity matrix
    n = np.diag(func)

    return n


# construct components of the external matrix


def _get_h_values(a, b, c, lmbda):
    """
    Get the h values for the N matrix. Each point has its own h value and hence
    external N matrix.

    Parameters
    ----------
    a, b, c : floats
        Semiaxis lengths of the given ellipsoid.

    lmbda : float
        the given lmbda value for the point we are considering with this
        matrix.

    Returns
    -------
    h : float
        the h value for the given point.

    """
    axes = np.array([a, b, c])
    r = np.sqrt(np.prod(axes**2 + lmbda))

    return -1 / ((axes**2 + lmbda) * r)


def _spatial_deriv_lambda(x, y, z, a, b, c, lmbda):
    """
    Get the spatial derivative of lambda with respect to the x,y,z.

    Parameters
    ----------
    x, y, z : floats
        A singular observation point in the local coordinate system.

    a, b, c : floats
        Semiaxis lengths of the given ellipsoid.

    lmbda : float
        the given lmbda value for the point we are considering with this
        matrix.


    Returns
    -------
    vals : array of x, y, z components
        The spatial derivative for the given point.

    """
    # explicitly state:

    denom = (
        (x / (a**2 + lmbda)) ** 2
        + (y / (b**2 + lmbda)) ** 2
        + (z / (c**2 + lmbda)) ** 2
    )

    dλ_dx = (2 * x) / (a**2 + lmbda) / denom
    dλ_dy = (2 * y) / (b**2 + lmbda) / denom
    dλ_dz = (2 * z) / (c**2 + lmbda) / denom

    return np.stack([dλ_dx, dλ_dy, dλ_dz], axis=-1)


def _construct_n_matrix_external(x, y, z, a, b, c, lmbda):
    """
    Construct the N matrix for the external field.

    Parameters
    ----------
    x, y, z : floats
        A singular observation point in the local coordinate system.

    a, b, c : floats
        Semiaxis lengths of the given ellipsoid.

    lmbda : float
        the given lmbda value for the point we are considering with this
        matrix.


    Returns
    -------
    N : matrix
        External points' depolarisation matrix for the given point.

    """
    # g values here are equivalent to the A(lambda) etc values previously.
    # h values as above
    # lambda derivatives as above
    n = np.empty((3, 3))
    r = [x, y, z]
    gvals = _get_g_values(a, b, c, lmbda)
    derivs_lmbda = _spatial_deriv_lambda(x, y, z, a, b, c, lmbda)
    h_vals = _get_h_values(a, b, c, lmbda)

    for i in range(len(n)):
        for j in range(len(n[0])):
            if i == j:
                n[i][j] = (
                    -1
                    * ((a * b * c) / 2)
                    * (derivs_lmbda[i] * h_vals[i] * r[i] + gvals[i])
                )
            else:
                n[i][j] = -1 * ((a * b * c) / 2) * (derivs_lmbda[i] * h_vals[j] * r[j])

    trace_terms = []
    for i in range(3):
        trace_component = derivs_lmbda[i] * h_vals[i] * r[i] + gvals[i]
        trace_terms.append(trace_component)

    # print("external field N", n)
    # np.testing.assert_allclose(n[0][0] + n[1][1] + n[2][2], 0)
    return n
