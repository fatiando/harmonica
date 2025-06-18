"""
Forward modelling for the magnetic field anomaly produced by ellipsoidal
bodies.
"""

import numpy as np
from scipy.constants import mu_0
from scipy.special import ellipeinc, ellipkinc

from .ellipsoid_gravity import _get_abc
from .utils_ellipsoids import _calculate_lambda, _get_v_as_Euler


# internal field N matrix functions
def ellipsoid_magnetics(coordinates, ellipsoids, k, h0, field="b"):
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

    k : list of floats or arrays
        Susceptibilty value. A single value or list of single values assumes
        isotropy in the body/bodies. An array or list of arrays should be a 3x3
        matrix with the given susceptibilty components, suggesting an
        anisotropic susceptibility.

    H0 : ndarray
        Three components of the uniform inducing field.

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
    Takenhasi, Y., et al. (2018), "Magentic modelling of ellipsoidal bodies"

    For derivations of the equations, and methods used in this code.
    """
    # unpack coordinates, set up arrays to hold results
    e, n, u = coordinates[0], coordinates[1], coordinates[2]
    cast = np.broadcast(e, n, u)
    be, bn, bu = np.zeros(e.shape), np.zeros(e.shape), np.zeros(e.shape)

    # check inputs are of the correct type
    if type(ellipsoids) is not list:
        ellipsoids = [ellipsoids]

    if type(k) is not list:
        k = [k]

    if len(ellipsoids) != len(k):
        raise ValueError(
            "Magnetic susceptibilty must be a list containing the value"
            " of k for each ellipsoid. Instead, number of ellipsoids"
            f" given is {len(ellipsoids)} and number of k values is"
            f" {len(k)}."
        )

    if type(h0) is not np.ndarray:
        raise ValueError("H0 values of the regional field  must be an array.")

    # loop over each given ellipsoid
    for index, ellipsoid in enumerate(ellipsoids):

        # unpack instance
        a, b, c = ellipsoid.a, ellipsoid.b, ellipsoid.c
        yaw, pitch, roll = ellipsoid.yaw, ellipsoid.pitch, ellipsoid.roll
        ox, oy, oz = ellipsoid.centre

        # preserve ellipsoid shape, translate origin of ellipsoid
        cast = np.broadcast(e, n, u)
        obs_points = np.vstack(((e - ox).ravel(), (n - oy).ravel(), (u - oz).ravel()))

        # get observation points, rotate them
        r = _get_v_as_Euler(yaw, pitch, roll)
        rotated_points = r.T @ obs_points
        x, y, z = tuple(c.reshape(cast.shape) for c in rotated_points)

        # create boolean for internal vs external field points
        # and compute lambda for each coordinate point
        lmbda = _calculate_lambda(x, y, z, a, b, c)
        internal_mask = (x**2) / (a**2) + (y**2) / (b**2) + (z**2) / (c**2) < 1

        # create K matrix
        if type(k[index]) is not np.ndarray:
            k_matrix = k[index] * np.eye(3)
        else:
            k_matrix = k[index]

        # create N matricies for each given point
        for i, j in np.ndindex(lmbda.shape):
            lam = lmbda[i, j]
            xi, yi, zi = x[i, j], y[i, j], z[i, j]
            is_internal = internal_mask[i, j]

            n_cross = _construct_n_matrix_internal(a, b, c)

            if is_internal:
                n = n_cross
            else:
                n = _construct_n_matrix_external(xi, yi, zi, a, b, c, lam)

            # compute rotation and final H() values
            nr = r.T @ n @ r
            h_cross = np.linalg.inv(np.eye(3) + n_cross @ k_matrix) @ h0
            hr = h0 + (nr @ k_matrix) @ h_cross

            # sum across all components and ellipsoids
            be[i, j] += 1e9 * mu_0 * hr[0]
            bn[i, j] += 1e9 * mu_0 * hr[1]
            bu[i, j] += 1e9 * mu_0 * hr[2]

    # return according to user
    return {"e": be, "n": bn, "u": bu}.get(field, (be, bn, bu))


# construct components of the internal matrix


def _depol_triaxial_int(a, b, c):
    """
    Calculate the internal depolarisation tensor (N(r)) for the triaxial case.

    parameters
    ----------
    a, b, c : floats
        Semiaxis lengths of the triaxial ellipsoid (a ≥ b ≥ c).

    returns
    -------
    nxx, nyy, nzz : floats
        individual diagonal components of the x, y, z matrix.


    """

    phi = np.arccos(c / a)
    k = np.sqrt(((a**2 - b**2) / (a**2 - c**2)))
    coeff = (a * b * c) / (np.sqrt(a**2 - c**2) * (a**2 - b**2))

    nxx = coeff * (ellipkinc(phi, k) - ellipeinc(phi, k))
    nyy = -nxx + coeff * ellipeinc(phi, k) - c**2 / (b**2 - c**2)
    nzz = -coeff * ellipeinc(phi, k) + b**2 / (b**2 - c**2)

    return nxx, nyy, nzz


def _depol_prolate_int(a, b, c):
    """
    Calcualte internal depolarisation factors for prolate case.


    parameters
    ----------
    a, b, c : floats
        Semiaxis lengths of the prolate ellipsoid (a > b = c).

    returns
    -------
    nxx, nyy, nzz : floats
        individual diagonal components of the x, y, z matrix.

    """
    m = a / b
    nxx = 1 / (m**2 - 1) * ((m / np.sqrt(m**2 - 1)) * np.log(m + np.sqrt(m**2 - 1)) - 1)
    nyy = nzz = 0.5 * (1 - nxx)

    return nxx, nyy, nzz


def _depol_oblate_int(a, b, c):
    """
    Calcualte internal depolarisation factors for oblate case.

    parameters
    ----------
    a, b, c : floats
        Semiaxis lengths of the oblate ellipsoid (a < b = c).

    returns
    -------
    nxx, nyy, nzz : floats
        individual diagonal components of the x, y, z matrix.

    """

    m = a / b
    nxx = 1 / (1 - m**2) * (1 - (m / np.sqrt(1 - m**2)) * np.arccos(m))
    nyy = nzz = 0.5 * (1 - nxx)

    return nxx, nyy, nzz


def _construct_n_matrix_internal(a, b, c):
    """
    Construct the N matrix for the internal field using the above functions.

    parameters
    ----------
    a, b, c : floats
        Semiaxis lengths of the given ellipsoid.

    returns
    -------
    N : matrix
        depolarisation matrix (diagonal-only values) for the given ellipsoid.

    """

    # only diagonal elements
    # Nii corresponds to the above functions
    if np.all((a > b) & (b > c)):
        func = _depol_triaxial_int(a, b, c)
    if np.all((a > b) & (b == c)):
        func = _depol_prolate_int(a, b, c)
    if np.all((a < b) & (b == c)):
        func = _depol_oblate_int(a, b, c)

    # construct identity matrix
    n = np.eye(3)

    for i in range(3):
        n[i][i] *= func[i]

    return n


# construct components of the external matrix


def _get_h_values(a, b, c, lmbda):
    """
    Get the h values for the N matrix. Each point has its own h value and hence
    external N matrix.

    parameters
    ----------

    a, b, c : floats
        Semiaxis lengths of the given ellipsoid.

    lmbda : float
        the given lmbda value for the point we are considering with this
        matrix.

    returns
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

    parameters
    ----------
    x, y, z : floats
        A singular observation point in the local coordinate system.

    a, b, c : floats
        Semiaxis lengths of the given ellipsoid.

    lmbda : float
        the given lmbda value for the point we are considering with this
        matrix.


    returns
    -------

    vals : array of x, y, z components
        The spatial derivative for the given point.

    """

    # numerators (shape: same as x, y, z)
    num_x = 2 * x / (a**2 + lmbda)
    num_y = 2 * y / (b**2 + lmbda)
    num_z = 2 * z / (c**2 + lmbda)

    # denominator
    denom = (
        (x / (a**2 + lmbda)) ** 2
        + (y / (b**2 + lmbda)) ** 2
        + (z / (c**2 + lmbda)) ** 2
    )

    vals = np.stack([num_x / denom, num_y / denom, num_z / denom], axis=-1)

    return vals


def _get_g_values_magnetics(a, b, c, lmbda):
    """
    Compute the gravity values (g) for the three ellipsoid types. See
    ellipsoid_gravity for the in depth production of gravity components.

    parameters
    ----------

    a, b, c : floats
        Semiaxis lengths of the given ellipsoid.

    lmbda : float
        the given lmbda value for the point we are considering.

    returns
    -------

    gvals (x, y, z) : floats
        the g values for the given ellipsoid type, and given observation point.


    """

    # trixial case
    if a > b > c:
        func = _get_abc(a, b, c, lmbda)
        gvals_x, gvals_y, gvals_z = func[0], func[1], func[2]

    # prolate case
    if a > b and b == c:
        g1 = (
            2
            / ((a**2 - b**2) ** 3 / 2)
            * (
                np.log(
                    ((a**2 - b**2) ** 0.5 + (a**2 + lmbda) ** 0.5)
                    / (b**2 + lmbda) ** 0.5
                )
                - ((a**2 - b**2) / (a**2 + lmbda)) ** 0.5
            )
        )
        g2 = 1 / ((a**2 - b**2) ** 3 / 2) * (
            ((a**2 - b**2) * (a**2 + lmbda) ** 0.5) / (b**2 + lmbda)
        ) - (
            np.log(
                ((a**2 - b**2) ** 0.5 + (a**2 + lmbda) ** 0.5) / (b**2 + lmbda) ** 0.5
            )
        )
        gvals_x, gvals_y, gvals_z = g1, g2, g2

    # oblate case
    if a < b and b == c:
        g1 = (
            2
            / ((b**2 - a**2) ** 3 / 2)
            * (
                (((b**2 - a**2) / (a**2 + lmbda)) ** 0.5)
                - np.arctan(((b**2 - a**2) / (a**2 + lmbda)) ** 0.5)
            )
        )
        g2 = (
            1
            / ((b**2 - a**2) ** 3 / 2)
            * (
                np.arctan(((b**2 - a**2) / (a**2 + lmbda)) ** 0.5)
                - (((b**2 - a**2) * (a**2 + lmbda)) ** 0.5) / (b**2 + lmbda)
            )
        )

        gvals_x, gvals_y, gvals_z = g1, g2, g2

    return gvals_x, gvals_y, gvals_z


def _construct_n_matrix_external(x, y, z, a, b, c, lmbda):
    """
    Construct the N matrix for the external field.

    parameters
    ----------
    x, y, z : floats
        A singular observation point in the local coordinate system.

    a, b, c : floats
        Semiaxis lengths of the given ellipsoid.

    lmbda : float
        the given lmbda value for the point we are considering with this
        matrix.


    returns
    -------

    N : matrix
        External points' depolarisation matrix for the given point.

    """

    # g values here are equivalent to the A(lambda) etc values previously.
    # h values as above
    # lambda derivatives as above
    n = np.eye(3)
    r = [x, y, z]
    gvals = _get_g_values_magnetics(a, b, c, lmbda)
    derivs_lmbda = _spatial_deriv_lambda(x, y, z, a, b, c, lmbda)
    h_vals = _get_h_values(a, b, c, lmbda)

    for i in range(len(n)):
        for j in range(len(n[0])):
            if i == j:
                n[i][j] = (-a * b * c / 2) * (
                    derivs_lmbda[i] * h_vals[i] * r[i] + gvals[i]
                )
            else:
                n[i][j] = (-a * b * c / 2) * (derivs_lmbda[i] * h_vals[j] * r[j])

    return n
