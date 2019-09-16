"""
Forward modelling for prisms
"""
import numpy as np
from numba import jit

from ..constants import GRAVITATIONAL_CONST


def prism_gravity(coordinates, prisms, density, field, dtype="float64"):
    """
    Compute gravitational fields of right-rectangular prisms in Cartesian coordinates.

    The gravitational fields are computed through the analytical solutions given by
    [Nagy2000]_ and [Nagy2002]_, which are valid on the entire domain. This means that
    the computation can be done at any point, either outside or inside the prism.

    This implementation makes use of the modified arctangent function proposed by
    [Fukushima2019]_ (eq. 12) so that the potential field to satisfies Poisson's
    equation in the entire domain. Moreover, the logarithm function was also modified in order to solve the
    singularities that the analytical solution has on some points (see [Nagy2000]_).

    .. warning::
        The **z direction points upwards**, i.e. positive and negative values of
        ``upward`` represent points above and below the surface, respectively. But
        remember that the ``g_z`` field returns the downward component of the gravity
        acceleration so that positive density contrasts produce positive anomalies.

    Parameters
    ----------
    coordinates : list or 1d-array
        List or array containing ``easting``, ``northing`` and ``upward`` of the
        computation points defined on a Cartesian coordinate system. All coordinates
        should be in meters.
    prisms : list, 1d-array, or 2d-array
        List or array containing the coordinates of the prism(s) in the following order:
       west, east, south, north, bottom, top in a Cartesian coordinate
        system. All coordinates should be in meters. 
        Coordinates for more than one prism can be provided. In this case, *prisms*
        should be a list of lists or 2d-array (with one prism per line).
    density : list or array
        List or array containing the density of each prism in kg/m^3.
    field : str
        Gravitational field that wants to be computed.
        The available fields are:

        - Gravitational potential: ``potential``
        - Downward acceleration: ``g_z``

    dtype : data-type (optional)
        Data type assigned to prism boundaries, computation points coordinates and
        resulting gravitational field. Default to ``np.float64``.

    Returns
    -------
    result : array
        Gravitational field generated by the prisms on the computation points.

    Examples
    --------

    Compute a single a prism located beneath the surface with density of 2670 kg/m³:
    
    >>> prism = [105, 155, 46, 104, -345, -146]
    >>> density = 2670
    >>> # Define a computation point above its center, at 30 meters above the surface
    >>> coordinates = (130, 75, 30)
    >>> # Compute the downward component that the prism generates on the computation point
    >>> prism_gravity(coordinates, prism, density, field="g_z")
    array(0.15375061)

    >>> # Define two prism, one with positive and the other one with negative density
    >>> prisms = [[-134, -5, -45, 45, -200, -50], [5, 134, -45, 45, -180, -30]]
    >>> densities = [-300, 300]
    >>> # Define three computation points along the easting axe
    >>> coordinates = ([-40, 0, 40], [0, 0, 0], [0, 0, 0])
    >>> # Compute the g_z that the prisms generate
    >>> prism_gravity(coordinates, prisms, densities, field="g_z")
    array([-0.11785066,  0.04643262,  0.21866826])

    """
    kernels = {"potential": kernel_potential, "g_z": kernel_g_z}
    if field not in kernels:
        raise ValueError("Gravity field {} not recognized".format(field))
    # Figure out the shape and size of the output array
    cast = np.broadcast(*coordinates[:3])
    result = np.zeros(cast.size, dtype=dtype)
    # Convert coordinates, tesseroids and density to arrays
    easting, northing, upward = (
        np.atleast_1d(i).ravel().astype(dtype) for i in coordinates[:3]
    )
    coordinates = np.vstack((easting, northing, upward))
    prisms = np.atleast_2d(prisms).astype(dtype)
    density = np.atleast_1d(density).ravel().astype(dtype)
    # Sanity checks
    if density.size != prisms.shape[0]:
        raise ValueError("Density array must have the same size as number of prisms.")
    _check_prisms(prisms)
    # Compute gravitational field
    jit_prism_gravity(coordinates, prisms, density, kernels[field], result)
    result *= GRAVITATIONAL_CONST
    # Convert to more convenient units
    if field == "g_z":
        result *= 1e5  # SI to mGal
    return result.reshape(cast.shape)


def _check_prisms(prisms):
    """
    Check if prisms boundaries are well defined

    Parameters
    ----------
    coordinates : 2d-array
        Array containing the coordinates of the computation points in the following
        order: ``easting``, ``northing`` and ``upward``.
        The array must have the following shape: (3, ``n_points``), where
        ``n_points`` is the total number of computation points.
    prisms : 2d-array
        Array containing the boundaries of the prisms in the following order:
        ``w``, ``e``, ``s``, ``n``, ``bottom``, ``top``.
        The array must have the following shape: (``n_prisms``, 6), where
        ``n_prisms`` is the total number of prisms.
        This array of prisms must have valid boundaries. Run ``_check_prisms`` before.
    """
    west, east, south, north, bottom, top = tuple(prisms[:, i] for i in range(6))
    err_msg = "Invalid prism or prisms. "
    if (west > east).any():
        err_msg += "The west boundary can't be greater than the east one.\n"
        for prism in prisms[west > east]:
            err_msg += "\tInvalid prism: {}\n".format(prism)
        raise ValueError(err_msg)
    if (south > north).any():
        err_msg += "The south boundary can't be greater than the north one.\n"
        for prism in prisms[south > north]:
            err_msg += "\tInvalid prism: {}\n".format(prism)
        raise ValueError(err_msg)
    if (bottom > top).any():
        err_msg += "The bottom radius boundary can't be greater than the top one.\n"
        for prism in prisms[bottom > top]:
            err_msg += "\tInvalid tesseroid: {}\n".format(prism)
        raise ValueError(err_msg)


@jit(nopython=True)
def jit_prism_gravity(
    coordinates, prisms, density, kernel, out
):  # pylint: disable=invalid-name
    """
    Compute gravitational field of prisms on computations points
    """
    # Iterate over computation points and prisms
    for l in range(coordinates.shape[1]):
        for m in range(prisms.shape[0]):
            # Iterate over the prism boundaries to compute the result of the
            # integration (see Nagy et al., 2000)
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        shift_east = prisms[m, 1 - i]
                        shift_north = prisms[m, 3 - j]
                        shift_upward = prisms[m, 5 - k]
                        # If i, j or k is 1, the shift_* will refer to the lower
                        # boundary, meaning the corresponding term should have a minus
                        # sign
                        out[l] += (
                            density[m]
                            * (-1) ** (i + j + k)
                            * kernel(
                                shift_east - coordinates[0, l],
                                shift_north - coordinates[1, l],
                                shift_upward - coordinates[2, l],
                            )
                        )


@jit(nopython=True)
def kernel_potential(easting, northing, upward):
    """
    Kernel function for potential gravity field generated by a prism
    """
    radius = np.sqrt(easting ** 2 + northing ** 2 + upward ** 2)
    kernel = (
        easting * northing * safe_log(upward + radius)
        + northing * upward * safe_log(easting + radius)
        + easting * upward * safe_log(northing + radius)
        - 0.5 * easting ** 2 * safe_atan2(upward * northing, easting * radius)
        - 0.5 * northing ** 2 * safe_atan2(upward * easting, northing * radius)
        - 0.5 * upward ** 2 * safe_atan2(easting * northing, upward * radius)
    )
    return kernel


@jit(nopython=True)
def kernel_g_z(easting, northing, upward):
    """
    Kernel function for downward component of gravity acceleration generated by a prism
    """
    radius = np.sqrt(easting ** 2 + northing ** 2 + upward ** 2)
    kernel = (
        easting * safe_log(northing + radius)
        + northing * safe_log(easting + radius)
        - upward * safe_atan2(easting * northing, upward * radius)
    )
    return kernel


@jit(nopython=True)
def safe_atan2(y, x):
    """
    Return the principal value of the arctangent expressed as a two variable function

    This modification has to be made to the arctangent function so the gravitational
    field of the prism satisfies the Poisson's equation. Therefore, it guarantees that
    the fields satisfies the symmetry properties of the prism. This modified function
    has been defined according to [Fukushima2019]_.
    """
    if x != 0:
        result = np.arctan(y / x)
    else:
        if y > 0:
            result = np.pi / 2
        elif y < 0:
            result = -np.pi / 2
        else:
            result = 0
    return result


@jit(nopython=True)
def safe_log(x):
    """
    Modified log to return 0 for log(0).
    The limits in the formula terms tend to 0 (see [Nagy2000]_).
    """
    if np.abs(x) < 1e-10:
        result = 0
    else:
        result = np.log(x)
    return result
