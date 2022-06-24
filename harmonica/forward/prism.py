# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Forward modelling for prisms
"""
import numpy as np
from numba import jit, prange

# Attempt to import numba_progress
try:
    from numba_progress import ProgressBar
except ImportError:
    ProgressBar = None

from ..constants import GRAVITATIONAL_CONST


def prism_gravity(
    coordinates,
    prisms,
    density,
    field,
    parallel=True,
    dtype="float64",
    progressbar=False,
    disable_checks=False,
):
    """
    Gravitational fields of right-rectangular prisms in Cartesian coordinates

    The gravitational fields are computed through the analytical solutions
    given by [Nagy2000]_ and [Nagy2002]_, which are valid on the entire domain.
    This means that the computation can be done at any point, either outside or
    inside the prism.

    This implementation makes use of the modified arctangent function proposed
    by [Fukushima2020]_ (eq. 12) so that the potential field to satisfies
    Poisson's equation in the entire domain. Moreover, the logarithm function
    was also modified in order to solve the singularities that the analytical
    solution has on some points (see [Nagy2000]_).

    .. warning::
        The **z direction points upwards**, i.e. positive and negative values
        of ``upward`` represent points above and below the surface,
        respectively. But remember that the ``g_z`` field returns the downward
        component of the gravitational acceleration so that positive density
        contrasts produce positive anomalies.

    Parameters
    ----------
    coordinates : list or 1d-array
        List or array containing ``easting``, ``northing`` and ``upward`` of
        the computation points defined on a Cartesian coordinate system.
        All coordinates should be in meters.
    prisms : list, 1d-array, or 2d-array
        List or array containing the coordinates of the prism(s) in the
        following order:
        west, east, south, north, bottom, top in a Cartesian coordinate system.
        All coordinates should be in meters. Coordinates for more than one
        prism can be provided. In this case, *prisms* should be a list of lists
        or 2d-array (with one prism per line).
    density : list or array
        List or array containing the density of each prism in kg/m^3.
    field : str
        Gravitational field that wants to be computed.
        The available fields are:

        - Gravitational potential: ``potential``
        - Downward acceleration: ``g_z``

    parallel : bool (optional)
        If True the computations will run in parallel using Numba built-in
        parallelization. If False, the forward model will run on a single core.
        Might be useful to disable parallelization if the forward model is run
        by an already parallelized workflow. Default to True.
    dtype : data-type (optional)
        Data type assigned to the resulting gravitational field. Default to
        ``np.float64``.
    progressbar : bool (optional)
        If True, a progress bar of the computation will be printed to standard
        error (stderr). Requires :mod:`numba_progress` to be installed.
        Default to ``False``.
    disable_checks : bool (optional)
        Flag that controls whether to perform a sanity check on the model.
        Should be set to ``True`` only when it is certain that the input model
        is valid and it does not need to be checked.
        Default to ``False``.

    Returns
    -------
    result : array
        Gravitational field generated by the prisms on the computation points.

    Examples
    --------

    Compute gravitational effect of a single a prism

    >>> # Define prisms boundaries, it must be beneath the surface
    >>> prism = [-34, 5, -18, 14, -345, -146]
    >>> # Set prism density to 2670 kg/m³
    >>> density = 2670
    >>> # Define three computation points along the easting axe at 30m above
    >>> # the surface
    >>> coordinates = ([-40, 0, 40], [0, 0, 0], [30, 30, 30])
    >>> # Compute the downward component of the gravitational acceleration that
    >>> # the prism generates on the computation points
    >>> gz = prism_gravity(coordinates, prism, density, field="g_z")
    >>> print("({:.5f}, {:.5f}, {:.5f})".format(*gz))
    (0.06551, 0.06628, 0.06173)

    Define two prisms with positive and negative density contrasts

    >>> prisms = [[-134, -5, -45, 45, -200, -50], [5, 134, -45, 45, -180, -30]]
    >>> densities = [-300, 300]
    >>> # Compute the g_z that the prisms generate on the computation points
    >>> gz = prism_gravity(coordinates, prisms, densities, field="g_z")
    >>> print("({:.5f}, {:.5f}, {:.5f})".format(*gz))
    (-0.05379, 0.02908, 0.11235)

    """
    kernels = {"potential": kernel_potential, "g_z": kernel_g_z}
    if field not in kernels:
        raise ValueError("Gravitational field {} not recognized".format(field))
    # Figure out the shape and size of the output array
    cast = np.broadcast(*coordinates[:3])
    result = np.zeros(cast.size, dtype=dtype)
    # Convert coordinates, prisms and density to arrays with proper shape
    coordinates = tuple(np.atleast_1d(i).ravel() for i in coordinates[:3])
    prisms = np.atleast_2d(prisms)
    density = np.atleast_1d(density).ravel()
    # Sanity checks
    if not disable_checks:
        if density.size != prisms.shape[0]:
            raise ValueError(
                "Number of elements in density ({}) ".format(density.size)
                + "mismatch the number of prisms ({})".format(prisms.shape[0])
            )
        _check_prisms(prisms)
    # Show progress bar for 'jit_prism_gravity' function
    if progressbar:
        if ProgressBar is None:
            raise ImportError(
                "Missing optional dependency 'numba_progress' required if progressbar=True"
            )
        progress_proxy = ProgressBar(total=coordinates[0].size)
    else:
        progress_proxy = None
    # Compute gravitational field
    dispatcher(parallel)(
        coordinates, prisms, density, kernels[field], result, progress_proxy
    )
    result *= GRAVITATIONAL_CONST
    # Close previously created progress bars
    if progressbar:
        progress_proxy.close()
    # Convert to more convenient units
    if field == "g_z":
        result *= 1e5  # SI to mGal
    return result.reshape(cast.shape)


def dispatcher(parallel):
    """
    Return the parallelized or serialized forward modelling function
    """
    dispatchers = {
        True: jit_prism_gravity_parallel,
        False: jit_prism_gravity_serial,
    }
    return dispatchers[parallel]


def _check_prisms(prisms):
    """
    Check if prisms boundaries are well defined

    Parameters
    ----------
    prisms : 2d-array
        Array containing the boundaries of the prisms in the following order:
        ``w``, ``e``, ``s``, ``n``, ``bottom``, ``top``.
        The array must have the following shape: (``n_prisms``, 6), where
        ``n_prisms`` is the total number of prisms.
        This array of prisms must have valid boundaries.
        Run ``_check_prisms`` before.
    """
    west, east, south, north, bottom, top = tuple(prisms[:, i] for i in range(6))
    err_msg = "Invalid prism or prisms. "
    bad_we = west > east
    bad_sn = south > north
    bad_bt = bottom > top
    if bad_we.any():
        err_msg += "The west boundary can't be greater than the east one.\n"
        for prism in prisms[bad_we]:
            err_msg += "\tInvalid prism: {}\n".format(prism)
        raise ValueError(err_msg)
    if bad_sn.any():
        err_msg += "The south boundary can't be greater than the north one.\n"
        for prism in prisms[bad_sn]:
            err_msg += "\tInvalid prism: {}\n".format(prism)
        raise ValueError(err_msg)
    if bad_bt.any():
        err_msg += "The bottom radius boundary can't be greater than the top one.\n"
        for prism in prisms[bad_bt]:
            err_msg += "\tInvalid tesseroid: {}\n".format(prism)
        raise ValueError(err_msg)


def jit_prism_gravity(coordinates, prisms, density, kernel, out, progress_proxy=None):
    """
    Compute gravitational field of prisms on computations points

    Parameters
    ----------
    coordinates : tuple
        Tuple containing ``easting``, ``northing`` and ``upward`` of the
        computation points as arrays, all defined on a Cartesian coordinate
        system and in meters.
    prisms : 2d-array
        Two dimensional array containing the coordinates of the prism(s) in the
        following order: west, east, south, north, bottom, top in a Cartesian
        coordinate system.
        All coordinates should be in meters.
    density : 1d-array
        Array containing the density of each prism in kg/m^3. Must have the
        same size as the number of prisms.
    kernel : func
        Kernel function that will be used to compute the desired field.
    out : 1d-array
        Array where the resulting field values will be stored.
        Must have the same size as the arrays contained on ``coordinates``.
    """
    # Check if we need to update the progressbar on each iteration
    update_progressbar = progress_proxy is not None
    # Iterate over computation points and prisms
    for l in prange(coordinates[0].size):
        for m in range(prisms.shape[0]):
            # Iterate over the prism boundaries to compute the result of the
            # integration (see Nagy et al., 2000)
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        shift_east = prisms[m, 1 - i]
                        shift_north = prisms[m, 3 - j]
                        shift_upward = prisms[m, 5 - k]
                        # If i, j or k is 1, the shift_* will refer to the
                        # lower boundary, meaning the corresponding term should
                        # have a minus sign
                        out[l] += (
                            density[m]
                            * (-1) ** (i + j + k)
                            * kernel(
                                shift_east - coordinates[0][l],
                                shift_north - coordinates[1][l],
                                shift_upward - coordinates[2][l],
                            )
                        )
        # Update progress bar if called
        if update_progressbar:
            progress_proxy.update(1)


@jit(nopython=True)
def kernel_potential(easting, northing, upward):
    """
    Kernel function for potential gravitational field generated by a prism
    """
    radius = np.sqrt(easting**2 + northing**2 + upward**2)
    kernel = (
        easting * northing * safe_log(upward + radius)
        + northing * upward * safe_log(easting + radius)
        + easting * upward * safe_log(northing + radius)
        - 0.5 * easting**2 * safe_atan2(upward * northing, easting * radius)
        - 0.5 * northing**2 * safe_atan2(upward * easting, northing * radius)
        - 0.5 * upward**2 * safe_atan2(easting * northing, upward * radius)
    )
    return kernel


@jit(nopython=True)
def kernel_g_z(easting, northing, upward):
    """
    Kernel for downward component of gravitational acceleration of a prism
    """
    radius = np.sqrt(easting**2 + northing**2 + upward**2)
    kernel = (
        easting * safe_log(northing + radius)
        + northing * safe_log(easting + radius)
        - upward * safe_atan2(easting * northing, upward * radius)
    )
    return kernel


@jit(nopython=True)
def safe_atan2(y, x):
    """
    Principal value of the arctangent expressed as a two variable function

    This modification has to be made to the arctangent function so the
    gravitational field of the prism satisfies the Poisson's equation.
    Therefore, it guarantees that the fields satisfies the symmetry properties
    of the prism. This modified function has been defined according to
    [Fukushima2020]_.
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


# Define jitted versions of the forward modelling function
jit_prism_gravity_serial = jit(nopython=True)(jit_prism_gravity)
jit_prism_gravity_parallel = jit(nopython=True, parallel=True)(jit_prism_gravity)
