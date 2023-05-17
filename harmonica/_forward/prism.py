# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Forward modelling for prisms
"""
import warnings

import numpy as np
from choclo.prism import (
    gravity_e,
    gravity_ee,
    gravity_en,
    gravity_eu,
    gravity_n,
    gravity_nn,
    gravity_nu,
    gravity_pot,
    gravity_u,
    gravity_uu,
)
from choclo.prism._utils import (
    is_point_on_easting_edge,
    is_point_on_northing_edge,
    is_point_on_upward_edge,
)
from numba import jit, prange

# Define dictionary with available gravity fields for prisms
FIELDS = {
    "potential": gravity_pot,
    "g_e": gravity_e,
    "g_n": gravity_n,
    "g_z": gravity_u,
    "g_ee": gravity_ee,
    "g_nn": gravity_nn,
    "g_zz": gravity_uu,
    "g_en": gravity_en,
    "g_ez": gravity_eu,
    "g_nz": gravity_nu,
}

# Attempt to import numba_progress
try:
    from numba_progress import ProgressBar
except ImportError:
    ProgressBar = None


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

    Compute the gravitational potential, gravitational acceleration and tensor
    components generated by a collection of prisms on a set of observation
    points.

    .. warning::
        The **vertical direction points upwards**, i.e. positive and negative
        values of ``upward`` represent points above and below the surface,
        respectively. But ``g_z`` field returns the **downward component** of
        the gravitational acceleration so that positive density contrasts
        produce positive anomalies. The same applies to the tensor components,
        i.e. the ``g_ez`` is the non-diagonal **easting-downward** tensor
        component.

    Parameters
    ----------
    coordinates : list of arrays
        List of arrays containing the ``easting``, ``northing`` and ``upward``
        coordinates of the computation points defined on a Cartesian coordinate
        system. All coordinates should be in meters.
    prisms : list, 1d-array, or 2d-array
        List or array containing the coordinates of the prism(s) in the
        following order:
        west, east, south, north, bottom, top, in a Cartesian coordinate system.
        All coordinates should be in meters. Coordinates for more than one
        prism can be provided. In this case, *prisms* should be a list of lists
        or 2d-array (with one prism per row).
    density : list or array
        List or array containing the density of each prism in kg/m^3.
    field : str
        Gravitational field that wants to be computed.
        The available fields are:

        - Gravitational potential: ``potential``
        - Eastward acceleration: ``g_e``
        - Northward acceleration: ``g_n``
        - Downward acceleration: ``g_z``
        - Diagonal tensor components: ``g_ee``, ``g_nn``, ``g_zz``
        - Non-diagonal tensor components: ``g_en``, ``g_ez``, ``g_nz``

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

    Notes
    -----
    This function makes use of :mod:`choclo.prism` forward modelling functions
    to compute each gravitational field.

    The gravitational potential (``"potential"``) and the acceleration
    components (``"g_e"``, ``"g_n"`` and ``"g_z"``) are well defined on the
    entire domain.
    Tensor components aren't defined on prism vertices.
    Diagonal tensor components aren't defined on edges normal to the direction
    of the tensor (e.g. `"g_ee"` is not defined on edges parallel to northing
    and upward directions).
    Non-diagonal tensor components aren't defined on edges normal to the
    remaining direction of the tensor (e.g. `"g_en"` is not defined on edges
    parallel to the upward direction).
    The function returns :func:`numpy.nan` on every singular point.

    The diagonal tensor components aren't defined on observation points that
    belong to the faces normal to their direction (e.g. ``"g_zz"`` is not
    define on horizontal faces): two different limits exist when approaching
    from the inside and from the outside of the prism.
    This functions returns the limit of these components while approaching from
    the outside.

    References
    ----------
    * [Nagy2000]_
    * [Nagy2002]_
    * [Fukushima2020]_

    Examples
    --------

    Compute gravitational effect of a single a prism

    >>> # Define prisms boundaries, it must be beneath the surface
    >>> prism = [-34, 5, -18, 14, -345, -146]
    >>> # Set prism density to 2670 kg/m³
    >>> density = 2670
    >>> # Define three computation points along the easting direction at 30m
    >>> # above the surface
    >>> easting = [-40, 0, 40]
    >>> northing = [0, 0, 0]
    >>> upward = [30, 30, 30]
    >>> coordinates = (easting, northing, upward)
    >>> # Compute the downward component of the gravitational acceleration that
    >>> # the prism generates on the computation points
    >>> gz = prism_gravity(coordinates, prism, density, field="g_z")
    >>> print("({:.5f}, {:.5f}, {:.5f})".format(*gz))
    (0.06552, 0.06629, 0.06174)

    Define two prisms with positive and negative density contrasts

    >>> prisms = [[-134, -5, -45, 45, -200, -50], [5, 134, -45, 45, -180, -30]]
    >>> densities = [-300, 300]
    >>> # Compute the g_z that the prisms generate on the computation points
    >>> gz = prism_gravity(coordinates, prisms, densities, field="g_z")
    >>> print("({:.5f}, {:.5f}, {:.5f})".format(*gz))
    (-0.05380, 0.02908, 0.11237)

    """
    if field not in FIELDS:
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
        _check_singular_points(coordinates, prisms, field)
    # Discard null prisms (zero volume or zero density)
    prisms, density = _discard_null_prisms(prisms, density)
    # Show progress bar for 'jit_prism_gravity' function
    if progressbar:
        if ProgressBar is None:
            raise ImportError(
                "Missing optional dependency 'numba_progress' required "
                "if progressbar=True"
            )
        progress_proxy = ProgressBar(total=coordinates[0].size)
    else:
        progress_proxy = None
    # Choose parallelized or serialized forward function
    if parallel:
        gravity_prism_func = jit_prism_gravity_parallel
    else:
        gravity_prism_func = jit_prism_gravity_serial
    # Compute gravitational field
    gravity_prism_func(
        coordinates, prisms, density, FIELDS[field], result, progress_proxy
    )
    # Close previously created progress bars
    if progress_proxy:
        progress_proxy.close()
    # Invert sign of gravity_u, gravity_eu, gravity_nu
    if field in ("g_z", "g_ez", "g_nz"):
        result *= -1
    # Convert to more convenient units
    if field in ("g_e", "g_n", "g_z"):
        result *= 1e5  # SI to mGal
    # Convert to more convenient units
    if field in ("g_ee", "g_nn", "g_zz", "g_en", "g_ez", "g_nz"):
        result *= 1e9  # SI to Eotvos
    return result.reshape(cast.shape)


def _check_singular_points(coordinates, prisms, field):
    """
    Check if any observation point is a singular point for tensor components

    The analytic solutions for the tensor components of the prism have some
    singular points:

      - All prism vertices are singular points for every tensor component.
      - Diagonal components aren't defined on edges perpendicular to the
        component direction (e.g. ``g_ee`` is not defined on edges parallel to
        northing and upward directions).
      - Non-diagonal components aren't defined on edges perpendicular to the
        two directions of the component (e.g. ``g_en`` is not defined on edges
        parallel to the upward direction).
    """
    functions = {
        "g_ee": _any_singular_point_g_ee,
        "g_nn": _any_singular_point_g_nn,
        "g_zz": _any_singular_point_g_zz,
        "g_en": _any_singular_point_g_en,
        "g_ez": _any_singular_point_g_ez,
        "g_nz": _any_singular_point_g_nz,
    }
    if field not in functions:
        return None
    if functions[field](coordinates, prisms):
        warnings.warn(
            "Found observation point on singular point of a prism.", UserWarning
        )


def _any_singular_point_g_ee(coordinates, prisms):
    """
    Check observation points as singular points of g_ee
    """
    easting, northing, upward = coordinates
    n_coords = easting.size
    n_prisms = prisms.shape[0]
    for l in range(n_coords):
        for m in range(n_prisms):
            if is_point_on_northing_edge(
                easting[l], northing[l], upward[l], *prisms[m, :]
            ) or is_point_on_upward_edge(
                easting[l], northing[l], upward[l], *prisms[m, :]
            ):
                return True
    return False


def _any_singular_point_g_nn(coordinates, prisms):
    """
    Check observation points as singular points of g_nn
    """
    easting, northing, upward = coordinates
    n_coords = easting.size
    n_prisms = prisms.shape[0]
    for l in range(n_coords):
        for m in range(n_prisms):
            if is_point_on_easting_edge(
                easting[l], northing[l], upward[l], *prisms[m, :]
            ) or is_point_on_upward_edge(
                easting[l], northing[l], upward[l], *prisms[m, :]
            ):
                return True
    return False


def _any_singular_point_g_zz(coordinates, prisms):
    """
    Check observation points as singular points of g_zz
    """
    easting, northing, upward = coordinates
    n_coords = easting.size
    n_prisms = prisms.shape[0]
    for l in range(n_coords):
        for m in range(n_prisms):
            if is_point_on_easting_edge(
                easting[l], northing[l], upward[l], *prisms[m, :]
            ) or is_point_on_northing_edge(
                easting[l], northing[l], upward[l], *prisms[m, :]
            ):
                return True
    return False


def _any_singular_point_g_en(coordinates, prisms):
    """
    Check observation points as singular points of g_en
    """
    easting, northing, upward = coordinates
    n_coords = easting.size
    n_prisms = prisms.shape[0]
    for l in range(n_coords):
        for m in range(n_prisms):
            if is_point_on_upward_edge(
                easting[l], northing[l], upward[l], *prisms[m, :]
            ):
                return True
    return False


def _any_singular_point_g_ez(coordinates, prisms):
    """
    Check observation points as singular points of g_ez
    """
    easting, northing, upward = coordinates
    n_coords = easting.size
    n_prisms = prisms.shape[0]
    for l in range(n_coords):
        for m in range(n_prisms):
            if is_point_on_northing_edge(
                easting[l], northing[l], upward[l], *prisms[m, :]
            ):
                return True
    return False


def _any_singular_point_g_nz(coordinates, prisms):
    """
    Check observation points as singular points of g_nz
    """
    easting, northing, upward = coordinates
    n_coords = easting.size
    n_prisms = prisms.shape[0]
    for l in range(n_coords):
        for m in range(n_prisms):
            if is_point_on_easting_edge(
                easting[l], northing[l], upward[l], *prisms[m, :]
            ):
                return True
    return False


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


def _discard_null_prisms(prisms, density):
    """
    Discard prisms with zero volume or zero density

    Parameters
    ----------
    prisms : 2d-array
        Array containing the boundaries of the prisms in the following order:
        ``w``, ``e``, ``s``, ``n``, ``bottom``, ``top``.
        The array must have the following shape: (``n_prisms``, 6), where
        ``n_prisms`` is the total number of prisms.
        This array of prisms must have valid boundaries.
        Run ``_check_prisms`` before.
    density : 1d-array
        Array containing the density of each prism in kg/m^3. Must have the
        same size as the number of prisms.

    Returns
    -------
    prisms : 2d-array
        A copy of the ``prisms`` array that doesn't include the null prisms
        (prisms with zero volume or zero density).
    density : 1d-array
        A copy of the ``density`` array that doesn't include the density values
        for null prisms (prisms with zero volume or zero density).
    """
    west, east, south, north, bottom, top = tuple(prisms[:, i] for i in range(6))
    # Mark prisms with zero volume as null prisms
    null_prisms = (west == east) | (south == north) | (bottom == top)
    # Mark prisms with zero density as null prisms
    null_prisms[density == 0] = True
    # Keep only non null prisms
    prisms = prisms[np.logical_not(null_prisms), :]
    density = density[np.logical_not(null_prisms)]
    return prisms, density


def jit_prism_gravity(
    coordinates, prisms, density, forward_func, out, progress_proxy=None
):
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
    forward_func : func
        Forward modelling function that will be used to compute the desired
        field. It could be one of the forward modelling functions in
        :mod:`choclo.prism`.
    out : 1d-array
        Array where the resulting field values will be stored.
        Must have the same size as the arrays contained on ``coordinates``.
    """
    # Unpack coordinates
    easting, northing, upward = coordinates
    # Check if we need to update the progressbar on each iteration
    update_progressbar = progress_proxy is not None
    # Iterate over computation points and prisms
    for l in prange(easting.size):
        for m in range(prisms.shape[0]):
            out[l] += forward_func(
                easting[l],
                northing[l],
                upward[l],
                prisms[m, 0],
                prisms[m, 1],
                prisms[m, 2],
                prisms[m, 3],
                prisms[m, 4],
                prisms[m, 5],
                density[m],
            )
        # Update progress bar if called
        if update_progressbar:
            progress_proxy.update(1)


# Define jitted versions of the forward modelling function
jit_prism_gravity_serial = jit(nopython=True)(jit_prism_gravity)
jit_prism_gravity_parallel = jit(nopython=True, parallel=True)(jit_prism_gravity)
