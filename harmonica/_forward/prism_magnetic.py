# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Compute magnetic field generated by rectangular prisms
"""
import numpy as np
from choclo.prism import magnetic_e, magnetic_field, magnetic_n, magnetic_u
from numba import jit, prange

from .prism_gravity import _check_prisms
from .utils import initialize_progressbar

VALID_FIELDS = ("b", "b_e", "b_n", "b_u")
FORWARD_FUNCTIONS = {
    "b_e": magnetic_e,
    "b_n": magnetic_n,
    "b_u": magnetic_u,
}


def prism_magnetic(
    coordinates,
    prisms,
    magnetization,
    field,
    parallel=True,
    dtype=np.float64,
    progressbar=False,
    disable_checks=False,
):
    """
    Magnetic field of right-rectangular prisms in Cartesian coordinates

    Parameters
    ----------
    coordinates : list of arrays
        List of arrays containing the ``easting``, ``northing`` and ``upward``
        coordinates of the computation points defined on a Cartesian coordinate
        system. All coordinates should be in meters.
    prisms : list, 1d-array, or 2d-array
        List or array containing the coordinates of the prism(s) in the
        following order:
        west, east, south, north, bottom, top in a Cartesian coordinate system.
        All coordinates should be in meters. Coordinates for more than one
        prism can be provided. In this case, *prisms* should be a list of lists
        or 2d-array (with one prism per row).
    magnetization : tuple or arrays
        Tuple containing the three arrays corresponding to the magnetization
        vector components of each prism in :math:`Am^{-1}`. These arrays should
        be provided in the following order: ``magnetization_e``,
        ``magnetization_n``, ``magnetization_u``.
    field : str
        Magnetic field that will be computed. The available fields are:

        - The full magnetic vector: ``b``
        - Easting component of the magnetic vector: ``b_e``
        - Northing component of the magnetic vector: ``b_n``
        - Upward component of the magnetic vector: ``b_u``

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
    magnetic_field : array or tuple of arrays
        Array with the computed magnetic field on every observation point.
        If ``field`` is set to ``"b"``, then a tuple containing the three
        components of the magnetic vector will be returned in the following
        order: ``b_e``, ``b_n``, ``b_u``.
    """
    # Check if field is valid
    if field not in VALID_FIELDS:
        raise ValueError(
            f"Invalid field '{field}'. "
            f"Please choose one of '{','.join(VALID_FIELDS)}'."
        )
    # Figure out the shape and size of the output array(s)
    cast = np.broadcast(*coordinates[:3])
    # Convert coordinates, prisms and magnetization to arrays with proper shape
    coordinates = tuple(np.atleast_1d(c).ravel() for c in coordinates[:3])
    prisms = np.atleast_2d(prisms)
    magnetization = tuple(np.atleast_1d(m).ravel() for m in magnetization)
    # Sanity checks
    if not disable_checks:
        _run_sanity_checks(prisms, magnetization)
    # Discard null prisms (zero volume or null magnetization)
    prisms, magnetization = _discard_null_prisms(prisms, magnetization)
    # Run computations
    if field == "b":
        result = _prism_magnetic_vector(
            coordinates, prisms, magnetization, cast, parallel, progressbar, dtype
        )
    else:
        forward_func = FORWARD_FUNCTIONS[field]
        result = _prism_single_component(
            coordinates,
            prisms,
            magnetization,
            forward_func,
            cast,
            parallel,
            progressbar,
            dtype,
        )
    return result


def _prism_magnetic_vector(
    coordinates, prisms, magnetization, cast, parallel, progressbar, dtype
):
    """
    Forward model the three components of the magnetic vector

    Parameters
    ----------
    coordinates : tuple of arrays
    prisms : 2d-array
    magnetization : tuple of arrays
    cast : np.broadcast
    parallel : bool
    dtype : np.dtype

    Returns
    -------
    magnetic_components : tuple of arrays
        Tuple containing the three components of the magnetic vector:
        ``b_e``, ``b_n``, ``b_u``.
    """
    # Decide which function should be used
    if parallel:
        jit_func = _jit_prism_magnetic_field_parallel
    else:
        jit_func = _jit_prism_magnetic_field_serial
    # Run forward model
    b_e, b_n, b_u = tuple(np.zeros(cast.size, dtype=dtype) for _ in range(3))
    with initialize_progressbar(coordinates[0].size, progressbar) as progress_proxy:
        jit_func(coordinates, prisms, magnetization, b_e, b_n, b_u, progress_proxy)
    # Convert to nT
    b_e *= 1e9
    b_n *= 1e9
    b_u *= 1e9
    # Cast shape and form the tuple
    result = tuple(component.reshape(cast.shape) for component in (b_e, b_n, b_u))
    return result


def _prism_single_component(
    coordinates, prisms, magnetization, forward_func, cast, parallel, progressbar, dtype
):
    """
    Forward model the three components of the magnetic vector

    Parameters
    ----------
    coordinates : tuple of arrays
    prisms : 2d-array
    magnetization : tuple of arrays
    forward_func : callable
    cast : np.broadcast
    parallel : bool
    dtype : np.dtype

    Returns
    -------
    magnetic_component : arrays
        Array containing the desired magnetic component.
    """
    # Decide which function should be used
    if parallel:
        jit_func = _jit_prism_magnetic_component_parallel
    else:
        jit_func = _jit_prism_magnetic_component_serial
    # Run computations
    result = np.zeros(cast.size, dtype=dtype)
    with initialize_progressbar(coordinates[0].size, progressbar) as progress_proxy:
        jit_func(
            coordinates,
            prisms,
            magnetization,
            result,
            forward_func,
            progress_proxy,
        )
    # Convert to nT
    result *= 1e9
    return result.reshape(cast.shape)


def _jit_prism_magnetic_field(
    coordinates, prisms, magnetization, b_e, b_n, b_u, progress_proxy=None
):
    """
    Compute magnetic fields of prisms on computation points

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
    magnetization : tuple of arrays
        Tuple containing the three arrays corresponding to the magnetization
        vector components of each prism in :math:`Am^{-1}`. These arrays should
        be provided in the following order: ``magnetization_e``,
        ``magnetization_n``, ``magnetization_u``.
    b_e : 1d-array
        Array where the resulting values of the easting component of the
        magnetic field will be stored.
    b_n : 1d-array
        Array where the resulting values of the northing component of the
        magnetic field will be stored.
    b_u : 1d-array
        Array where the resulting values of the upward component of the
        magnetic field will be stored.
    progress_proxy : :class:`numba_progress.ProgressBar` or None
        Instance of :class:`numba_progress.ProgressBar` that gets updated after
        each iteration on the observation points. Use None if no progress bar
        is should be used.
    """
    # Check if we need to update the progressbar on each iteration
    update_progressbar = progress_proxy is not None
    # Iterate over computation points and prisms
    for l in prange(coordinates[0].size):
        for m in range(prisms.shape[0]):
            easting_comp, northing_comp, upward_comp = magnetic_field(
                coordinates[0][l],
                coordinates[1][l],
                coordinates[2][l],
                prisms[m, 0],
                prisms[m, 1],
                prisms[m, 2],
                prisms[m, 3],
                prisms[m, 4],
                prisms[m, 5],
                magnetization[0][m],
                magnetization[1][m],
                magnetization[2][m],
            )
            b_e[l] += easting_comp
            b_n[l] += northing_comp
            b_u[l] += upward_comp
        # Update progress bar if called
        if update_progressbar:
            progress_proxy.update(1)


def _jit_prism_magnetic_component(
    coordinates, prisms, magnetization, result, forward_function, progress_proxy=None
):
    """
    Compute a single component of the magnetic field of prisms

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
    magnetization : tuple of arrays
        Tuple containing the three arrays corresponding to the magnetization
        vector components of each prism in :math:`Am^{-1}`. These arrays should
        be provided in the following order: ``magnetization_e``,
        ``magnetization_n``, ``magnetization_u``.
    result : 1d-array
        Array where the resulting values of the desired component of the
        magnetic field will be stored.
    forward_function : callable
        Forward function to be used to compute the desired component of the
        magnetic field. Choose one of :func:`choclo.prism.magnetic_easting`,
        :func:`choclo.prism.magnetic_northing` or
        :func:`choclo.prism.magnetic_upward`.
    progress_proxy : :class:`numba_progress.ProgressBar` or None
        Instance of :class:`numba_progress.ProgressBar` that gets updated after
        each iteration on the observation points. Use None if no progress bar
        is should be used.
    """
    # Check if we need to update the progressbar on each iteration
    update_progressbar = progress_proxy is not None
    # Iterate over computation points and prisms
    for l in prange(coordinates[0].size):
        for m in range(prisms.shape[0]):
            result[l] += forward_function(
                coordinates[0][l],
                coordinates[1][l],
                coordinates[2][l],
                prisms[m, 0],
                prisms[m, 1],
                prisms[m, 2],
                prisms[m, 3],
                prisms[m, 4],
                prisms[m, 5],
                magnetization[0][m],
                magnetization[1][m],
                magnetization[2][m],
            )
        # Update progress bar if called
        if update_progressbar:
            progress_proxy.update(1)


def _discard_null_prisms(prisms, magnetization):
    """
    Discard prisms with zero volume or null magnetization

    Parameters
    ----------
    prisms : 2d-array
        Array containing the boundaries of the prisms in the following order:
        ``w``, ``e``, ``s``, ``n``, ``bottom``, ``top``.
        The array must have the following shape: (``n_prisms``, 6), where
        ``n_prisms`` is the total number of prisms.
        This array of prisms must have valid boundaries.
        Run ``_check_prisms`` before.
    magnetization : 2d-array
        Array containing the magnetization vector of each prism in
        :math:`Am^{-1}`. Each vector will be a row in the 2d-array.

    Returns
    -------
    prisms : 2d-array
        A copy of the ``prisms`` array that doesn't include the null prisms
        (prisms with zero volume or zero density).
    magnetization : 2d-array
        A copy of the ``magnetization`` array that doesn't include the
        magnetization vectors for null prisms (prisms with zero volume or
        null magnetization).
    """
    west, east, south, north, bottom, top = tuple(prisms[:, i] for i in range(6))
    # Mark prisms with zero volume as null prisms
    null_prisms = (west == east) | (south == north) | (bottom == top)
    # Mark prisms with null magnetization as null prisms
    mag_e, mag_n, mag_u = magnetization
    null_mag = (mag_e == 0) & (mag_n == 0) & (mag_u == 0)
    null_prisms[null_mag] = True
    # Keep only non null prisms
    prisms = prisms[np.logical_not(null_prisms), :]
    magnetization = tuple(m[np.logical_not(null_prisms)] for m in magnetization)
    return prisms, magnetization


def _run_sanity_checks(prisms, magnetization):
    """
    Run sanity checks on prisms and their magnetization
    """
    if (size := len(magnetization)) != 3:
        raise ValueError(
            f"Invalid magnetization vectors with '{size}' elements. "
            + "Magnetization vectors should have only 3 elements."
        )
    if magnetization[0].size != prisms.shape[0]:
        raise ValueError(
            f"Number of magnetization vectors ({magnetization[0].size}) "
            + f"mismatch the number of prisms ({prisms.shape[0]})"
        )
    _check_prisms(prisms)


# Define jitted versions of the forward modelling function
_jit_prism_magnetic_field_serial = jit(nopython=True)(_jit_prism_magnetic_field)
_jit_prism_magnetic_field_parallel = jit(nopython=True, parallel=True)(
    _jit_prism_magnetic_field
)
_jit_prism_magnetic_component_serial = jit(nopython=True)(_jit_prism_magnetic_component)
_jit_prism_magnetic_component_parallel = jit(nopython=True, parallel=True)(
    _jit_prism_magnetic_component
)
