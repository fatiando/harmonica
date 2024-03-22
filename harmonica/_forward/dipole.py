# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Forward modelling for magnetic fields of dipoles
"""
from math import prod

import numpy as np
from choclo.dipole import magnetic_e, magnetic_field, magnetic_n, magnetic_u
from numba import jit, prange

from .utils import initialize_progressbar

VALID_FIELDS = ("b", "b_e", "b_n", "b_u")
FORWARD_FUNCTIONS = {
    "b_e": magnetic_e,
    "b_n": magnetic_n,
    "b_u": magnetic_u,
}


def dipole_magnetic(
    coordinates,
    dipoles,
    magnetic_moments,
    field,
    parallel=True,
    dtype="float64",
    progressbar=False,
    disable_checks=False,
):
    """
    Magnetic field of dipoles in Cartesian coordinates

    Parameters
    ----------
    coordinates : list of arrays
        List of arrays containing the ``easting``, ``northing`` and ``upward``
        coordinates of the computation points defined on a Cartesian coordinate
        system. All coordinates should be in meters.
    dipoles : tuple of arrays
        Tuple of arrays containing the ``easting``, ``northing`` and ``upward``
        locations of the dipoles defined on a Cartesian coordinate system. All
        coordinates should be in meters.
    magnetic_moments : tuple of arrays
        Tuple containing the three arrays corresponding to the magnetic moment
        components of each dipole in :math:`Am^2`. These arrays should
        be provided in the following order: ``mag_moment_easting``,
        ``mag_moment_northing``, ``mag_moment_upward``.
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
        Data type assigned to the resulting magnetic field. Default to
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
            f"Please choose one of '{', '.join(VALID_FIELDS)}'."
        )
    # Figure out the shape and size of the output array(s)
    cast = np.broadcast(*coordinates[:3])
    # Convert coordinates, dipoles and magnetic moments to arrays
    coordinates = tuple(np.atleast_1d(i).ravel() for i in coordinates[:3])
    dipoles = tuple(np.atleast_1d(i).ravel() for i in dipoles[:3])
    magnetic_moments = tuple(np.atleast_1d(m).ravel() for m in magnetic_moments)
    # Sanity checks
    if not disable_checks:
        _check_dipoles_and_magnetic_moments(dipoles, magnetic_moments)
    # Run computations
    if field == "b":
        result = _dipole_magnetic_vector(
            coordinates,
            dipoles,
            magnetic_moments,
            cast.shape,
            dtype,
            parallel,
            progressbar,
        )
    else:
        forward_func = FORWARD_FUNCTIONS[field]
        result = _dipole_magnetic_component(
            coordinates,
            dipoles,
            magnetic_moments,
            forward_func,
            cast.shape,
            dtype,
            parallel,
            progressbar,
        )
    return result


def _dipole_magnetic_vector(
    coordinates,
    dipoles,
    magnetic_moments,
    shape,
    dtype,
    parallel,
    progressbar,
):
    """
    Forward model the three components of the magnetic vector

    Parameters
    ----------
    coordinates : tuple of arrays
        Tuple containing ``easting``, ``northing`` and ``upward`` of the
        computation points as arrays, all defined on a Cartesian coordinate
        system and in meters.
    dipoles : tuple of arrays
        Tuple of arrays containing the ``easting``, ``northing`` and ``upward``
        locations of the dipoles defined on a Cartesian coordinate system. All
        coordinates should be in meters.
    magnetic_moments : tuple of arrays
        Tuple containing the three arrays corresponding to the magnetic moment
        components of each dipole in :math:`Am^2`. These arrays should
        be provided in the following order: ``mag_moment_easting``,
        ``mag_moment_northing``, ``mag_moment_upward``.
    shape : tuple of int
        Shape of the expected output arrays.
    dtype : np.dtype
        Data type of the expected output arrays.
    parallel : bool
        If True, the forward modelling will be run in parallel. If False, it
        will be run in a single thread.
    progressbar : bool
        If True, a progress bar of the computation will be printed to standard
        error (stderr). Requires :mod:`numba_progress` to be installed.

    Returns
    -------
    magnetic_components : tuple of arrays
        Tuple containing the three components of the magnetic vector in
        :math:`nT`: ``b_e``, ``b_n``, ``b_u``.
    """
    # Decide which function should be used
    if parallel:
        jit_func = _jit_dipole_magnetic_field_cartesian_parallel
    else:
        jit_func = _jit_dipole_magnetic_field_cartesian_serial
    # Run forward model
    size = prod(shape)
    b_e, b_n, b_u = tuple(np.zeros(size, dtype=dtype) for _ in range(3))
    with initialize_progressbar(coordinates[0].size, progressbar) as progress_proxy:
        jit_func(coordinates, dipoles, magnetic_moments, b_e, b_n, b_u, progress_proxy)
    # Convert to nT
    b_e *= 1e9
    b_n *= 1e9
    b_u *= 1e9
    # Cast shape and form the tuple
    result = tuple(component.reshape(shape) for component in (b_e, b_n, b_u))
    return result


def _dipole_magnetic_component(
    coordinates,
    dipoles,
    magnetic_moments,
    forward_func,
    shape,
    dtype,
    parallel,
    progressbar,
):
    """
    Forward model a single component of the magnetic vector

    Parameters
    ----------
    coordinates : tuple of arrays
        Tuple containing ``easting``, ``northing`` and ``upward`` of the
        computation points as arrays, all defined on a Cartesian coordinate
        system and in meters.
    dipoles : tuple of arrays
        Tuple of arrays containing the ``easting``, ``northing`` and ``upward``
        locations of the dipoles defined on a Cartesian coordinate system. All
        coordinates should be in meters.
    magnetic_moments : tuple of arrays
        Tuple containing the three arrays corresponding to the magnetic moment
        components of each dipole in :math:`Am^2`. These arrays should
        be provided in the following order: ``mag_moment_easting``,
        ``mag_moment_northing``, ``mag_moment_upward``.
    forward_func : callable
        Forward function to be used to compute the desired component of the
        magnetic field. Choose one of :func:`choclo.dipole.magnetic_easting`,
        :func:`choclo.dipole.magnetic_northing` or
        :func:`choclo.dipole.magnetic_upward`.
    shape : tuple of int
        Shape of the expected output array.
    dtype : np.dtype
        Data type of the expected output array.
    parallel : bool
        If True, the forward modelling will be run in parallel. If False, it
        will be run in a single thread.
    progressbar : bool
        If True, a progress bar of the computation will be printed to standard
        error (stderr). Requires :mod:`numba_progress` to be installed.

    Returns
    -------
    magnetic_component : arrays
        Array containing the desired magnetic component.
    """
    # Decide which function should be used
    if parallel:
        jit_func = _jit_dipole_magnetic_component_cartesian_parallel
    else:
        jit_func = _jit_dipole_magnetic_component_cartesian_serial
    # Run computations
    size = prod(shape)
    result = np.zeros(size, dtype=dtype)
    with initialize_progressbar(coordinates[0].size, progressbar) as progress_proxy:
        jit_func(
            coordinates,
            dipoles,
            magnetic_moments,
            result,
            forward_func,
            progress_proxy,
        )
    # Convert to nT
    result *= 1e9
    return result.reshape(shape)


def _check_dipoles_and_magnetic_moments(dipoles, magnetic_moments):
    """
    Check if dipoles and magnetic moments have valid shape and size
    """
    if (size := len(magnetic_moments)) != 3:
        raise ValueError(
            f"Invalid magnetic moments with '{size}' elements."
            " Magnetic moments vectors should have 3 components."
        )
    if magnetic_moments[0].size != dipoles[0].size:
        raise ValueError(
            f"Number of elements in magnetic_moments ({magnetic_moments[0].size})"
            f" mismatch the number of dipoles ({dipoles[0].size})."
        )


def _jit_dipole_magnetic_field_cartesian(
    coordinates, dipoles, magnetic_moments, b_e, b_n, b_u, progress_proxy
):
    """
    Compute the magnetic field components of dipoles in Cartesian coordinates

    Parameters
    ----------
    coordinates : tuple
        Tuple containing the ``easting``, ``northing``, ``upward`` coordinates
        of the computation points in Cartesian coordinate system.
    dipoles : tuple
        Tuple containing the ``easting``, ``northing``, ``upward`` coordinates
        of the dipoles in Cartesian coordinate system.
    magnetic_moments : 2d-array
        Array with the components of the magnetic moment for each dipole.
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
    # Unpack coordinates and magnetic_moments
    easting, northing, upward = coordinates
    easting_p, northing_p, upward_p = dipoles
    mag_e, mag_n, mag_u = magnetic_moments
    # Iterate over computation points and prisms
    for l in prange(easting.size):
        for m in range(easting_p.size):
            easting_comp, northing_comp, upward_comp = magnetic_field(
                easting[l],
                northing[l],
                upward[l],
                easting_p[m],
                northing_p[m],
                upward_p[m],
                mag_e[m],
                mag_n[m],
                mag_u[m],
            )
            b_e[l] += easting_comp
            b_n[l] += northing_comp
            b_u[l] += upward_comp
        # Update progress bar if called
        if update_progressbar:
            progress_proxy.update(1)


def _jit_dipole_magnetic_component_cartesian(
    coordinates, dipoles, magnetic_moments, result, forward_func, progress_proxy
):
    """
    Compute a single magnetic component of dipoles in Cartesian coordinates

    Parameters
    ----------
    coordinates : tuple
        Tuple containing the ``easting``, ``northing``, ``upward`` coordinates
        of the computation points in Cartesian coordinate system.
    dipoles : tuple
        Tuple containing the ``easting``, ``northing``, ``upward`` coordinates
        of the dipoles in Cartesian coordinate system.
    magnetic_moments : 2d-array
        Array with the components of the magnetic moment for each dipole.
    result : 1d-array
        Array where the resulting values of the selected component of the
        magnetic field will be stored.
    forward_function : callable
        Forward function to be used to compute the desired component of the
        magnetic field. Choose one of :func:`choclo.dipole.magnetic_easting`,
        :func:`choclo.dipole.magnetic_northing` or
        :func:`choclo.dipole.magnetic_upward`.
    progress_proxy : :class:`numba_progress.ProgressBar` or None
        Instance of :class:`numba_progress.ProgressBar` that gets updated after
        each iteration on the observation points. Use None if no progress bar
        is should be used.
    """
    # Check if we need to update the progressbar on each iteration
    update_progressbar = progress_proxy is not None
    # Unpack coordinates and magnetic_moments
    easting, northing, upward = coordinates
    easting_p, northing_p, upward_p = dipoles
    mag_e, mag_n, mag_u = magnetic_moments
    # Iterate over computation points and prisms
    for l in prange(easting.size):
        for m in range(easting_p.size):
            result[l] += forward_func(
                easting[l],
                northing[l],
                upward[l],
                easting_p[m],
                northing_p[m],
                upward_p[m],
                mag_e[m],
                mag_n[m],
                mag_u[m],
            )
        # Update progress bar if called
        if update_progressbar:
            progress_proxy.update(1)


_jit_dipole_magnetic_field_cartesian_serial = jit(nopython=True)(
    _jit_dipole_magnetic_field_cartesian
)
_jit_dipole_magnetic_field_cartesian_parallel = jit(nopython=True, parallel=True)(
    _jit_dipole_magnetic_field_cartesian
)
_jit_dipole_magnetic_component_cartesian_serial = jit(nopython=True)(
    _jit_dipole_magnetic_component_cartesian
)
_jit_dipole_magnetic_component_cartesian_parallel = jit(nopython=True, parallel=True)(
    _jit_dipole_magnetic_component_cartesian
)
