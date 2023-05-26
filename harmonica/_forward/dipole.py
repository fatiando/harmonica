# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Forward modelling for magnetic fields of dipoles
"""
import numpy as np
from choclo.dipole import magnetic_e, magnetic_field, magnetic_n, magnetic_u
from numba import jit, prange


def dipole_magnetic(
    coordinates,
    dipoles,
    magnetic_moments,
    parallel=True,
    dtype="float64",
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
    dipoles : list of arrays
        List of arrays containing the ``easting``, ``northing`` and ``upward``
        locations of the dipoles defined on a Cartesian coordinate system. All
        coordinates should be in meters.
    magnetic_moments : 2d-array
        Two dimensional array containing the magnetic moments of each dipole in
        :math:`Am^2`. Each row contains the components of the magnetic moment
        for each dipole in the following order: ``mag_moment_easting``,
        ``mag_moment_northing``, ``mag_moment_upward``.
    parallel : bool (optional)
        If True the computations will run in parallel using Numba built-in
        parallelization. If False, the forward model will run on a single core.
        Might be useful to disable parallelization if the forward model is run
        by an already parallelized workflow. Default to True.
    dtype : data-type (optional)
        Data type assigned to the resulting gravitational field. Default to
        ``np.float64``.
    disable_checks : bool (optional)
        Flag that controls whether to perform a sanity check on the model.
        Should be set to ``True`` only when it is certain that the input model
        is valid and it does not need to be checked.
        Default to ``False``.

    Returns
    -------
    magnetic_field : tuple of array
        Tuple containing each component of the magnetic field generated by the
        dipoles as arrays. The three components are returned in the following
        order: ``b_e``, ``b_n``, ``b_u``.
    """
    # Figure out the shape and size of the output arrays
    cast = np.broadcast(*coordinates[:3])
    b_e, b_n, b_u = tuple(np.zeros(cast.size, dtype=dtype) for _ in range(3))
    # Convert coordinates, dipoles and magnetic moments to arrays
    coordinates = tuple(np.atleast_1d(i).ravel() for i in coordinates[:3])
    dipoles = tuple(np.atleast_1d(i).ravel() for i in dipoles[:3])
    magnetic_moments = np.atleast_2d(magnetic_moments)
    # Sanity checks
    if not disable_checks:
        if magnetic_moments.shape[1] != 3:
            raise ValueError(
                f"Invalid magnetic moments with '{magnetic_moments.shape[1]}' elements."
                " Magnetic moments vectors should have 3 components."
            )
        if magnetic_moments.shape[0] != dipoles[0].size:
            raise ValueError(
                f"Number of elements in magnetic_moments ({magnetic_moments.shape[0]})"
                f" mismatch the number of dipoles ({dipoles[0].size})."
            )
    # Compute the magnetic fields of the dipoles
    if parallel:
        _jit_dipole_magnetic_field_cartesian_parallel(
            coordinates, dipoles, magnetic_moments, b_e, b_n, b_u
        )
    else:
        _jit_dipole_magnetic_field_cartesian_serial(
            coordinates, dipoles, magnetic_moments, b_e, b_n, b_u
        )
    # Convert to nT
    b_e *= 1e9
    b_n *= 1e9
    b_u *= 1e9
    return b_e.reshape(cast.shape), b_n.reshape(cast.shape), b_u.reshape(cast.shape)


def dipole_magnetic_component(
    coordinates,
    dipoles,
    magnetic_moments,
    component,
    parallel=True,
    dtype="float64",
    disable_checks=False,
):
    """
    Compute single magnetic field component of dipoles in Cartesian coordinates

    .. important::

        Use this function only if you need to compute a single component of the
        magnetic field. Use :func:`harmonica.dipole_magnetic` to compute the
        three components more efficiently.

    Parameters
    ----------
    coordinates : list of arrays
        List of arrays containing the ``easting``, ``northing`` and ``upward``
        coordinates of the computation points defined on a Cartesian coordinate
        system. All coordinates should be in meters.
    dipoles : list of arrays
        List of arrays containing the ``easting``, ``northing`` and ``upward``
        locations of the dipoles defined on a Cartesian coordinate system. All
        coordinates should be in meters.
    magnetic_moments : 2d-array
        Two dimensional array containing the magnetic moments of each dipole in
        :math:`Am^2`. Each row contains the components of the magnetic moment
        for each dipole in the following order: ``mag_moment_easting``,
        ``mag_moment_northing``, ``mag_moment_upward``.
    component : str
        Computed that will be computed. Available options are: ``"easting"``,
        ``"northing"`` or ``"upward"``.
    parallel : bool (optional)
        If True the computations will run in parallel using Numba built-in
        parallelization. If False, the forward model will run on a single core.
        Might be useful to disable parallelization if the forward model is run
        by an already parallelized workflow. Default to True.
    dtype : data-type (optional)
        Data type assigned to the resulting gravitational field. Default to
        ``np.float64``.
    disable_checks : bool (optional)
        Flag that controls whether to perform a sanity check on the model.
        Should be set to ``True`` only when it is certain that the input model
        is valid and it does not need to be checked.
        Default to ``False``.

    Returns
    -------
    magnetic_field :  array
        Array containing the magnetic field component in :math:`nT`.
    """
    # Figure out the shape and size of the output arrays
    cast = np.broadcast(*coordinates[:3])
    result = np.zeros(cast.size, dtype=dtype)
    # Convert coordinates, dipoles and magnetic moments to arrays
    coordinates = tuple(np.atleast_1d(i).ravel() for i in coordinates[:3])
    dipoles = tuple(np.atleast_1d(i).ravel() for i in dipoles[:3])
    magnetic_moments = np.atleast_2d(magnetic_moments)
    # Choose forward modelling function based on the chosen component
    forward_function = _get_magnetic_forward_function(component)
    # Sanity checks
    if not disable_checks:
        if magnetic_moments.shape[1] != 3:
            raise ValueError(
                f"Invalid magnetic moments with '{magnetic_moments.shape[1]}' elements."
                " Magnetic moments vectors should have 3 components."
            )
        if magnetic_moments.shape[0] != dipoles[0].size:
            raise ValueError(
                f"Number of elements in magnetic_moments ({magnetic_moments.shape[0]})"
                f" mismatch the number of dipoles ({dipoles[0].size})."
            )
    # Compute the magnetic fields of the dipoles
    if parallel:
        _jit_dipole_magnetic_component_cartesian_parallel(
            coordinates, dipoles, magnetic_moments, result, forward_function
        )
    else:
        _jit_dipole_magnetic_component_cartesian_serial(
            coordinates, dipoles, magnetic_moments, result, forward_function
        )
    # Convert to nT
    result *= 1e9
    return result.reshape(cast.shape)


def _jit_dipole_magnetic_field_cartesian(
    coordinates, dipoles, magnetic_moments, b_e, b_n, b_u
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
    """
    easting, northing, upward = coordinates
    easting_p, northing_p, upward_p = dipoles
    for l in prange(easting.size):
        for m in range(easting_p.size):
            easting_comp, northing_comp, upward_comp = magnetic_field(
                easting[l],
                northing[l],
                upward[l],
                easting_p[m],
                northing_p[m],
                upward_p[m],
                magnetic_moments[m, 0],
                magnetic_moments[m, 1],
                magnetic_moments[m, 2],
            )
            b_e[l] += easting_comp
            b_n[l] += northing_comp
            b_u[l] += upward_comp


def _jit_dipole_magnetic_component_cartesian(
    coordinates, dipoles, magnetic_moments, result, forward_func
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
    """
    easting, northing, upward = coordinates
    easting_p, northing_p, upward_p = dipoles
    for l in prange(easting.size):
        for m in range(easting_p.size):
            result[l] += forward_func(
                easting[l],
                northing[l],
                upward[l],
                easting_p[m],
                northing_p[m],
                upward_p[m],
                magnetic_moments[m, 0],
                magnetic_moments[m, 1],
                magnetic_moments[m, 2],
            )


def _get_magnetic_forward_function(component):
    """
    Returns the Choclo magnetic forward modelling function for the desired
    component

    Parameters
    ----------
    component : str
        Magnetic field component.

    Returns
    -------
    forward_function : callable
        Forward modelling function for the desired component.
    """
    if component not in ("easting", "northing", "upward"):
        raise ValueError(
            f"Invalid component '{component}'. "
            "It must be either 'easting', 'northing' or 'upward'."
        )
    functions = {"easting": magnetic_e, "northing": magnetic_n, "upward": magnetic_u}
    return functions[component]


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
