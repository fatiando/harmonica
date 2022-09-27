# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test forward modelling for prisms.
"""
from unittest.mock import patch

import numpy as np
import numpy.testing as npt
import pytest
import verde as vd

try:
    from numba_progress import ProgressBar
except ImportError:
    ProgressBar = None

from ..forward.prism import (
    _check_prisms,
    _discard_null_prisms,
    prism_gravity,
    safe_atan2,
    safe_log,
)
from ..gravity_corrections import bouguer_correction
from .utils import run_only_with_numba


def test_invalid_field():
    "Check if passing an invalid field raises an error"
    prism = [-100, 100, -100, 100, -200, -100]
    density = 1000
    coordinates = [0, 0, 0]
    with pytest.raises(ValueError):
        prism_gravity(coordinates, prism, density, field="Not a valid field")


def test_invalid_density_array():
    "Check if error is raised when density shape does not match prisms shape"
    # Create a set of 4 tesseroids
    prisms = [
        [-100, 0, -100, 0, -200, -100],
        [-100, 0, 0, 100, -200, -100],
        [0, 100, -100, 0, -200, -100],
        [0, 100, 0, 100, -200, -100],
    ]
    # Generate a two element density
    density = [1000, 2000]
    coordinates = [0, 0, 0]
    with pytest.raises(ValueError):
        prism_gravity(coordinates, prisms, density, field="potential")


def test_invalid_prisms():
    "Check if invalid prism boundaries are caught by _check_prisms"
    w, e, s, n, bottom, top = -100, 100, -100, 100, -200, -100
    # Check if it works properly on valid prisms
    _check_prisms(np.atleast_2d([w, e, s, n, bottom, top]))
    # Check if it works properly on valid prisms with zero volume
    _check_prisms(np.atleast_2d([w, w, s, n, bottom, top]))
    _check_prisms(np.atleast_2d([w, e, s, s, bottom, top]))
    _check_prisms(np.atleast_2d([w, e, s, n, bottom, bottom]))
    # Test invalid boundaries
    with pytest.raises(ValueError):
        _check_prisms(np.atleast_2d([e, w, s, n, bottom, top]))
    with pytest.raises(ValueError):
        _check_prisms(np.atleast_2d([w, e, n, s, bottom, top]))
    with pytest.raises(ValueError):
        _check_prisms(np.atleast_2d([w, e, s, n, top, bottom]))


def test_discard_null_prisms():
    """
    Test if discarding invalid prisms works as expected
    """
    # Define a set of sample prisms, including invalid ones
    prisms = np.array(
        [
            [0, 10, -50, 33, -2e3, 150],  # ok prism
            [-10, 0, 33, 66, -1e3, -100],  # ok prism (will set zero density)
            [7, 7, -50, 50, -3e3, -1e3],  # no volume due to easting bounds
            [-50, 50, 7, 7, -3e3, -1e3],  # no volume due to northing bounds
            [-50, 50, -50, 50, -3e3, -3e3],  # no volume due to upward bounds
            [7, 7, 7, 7, -200, -200],  # no volume due to multiple bounds
        ]
    )
    density = np.array([2670, 0, 3300, 3000, 2800, 2700])
    prisms, density = _discard_null_prisms(prisms, density)
    npt.assert_allclose(prisms, np.array([[0, 10, -50, 33, -2e3, 150]]))
    npt.assert_allclose(density, np.array([2670]))


@pytest.mark.use_numba
def test_forward_with_null_prisms():
    """
    Test if the forward model with null prisms gives sensible results
    """
    # Create a set of observation points
    coordinates = vd.grid_coordinates(
        region=(-50, 50, -50, 50), shape=(3, 3), extra_coords=0
    )
    # Build a set of prisms that includes null ones (no volume or zero density)
    prisms = [
        [-100, 100, -200, 200, -10e3, -5e3],  # ok prism
        [100, 200, 200, 300, -10e3, -4e3],  # no density prism
        [30, 30, -200, 200, -10e3, -5e3],  # no volume (easting)
        [100, 200, 30, 30, -10e3, -5e3],  # no volume (northing)
        [-100, 100, -200, 200, -10e3, -10e3],  # no volume (upward)
    ]
    density = [2600, 0, 3000, 3100, 3200]
    npt.assert_allclose(
        prism_gravity(coordinates, prisms, density, field="g_z"),
        prism_gravity(coordinates, [prisms[0]], [density[0]], field="g_z"),
    )


@pytest.mark.use_numba
def test_disable_checks():
    "Check if the disable_checks flag works properly"
    valid_prism = [0, 10, 0, 10, -10, 0]
    invalid_prism = [0, 10, 0, 10, 0, -10]
    density = 100
    coordinates = [0, 0, 0]
    # By default, an error should be raised for invalid input
    with pytest.raises(ValueError):
        prism_gravity(coordinates, invalid_prism, density, field="potential")
    # Check if an invalid prism doesn't raise an error with the disable_checks
    # flag set to True
    invalid_result = prism_gravity(
        coordinates, invalid_prism, density, field="potential", disable_checks=True
    )
    # Check if the invalid prism generates a potential field inverse
    # than the one generated by the valid prism
    valid_result = prism_gravity(coordinates, valid_prism, density, field="potential")
    npt.assert_allclose(invalid_result, -valid_result)


@pytest.mark.use_numba
def test_potential_field_symmetry():
    "Test if the potential field satisfies symmetry"
    prism = [-100, 100, -100, 100, -100, 100]
    density = 2670
    # Create six outside computation points located on the normal directions to
    # the prism faces and at the same distance from its center
    coordinates = (
        [-200, 200, 0, 0, 0, 0],
        [0, 0, -200, 200, 0, 0],
        [0, 0, 0, 0, -200, 200],
    )
    result = prism_gravity(coordinates, prism, density, field="potential")
    npt.assert_allclose(result[0], result)
    # Create six inside computation points located on the normal directions to
    # the prism faces and at the same distance from its center
    coordinates = ([-50, 50, 0, 0, 0, 0], [0, 0, -50, 50, 0, 0], [0, 0, 0, 0, -50, 50])
    result = prism_gravity(coordinates, prism, density, field="potential")
    npt.assert_allclose(result[0], result)
    # Create twelve outside computation points located on the diagonal
    # directions to the prism faces and at the same distance from its center.
    # They can be divided into three sets: one made by those points that live
    # on the horizontal plane that passes through the prism center, and the
    # other two that live on the pair of vertical and perpendicular planes that
    # also passes through the center of the prism.
    coordinates = (
        [-200, -200, 200, 200, -200, -200, 200, 200, 0, 0, 0, 0],
        [-200, 200, -200, 200, 0, 0, 0, 0, -200, -200, 200, 200],
        [0, 0, 0, 0, -200, 200, -200, 200, -200, 200, -200, 200],
    )
    result = prism_gravity(coordinates, prism, density, field="potential")
    npt.assert_allclose(result[0], result)
    # Create the same twelve points as before, but now all points fall inside
    # the prism
    coordinates = (
        [-50, -50, 50, 50, -50, -50, 50, 50, 0, 0, 0, 0],
        [-50, 50, -50, 50, 0, 0, 0, 0, -50, -50, 50, 50],
        [0, 0, 0, 0, -50, 50, -50, 50, -50, 50, -50, 50],
    )
    result = prism_gravity(coordinates, prism, density, field="potential")
    npt.assert_allclose(result[0], result)


@pytest.mark.use_numba
def test_g_z_symmetry_outside():
    """
    Test if the g_z field satisfies symmetry

    In order to test if the computed g_z satisfies the symmetry of a square
    prism we will define several set of computation points:

    A. Two points located on the vertical axis of the prism (``easting == 0``
       and ``northing == 0``), one above and one below the prism at the same
       distance from its center.
    B. Four points located on the ``upward == 0`` plane around the prism
       distributed normally to its faces , i.e. only one of the horizontal
       coordinates will be nonzero.
    C. Same as points defined in B, but located on a plane above the prism.
    D. Same as points defined in B, but located on a plane below the prism.
    E. Same as points defined in B, but located on a plane slightly above the
       ``upward == 0`` plane.
    F. Same as points defined in B, but located on a plane slightly below the
       ``upward == 0`` plane.
    G. Four points located on the ``upward == 0`` plane around the prism
       distributed on the diagonal directions , i.e. both horizontal
       coordinates will be equal and nonzero.
    H. Same as points defined in G, but located on an plane above the prism.
    I. Same as points defined in G, but located on an plane below the prism.
    J. Same as points defined in G, but located on a plane slightly above the
       ``upward == 0`` plane.
    K. Same as points defined in G, but located on a plane slightly below the
       ``upward == 0`` plane.

    All computation points defined on the previous groups fall outside of the
    prism.

    The g_z field for a square prism (the horizontal dimensions of the prism
    are equal) must satisfy the following symmetry rules:

    - The g_z values on points A must be opposite.
    - The g_z values on points B must be all zero.
    - The g_z values on points C must be all equal.
    - The g_z values on points D must be all equal.
    - The g_z values on points E must be all equal.
    - The g_z values on points F must be all equal.
    - The g_z values on points C and D must be opposite.
    - The g_z values on points E and F must be opposite.
    - The g_z values on points G must be all zero.
    - The g_z values on points H must be all equal.
    - The g_z values on points I must be all equal.
    - The g_z values on points J must be all equal.
    - The g_z values on points K must be all equal.
    - The g_z values on points H and I must be opposite.
    - The g_z values on points J and K must be opposite.
    """
    prism = [-100, 100, -100, 100, -150, 150]
    density = 2670
    computation_points = {
        "A": ([0, 0], [0, 0], [-200, 200]),
        "B": ([-200, 200, 0, 0], [0, 0, -200, 200], [0, 0, 0, 0]),
        "C": ([-200, 200, 0, 0], [0, 0, -200, 200], [200, 200, 200, 200]),
        "D": ([-200, 200, 0, 0], [0, 0, -200, 200], [-200, -200, -200, -200]),
        "E": ([-200, 200, 0, 0], [0, 0, -200, 200], [1, 1, 1, 1]),
        "F": ([-200, 200, 0, 0], [0, 0, -200, 200], [-1, -1, -1, -1]),
        "G": ([-200, 200, 0, 0], [0, 0, -200, 200], [0, 0, 0, 0]),
        "H": ([-200, -200, 200, 200], [-200, 200, -200, 200], [200, 200, 200, 200]),
        "I": ([-200, -200, 200, 200], [-200, 200, -200, 200], [-200, -200, -200, -200]),
        "J": ([-200, -200, 200, 200], [-200, 200, -200, 200], [1, 1, 1, 1]),
        "K": ([-200, -200, 200, 200], [-200, 200, -200, 200], [-1, -1, -1, -1]),
    }
    # Compute g_z on each set of points
    results = {}
    for group, coords in computation_points.items():
        results[group] = prism_gravity(coords, prism, density, field="g_z")
    # Check symmetries
    # Values on A must be opposite, and the value of g_z at the point above the
    # prism must have the same sign as the density, while the one below should
    # have the opposite
    npt.assert_allclose(results["A"][0], -results["A"][1])
    npt.assert_allclose(np.sign(results["A"][0]), -np.sign(density))
    npt.assert_allclose(np.sign(results["A"][1]), np.sign(density))
    # Values on C, D, E, F, H, I, J, K must be all equal within each set
    for group in "C D E F H I J K".split():
        npt.assert_allclose(results[group][0], results[group])
    # Values on B and G must be zero
    for group in "B G".split():
        npt.assert_allclose(0, results[group])
    # Values on C and D, E and F, H and I, J and K must be opposite
    # Moreover, the set of points that are above the prism must have the same
    # sign as the density, while the ones below should have the opposite
    for above, below in (("C", "D"), ("E", "F"), ("H", "I"), ("J", "K")):
        npt.assert_allclose(results[above], -results[below])
        npt.assert_allclose(np.sign(results[above]), np.sign(density))
        npt.assert_allclose(np.sign(results[below]), -np.sign(density))


def test_g_z_symmetry_inside():
    """
    Test g_z symmetry on computation points that fall inside the prism

    In order to test if the computed g_z satisfies the symmetry of a square
    prism on computation points that fall inside the prism, we will define
    several set of computation points:

    A. Two points located on the vertical axis of the prism (``easting == 0``
       and ``northing == 0``), one above and one below the center of prism,
       but at the same distance from it.
    B. Four points located on the ``upward == 0`` plane around the prism
       distributed normally to its faces , i.e. only one of the horizontal
       coordinates will be nonzero.
    C. Same as points defined in B, but located on a plane above the
       ``upward == 0`` plane.
    D. Same as points defined in B, but located on a plane below the
       ``upward == 0`` plane.
    E. Four points located on the ``upward == 0`` plane around the prism
       distributed on the diagonal directions , i.e. both horizontal
       coordinates will be equal and nonzero.
    F. Same as points defined in E, but located on a plane above the
       ``upward == 0`` plane.
    G. Same as points defined in E, but located on a plane below the
       ``upward == 0`` plane.

    All computation points defined on the previous groups fall outside of the
    prism.

    The g_z field for a square prism (the horizontal dimensions of the prism
    are equal) must satisfy the following symmetry rules:

    - The g_z values on points A must be opposite.
    - The g_z values on points B must be all zero.
    - The g_z values on points C must be all equal.
    - The g_z values on points D must be all equal.
    - The g_z values on points C and D must be opposite.
    - The g_z values on points E must be all zero.
    - The g_z values on points F must be all equal.
    - The g_z values on points G must be all equal.
    - The g_z values on points F and G must be opposite.
    """
    prism = [-100, 100, -100, 100, -150, 150]
    density = 2670
    computation_points = {
        "A": ([0, 0], [0, 0], [-50, 50]),
        "B": ([-50, 50, 0, 0], [0, 0, -50, 50], [0, 0, 0, 0]),
        "C": ([-50, 50, 0, 0], [0, 0, -50, 50], [50, 50, 50, 50]),
        "D": ([-50, 50, 0, 0], [0, 0, -50, 50], [-50, -50, -50, -50]),
        "E": ([-50, -50, 50, 50], [-50, 50, -50, 50], [0, 0, 0, 0]),
        "F": ([-50, -50, 50, 50], [-50, 50, -50, 50], [50, 50, 50, 50]),
        "G": ([-50, -50, 50, 50], [-50, 50, -50, 50], [-50, -50, -50, -50]),
    }
    # Compute g_z on each set of points
    results = {}
    for group, coords in computation_points.items():
        results[group] = prism_gravity(coords, prism, density, field="g_z")
    # Check symmetries
    # Values on A must be opposite, and the value of g_z at the point above the
    # center of the prism must have the same sign as the density, while the one
    # below should have the opposite
    npt.assert_allclose(results["A"][0], -results["A"][1])
    npt.assert_allclose(np.sign(results["A"][0]), -np.sign(density))
    npt.assert_allclose(np.sign(results["A"][1]), np.sign(density))
    # Values on C, D, F, G must be all equal within each set
    for group in "C D F G".split():
        npt.assert_allclose(results[group][0], results[group])
    # Values on B and E must be zero
    for group in "B E".split():
        npt.assert_allclose(0, results[group])
    # Values on C and D, F and G must be opposite
    # Moreover, the set of points that are above the center of the prism must
    # have the same sign as the density, while the ones below should have the
    # opposite
    for above, below in (("C", "D"), ("F", "G")):
        npt.assert_allclose(results[above], -results[below])
        npt.assert_allclose(np.sign(results[above]), np.sign(density))
        npt.assert_allclose(np.sign(results[below]), -np.sign(density))


@pytest.mark.use_numba
def test_safe_atan2():
    "Test the safe_atan2 function"
    # Test safe_atan2 for one point per quadrant
    # First quadrant
    x, y = 1, 1
    npt.assert_allclose(safe_atan2(y, x), np.pi / 4)
    # Second quadrant
    x, y = -1, 1
    npt.assert_allclose(safe_atan2(y, x), -np.pi / 4)
    # Third quadrant
    x, y = -1, -1
    npt.assert_allclose(safe_atan2(y, x), np.pi / 4)
    # Forth quadrant
    x, y = 1, -1
    npt.assert_allclose(safe_atan2(y, x), -np.pi / 4)
    # Test safe_atan2 if the denominator is equal to zero
    npt.assert_allclose(safe_atan2(1, 0), np.pi / 2)
    npt.assert_allclose(safe_atan2(-1, 0), -np.pi / 2)
    # Test safe_atan2 if both numerator and denominator are equal to zero
    assert safe_atan2(0, 0) == 0


@pytest.mark.use_numba
def test_safe_log():
    "Test the safe_log function"
    # Check if safe_log function satisfies safe_log(0) == 0
    assert safe_log(0) == 0
    # Check if safe_log behaves like the natural logarithm in case that x != 0
    x = np.linspace(1, 100, 101)
    for x_i in x:
        npt.assert_allclose(safe_log(x_i), np.log(x_i))


@pytest.mark.use_numba
def test_prism_against_infinite_slab():
    """
    Test if g_z of a large prism matches the solution for an infinite slab
    """
    # Define an observation point 1.5 m above the prism
    coordinates = (0, 0, 1.5)
    # Define prisms with thickness of 10.5 m and horizontal dimensions from 1e3
    # to 1e9m and density of 2670 kg/m^3
    thickness = 10.5
    sizes = np.logspace(3, 9, 7)
    bottom, top = -thickness, 0
    density = 2670
    # Compute the gravity fields generated by each prism
    results = np.zeros_like(sizes)
    for i, size in enumerate(sizes):
        prism = [-size / 2, size / 2, -size / 2, size / 2, bottom, top]
        results[i] = prism_gravity(coordinates, prism, density, field="g_z")
    # Check convergence: assert if as the prism size increases, the result gets
    # closer to the analytical solution for an infinite slab
    analytical = bouguer_correction(np.array(thickness), density)
    diffs = abs((results - analytical) / analytical)
    assert (diffs[1:] < diffs[:-1]).all()
    # Check if the largest size is close enough to the analytical solution
    npt.assert_allclose(analytical, results[-1])


@run_only_with_numba
def test_prisms_parallel_vs_serial():
    """
    Check if the parallelized run returns the same results as the serial one
    """
    prisms = [
        [-100, 0, -100, 0, -10, 0],
        [0, 100, -100, 0, -10, 0],
        [-100, 0, 0, 100, -10, 0],
        [0, 100, 0, 100, -10, 0],
    ]
    densities = [2000, 3000, 4000, 5000]
    coordinates = vd.grid_coordinates(
        region=(-100, 100, -100, 100), spacing=20, extra_coords=10
    )
    for field in ("potential", "g_z"):
        result_parallel = prism_gravity(
            coordinates, prisms, densities, field=field, parallel=True
        )
        result_serial = prism_gravity(
            coordinates, prisms, densities, field=field, parallel=False
        )
        npt.assert_allclose(result_parallel, result_serial)


@pytest.mark.skipif(ProgressBar is None, reason="requires numba_progress")
@pytest.mark.use_numba
def test_progress_bar():
    """
    Check if forward gravity results with and without progress bar match
    """
    prisms = [
        [-100, 0, -100, 0, -10, 0],
        [0, 100, -100, 0, -10, 0],
        [-100, 0, 0, 100, -10, 0],
        [0, 100, 0, 100, -10, 0],
    ]
    densities = [2000, 3000, 4000, 5000]
    coordinates = vd.grid_coordinates(
        region=(-100, 100, -100, 100), spacing=20, extra_coords=10
    )
    for field in ("potential", "g_z"):
        result_progress_true = prism_gravity(
            coordinates, prisms, densities, field=field, progressbar=True
        )
        result_progress_false = prism_gravity(
            coordinates, prisms, densities, field=field, progressbar=False
        )
        npt.assert_allclose(result_progress_true, result_progress_false)


@patch("harmonica.forward.prism.ProgressBar", None)
def test_numba_progress_missing_error():
    """
    Check if error is raised when progresbar=True and numba_progress package
    is not installed.
    """
    prisms = [
        [-100, 0, -100, 0, -10, 0],
        [0, 100, -100, 0, -10, 0],
        [-100, 0, 0, 100, -10, 0],
        [0, 100, 0, 100, -10, 0],
    ]
    densities = [2000, 3000, 4000, 5000]
    coordinates = [0, 0, 0]
    # Check if error is raised
    with pytest.raises(ImportError):
        prism_gravity(
            coordinates, prisms, densities, field="potential", progressbar=True
        )
