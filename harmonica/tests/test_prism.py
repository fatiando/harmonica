"""
Test forward modelling for prisms.
"""
import pytest
import numpy as np
import numpy.testing as npt

from ..constants import GRAVITATIONAL_CONST
from ..forward.prism import (
    prism_gravity,
    _check_prisms,
    _check_points_outside_prisms,
    atan2,
    log,
)


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
    # Check if it works properly on valid tesseroid with zero volume
    _check_prisms(np.atleast_2d([w, w, s, n, bottom, top]))
    _check_prisms(np.atleast_2d([w, e, s, s, bottom, top]))
    _check_prisms(np.atleast_2d([w, e, s, n, bottom, bottom]))
    # Test invalid longitude and latitude boundaries
    with pytest.raises(ValueError):
        _check_prisms(np.atleast_2d([150, 100, s, n, bottom, top]))
    with pytest.raises(ValueError):
        _check_prisms(np.atleast_2d([w, e, 150, 100, bottom, top]))
    # Test invalid radial boundaries
    with pytest.raises(ValueError):
        _check_prisms(np.atleast_2d([w, e, s, n, 150, 100]))
    with pytest.raises(ValueError):
        _check_prisms(np.atleast_2d([w, e, s, n, 300, top]))
    with pytest.raises(ValueError):
        _check_prisms(np.atleast_2d([w, e, s, n, bottom, -500]))


def test_point_inside_prisms():
    "Check if a computation point inside a prism is caught"
    prisms = np.atleast_2d([-100, 100, -100, 100, 1000, 2000])
    # Test if outside point is not caught
    points = [
        np.atleast_2d([0, 0, 2500]).T,  # outside point on vertical
        np.atleast_2d([200, 0, 1500]).T,  # outside point on easting
        np.atleast_2d([0, 200, 1500]).T,  # outside point on northing
        np.atleast_2d([0, 0, 2000]).T,  # point on top surface
        np.atleast_2d([0, 0, 1000]).T,  # point on bottom surface
        np.atleast_2d([-100, 0, 1500]).T,  # point on western surface
        np.atleast_2d([100, 0, 1500]).T,  # point on eastern surface
        np.atleast_2d([0, -100, 1500]).T,  # point on southern surface
        np.atleast_2d([0, 100, 1500]).T,  # point on northern surface
    ]
    for coordinates in points:
        _check_points_outside_prisms(coordinates, prisms)
    # Test if computation point is inside the prism
    coordinates = np.atleast_2d([0, 0, 1500]).T
    with pytest.raises(ValueError):
        _check_points_outside_prisms(coordinates, prisms)


@pytest.mark.use_numba
def test_potential_field_symmetry():
    "Test if the potential field satisfies symmetry"
    prism = [-100, 100, -100, 100, -100, 100]
    density = 2670
    # Create six computation points at the same distance from the prism
    coords = [-200, 200]
    coordinates = tuple(i.ravel() for i in np.meshgrid(coords, coords, coords))
    result = prism_gravity(coordinates, prism, density, field="potential")
    npt.assert_allclose(result[0], result)


@pytest.mark.use_numba
def test_g_z_symmetry():
    """
    Test if the g_z field satisfies symmetry

    In order to test if the computed g_z satisfies the symmetry of a square prism
    we will define several set of computation points:

    A. Two points located on the vertical axis of the prism (``easting == 0`` and
    ``northing == 0``), one above and one bellow the prism at the same distance from its
    center.
    B. Four points located around the prism on the ``upward == 0`` plane distributed
       normally to the faces of the prism, i.e. only one of the horizontal coordinates
       will be nonzero.
    C. Same as points defined in B, but located on an ``upward`` plane above the prism.
    D. Same as points defined in B, but located on an ``upward`` plane bellow the prism.
    E. Four points located around the prism on the ``upward == 0`` plane distributed on
       the diagonal directions, i.e. both horizontal coordinates will be equal and
       nonzero.
    F. Same as points defined in E, but located on an ``upward`` plane above the prism.
    G. Same as points defined in E, but located on an ``upward`` plane bellow the prism.

    The g_z field for a square prism (the horizontal dimensions of the prism are equal)
    must satisfy the following symmetry rules:

    - The g_z values on points A must be opposite.
    - The g_z values on points B must be all equal.
    - The g_z values on points C must be all equal.
    - The g_z values on points D must be all equal.
    - The g_z values on points C and D must be opposite.
    - The g_z values on points E must be all equal.
    - The g_z values on points F must be all equal.
    - The g_z values on points G must be all equal.
    - The g_z values on points F and G must be opposite.
    """
    prism = [-100, 100, -100, 100, -150, 150]
    density = 2670
    computation_points = {
        "A": ([0, 0], [0, 0], [-200, 200]),
        "B": ([-200, 200, 0, 0], [0, 0, -200, 200], [0, 0, 0, 0]),
        "C": ([-200, 200, 0, 0], [0, 0, -200, 200], [200, 200, 200, 200]),
        "D": ([-200, 200, 0, 0], [0, 0, -200, 200], [-200, -200, -200, -200]),
        "E": ([-200, -200, 200, 200], [-200, 200, -200, 200], [0, 0, 0, 0]),
        "F": ([-200, -200, 200, 200], [-200, 200, -200, 200], [200, 200, 200, 200]),
        "G": ([-200, -200, 200, 200], [-200, 200, -200, 200], [-200, -200, -200, -200]),
    }
    # Compute g_z on each set of points
    results = {}
    for group in computation_points:
        results[group] = prism_gravity(
            computation_points[group], prism, density, field="g_z"
        )
    # Check symmetries
    # Values on A must be opposite
    npt.assert_allclose(results["A"][0], -results["A"][1])
    # Values on B, C, D, E, F and G must be all equal within each set
    for group in "B C D E F G".split():
        npt.assert_allclose(results[group][0], results[group])
    # Values on C and D, and in F and G must be opposite
    for above, bellow in (("C", "D"), ("F", "G")):
        npt.assert_allclose(results[above], -results[bellow])


@pytest.mark.use_numba
def test_custom_atan2():
    "Test the custom atan2 function"
    # Test atan2 for one poit per quadrant
    # First quadrant
    x, y = 1, 1
    npt.assert_allclose(atan2(y, x), np.pi / 4)
    # Second quadrant
    x, y = -1, 1
    npt.assert_allclose(atan2(y, x), 3 / 4 * np.pi)
    # Third quadrant
    x, y = -1, -1
    npt.assert_allclose(atan2(y, x), -3 / 4 * np.pi)
    # Forth quadrant
    x, y = 1, -1
    npt.assert_allclose(atan2(y, x), -np.pi / 4)
    # Numerator equal to zero
    assert atan2(0, 1) == 0


@pytest.mark.use_numba
def test_custom_log():
    "Test the custom log function"
    # Check if custom log function satisfies log(0) == 0
    assert log(0) == 0
    # Check if x != 0 the custom log behavies like the natural logarithm
    x = np.linspace(1, 100, 101)
    for x_i in x:
        npt.assert_allclose(log(x_i), np.log(x_i))


def gravity_infinite_slab(thickness, density):
    """
    Compute the downward gravity acceleration generated by a infinite slab

    Parameters
    ----------
    thickness : float
        Thickness of the infinite slab in meters.
    density : float
        Density of the infinite slab in kg/m^3.

    Returns
    -------
    result : float
        Downward component of the gravity acceleration generated by the infinite slab
        in mGal.
    """
    result = 2 * np.pi * GRAVITATIONAL_CONST * density * thickness
    # Convert the result for g_z to mGal
    result *= 1e5
    return result


@pytest.mark.use_numba
def test_prism_against_infinite_slab():
    """
    Test if g_z of a large prism matches the solution for an infinite slab
    """
    # Define an observation point 1.5 m above the prism
    coordinates = (0, 0, 1.5)
    # Define prisms with thickness of 10.5 m and horizontal dimensions from 1e3 to 1e9m
    # and density of 2670 kg/m^3
    thickness = 10.5
    sizes = np.logspace(3, 9, 7)
    bottom, top = -thickness, 0
    density = 2670
    # Compute the gravity fields generated by each prism
    results = np.zeros_like(sizes)
    for i, size in enumerate(sizes):
        west, east = -size / 2, size / 2
        south, north = west, east
        prism = [west, east, south, north, bottom, top]
        results[i] = prism_gravity(coordinates, prism, density, field="g_z")
    # Check convergence: assert if as the prism size increases, the result gets closer
    # to the analytical solution for an infinite slab
    analytical = gravity_infinite_slab(thickness, density)
    diffs = abs((results - analytical) / analytical)
    assert (diffs[1:] < diffs[:-1]).all()
    # Check if the largest size is close enough to the analytical solution
    npt.assert_allclose(analytical, results[-1])
