"""
Test forward modelling for prisms.
"""
import pytest
import numpy as np
import numpy.testing as npt

from ..forward.prism import (
    prism_gravity,
    _check_prisms,
    _check_points_outside_prisms,
    atan2,
    log,
)


def test_invalid_field():
    "Check if passing an invalid field raises an error"
    prism = [-100, 100, -100, 100, 100, 200]
    density = 1000
    coordinates = [0, 0, 0]
    with pytest.raises(ValueError):
        prism_gravity(coordinates, prism, density, field="Not a valid field")


def test_invalid_density_array():
    "Check if error is raised when density shape does not match prisms shape"
    # Create a set of 4 tesseroids
    prisms = [
        [-100, 0, -100, 0, 100, 200],
        [-100, 0, 0, 100, 100, 200],
        [0, 100, -100, 0, 100, 200],
        [0, 100, 0, 100, 100, 200],
    ]
    # Generate a two element density
    density = [1000, 2000]
    coordinates = [0, 0, 0]
    with pytest.raises(ValueError):
        prism_gravity(coordinates, prisms, density, field="potential")


def test_invalid_prisms():
    "Check if invalid prism boundaries are caught by _check_prisms"
    w, e, s, n, bottom, top = -100, 100, -100, 100, 100, 200
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
        _check_prisms(np.atleast_2d([w, e, s, n, bottom, -100]))


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
    "Test if the g_z field satisfies symmetry"
    prism = [-100, 100, -100, 100, -100, 100]
    density = 2670
    # Create two computation points:
    # one above and one bellow at same distance from the prism
    coordinates = [[0, 0], [0, 0], [-200, 200]]
    result = prism_gravity(coordinates, prism, density, field="g_z")
    npt.assert_allclose(result[0], -result[1])


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
