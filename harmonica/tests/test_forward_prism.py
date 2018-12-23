"""
Test the prism forward modeling code.
"""
import pytest
import numpy as np
import numpy.testing as npt
import verde as vd

from ..forward.prism import prism_gravity, atan2, log


def test_forward_prism_atan2():
    "Check that the modified atan2 returns the expected values."
    npt.assert_allclose(atan2(1, 1), np.pi / 4)
    npt.assert_allclose(atan2(0, 12), 0)
    npt.assert_allclose(atan2(1, -1), -np.pi / 4)
    npt.assert_allclose(atan2(-1, -1), np.pi / 4)


def test_forward_prism_log():
    "Check that the modified log returns 0 for 0 input"
    npt.assert_allclose(log(0), 0)
    npt.assert_allclose(log(1), 0)
    npt.assert_allclose(log(2.5), np.log(2.5))


@pytest.fixture
def symmetric_prism():
    "Symmetric prism and generic grid coordinates surrounding it"
    prism = (-500, 500, -500, 500, -500, 500)
    # Make the computation points surround the prism
    region = [-1000, 1000, -1000, 1000]
    spacing = 10
    x, y, distance = vd.grid_coordinates(region, spacing=spacing, extra_coords=[510])
    return x, y, distance, prism


def calculate_gravity_around(x, y, distance, prism, density, field):
    "Calculate the given gravity field all around the prism"
    top = prism_gravity((x, y, -distance), prism, density=density, field=field)
    bottom = prism_gravity((x, y, distance), prism, density=density, field=field)
    east = prism_gravity((distance, x, y), prism, density=density, field=field)
    west = prism_gravity((-distance, x, y), prism, density=density, field=field)
    north = prism_gravity((x, distance, y), prism, density=density, field=field)
    south = prism_gravity((x, -distance, y), prism, density=density, field=field)
    return west, east, south, north, top, bottom


def test_forward_prism_gz(symmetric_prism):
    "The vertical gravity component should be consistent around the prism"
    west, east, south, north, top, bottom = calculate_gravity_around(
        *symmetric_prism, density=1000, field="gz"
    )
    npt.assert_allclose(top, -bottom, atol=1e-5)
    npt.assert_allclose(north, south, atol=1e-5)
    npt.assert_allclose(east, west, atol=1e-5)


def test_forward_prism_potential(symmetric_prism):
    "The potential should be consistent around the prism"
    west, east, south, north, top, bottom = calculate_gravity_around(
        *symmetric_prism, density=1000, field="potential"
    )
    npt.assert_allclose(top, bottom, atol=1e-5)
    npt.assert_allclose(top, south, atol=1e-5)
    npt.assert_allclose(top, north, atol=1e-5)
    npt.assert_allclose(top, west, atol=1e-5)
    npt.assert_allclose(top, east, atol=1e-5)


# Test around other fields, check numerical derivative against analytical, check laplace
# for gradients
