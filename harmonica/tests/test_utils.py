# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
import numpy as np
import numpy.testing as npt
import pytest

from .. import magnetic_angles_to_vec, magnetic_vec_to_angles, total_field_anomaly

VECTORS = [
    [0.5, 0.5, -0.70710678],
    [0.5, 0.5, 0.70710678],
    [-0.5, 0.5, -0.70710678],
    [0, 0, -1],  # Over -z axis
    [1, 0, 0],  # Over east (y) axis
    [0, 1, 0],  # Over north (x) axis
    [0, 0, -1e-13],  # Over -z axis
    [1e-13, 0, 0],  # Over east (y) axis
    [0, 1e-13, 0],  # Over north (x) axis
]

ANGLES = [
    [1, 45, 45],
    [1, -45, 45.0],
    [1, 45, -45],
    [1, 90, 0],
    [1, 0, 90],
    [1, 0, 0],
    [1e-13, 90, 0],
    [1e-13, 0, 90],
    [1e-13, 0, 0],
]


@pytest.mark.parametrize("angles, vector", [(a, v) for a, v in zip(ANGLES, VECTORS)])
def test_magnetic_ang_to_vec_float(angles, vector):
    """
    Check if the function returns the expected values for a given intensity
    inclination and declination as float
    """
    intensity, inclination, declination = angles
    magnetic_e, magnetic_n, magnetic_u = vector
    npt.assert_almost_equal(
        magnetic_angles_to_vec(intensity, inclination, declination),
        (magnetic_e, magnetic_n, magnetic_u),
    )


@pytest.mark.parametrize("degrees", (False, True), ids=("radians", "degrees"))
@pytest.mark.parametrize("angles, vector", [(a, v) for a, v in zip(ANGLES, VECTORS)])
def test_magnetic_vec_to_angles_float(angles, vector, degrees):
    """
    Check if the function returns the expected values for a given magnetic
    vector as float
    """
    intensity, inclination, declination = angles
    magnetic_e, magnetic_n, magnetic_u = vector
    if not degrees:
        inclination, declination = np.radians(inclination), np.radians(declination)
    npt.assert_allclose(
        magnetic_vec_to_angles(magnetic_e, magnetic_n, magnetic_u, degrees=degrees),
        (intensity, inclination, declination),
    )


@pytest.fixture(name="arrays", params=["single-element", "multi-element"])
def angles_vectors_as_arrays(request):
    """
    Generate magnetic angles and vectors as arrays
    """
    if request.param == "single-element":
        # Generate arrays with a single element
        intensity, inclination, declination = tuple(np.atleast_1d(i) for i in ANGLES[0])
        magnetic_e, magnetic_n, magnetic_u = tuple(np.atleast_1d(i) for i in VECTORS[0])
    else:
        intensity, inclination, declination = np.vstack(ANGLES).T
        magnetic_e, magnetic_n, magnetic_u = np.vstack(VECTORS).T
    return (intensity, inclination, declination), (magnetic_e, magnetic_n, magnetic_u)


def test_magnetic_ang_to_vec_array(arrays):
    """
    Check if the function returns the expected values for a given intensity,
    inclination and declination a array
    """
    intensity, inclination, declination = arrays[0]
    magnetic_e, magnetic_n, magnetic_u = arrays[1]
    npt.assert_almost_equal(
        magnetic_angles_to_vec(intensity, inclination, declination),
        (magnetic_e, magnetic_n, magnetic_u),
    )


@pytest.mark.parametrize("degrees", (False, True), ids=("radians", "degrees"))
def test_magnetic_vec_to_angles_array(arrays, degrees):
    """
    Check if the function returns the expected values for the given magnetic
    vector as arrays
    """
    intensity, inclination, declination = arrays[0]
    magnetic_e, magnetic_n, magnetic_u = arrays[1]
    if not degrees:
        inclination, declination = np.radians(inclination), np.radians(declination)
    npt.assert_allclose(
        magnetic_vec_to_angles(magnetic_e, magnetic_n, magnetic_u, degrees=degrees),
        (intensity, inclination, declination),
    )


@pytest.mark.parametrize("start_with", ("angles", "vectors"))
def test_identity(arrays, start_with):
    """
    Check if applying both conversions return the original set of vectors
    """
    if start_with == "angles":
        intensity, inclination, declination = arrays[0]
        vector = magnetic_angles_to_vec(intensity, inclination, declination)
        npt.assert_almost_equal(
            magnetic_vec_to_angles(*vector), (intensity, inclination, declination)
        )
    else:
        magnetic_e, magnetic_n, magnetic_u = arrays[1]
        angles = magnetic_vec_to_angles(magnetic_e, magnetic_n, magnetic_u)
        npt.assert_almost_equal(
            magnetic_angles_to_vec(*angles), (magnetic_e, magnetic_n, magnetic_u)
        )


@pytest.mark.parametrize("direction", ("easting", "northing", "upward"))
def test_tfa(direction):
    b = [30.0, -40.0, 50.0]
    if direction == "easting":
        inc, dec = 0.0, 90.0
        expected_tfa = b[0]
    elif direction == "northing":
        inc, dec = 0.0, 0.0
        expected_tfa = b[1]
    else:
        inc, dec = -90.0, 0.0
        expected_tfa = b[2]
    tfa = total_field_anomaly(b, inc, dec)
    npt.assert_allclose(tfa, expected_tfa)


@pytest.mark.parametrize("direction", ("easting", "northing", "upward"))
def test_tfa_b_as_array(direction):
    b = [[20, -30, -40, 50], [-60, 70, -80, 10], [21, -31, 41, -51]]
    if direction == "easting":
        inc, dec = 0.0, 90.0
        expected_tfa = b[0]
    elif direction == "northing":
        inc, dec = 0.0, 0.0
        expected_tfa = b[1]
    else:
        inc, dec = -90.0, 0.0
        expected_tfa = b[2]
    tfa = total_field_anomaly(b, inc, dec)
    npt.assert_allclose(tfa, expected_tfa)


def test_tfa_inc_dec_as_array():
    be = [20.0, -30.0, -40.0, 50.0]
    bn = [-60.0, 70.0, -80.0, 10.0]
    bu = [21.0, -31.0, 41.0, -51.0]
    b = [be, bn, bu]
    inc = [0.0, 0.0, 45.0, -90.0]
    dec = [90.0, 0.0, 45.0, 0.0]
    tfa = total_field_anomaly(b, inc, dec)
    expected_tfa = [
        be[0],
        bn[1],
        0.5 * be[2] + 0.5 * bn[2] - 1 / np.sqrt(2) * bu[2],
        bu[3],
    ]
    npt.assert_allclose(tfa, expected_tfa)
