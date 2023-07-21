# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
import numpy as np
import numpy.testing as npt
import pytest

from .. import magnetic_angles_to_vec, magnetic_vec_to_angles

VECTORS = [
    [0.5, 0.5, -0.70710678],
    [0.5, 0.5, 0.70710678],
    [-0.5, 0.5, -0.70710678],
    [0, 0, -1],  # Over -z axis
    [1, 0, 0],  # Over east (y) axis
    [0, 1, 0],  # Over north (x) axis
]

ANGLES = [[1, 45, 45], [1, -45, 45.0], [1, 45, -45], [1, 90, 0], [1, 0, 90], [1, 0, 0]]


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


@pytest.mark.parametrize("angles, vector", [(a, v) for a, v in zip(ANGLES, VECTORS)])
def test_magnetic_vec_to_angles_float(angles, vector):
    """
    Check if the function returns the expected values for a given magnetic
    vector as float
    """
    intensity, inclination, declination = angles
    magnetic_e, magnetic_n, magnetic_u = vector
    npt.assert_allclose(
        magnetic_vec_to_angles(magnetic_e, magnetic_n, magnetic_u),
        (intensity, inclination, declination),
    )


@pytest.fixture(name="arrays")
def angles_vectors_as_arrays():
    """
    Generate magnetic angles and vectors as arrays
    """
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


def test_magnetic_vec_to_angles_array(arrays):
    """
    Check if the function returns the expected values for the given magnetic
    vector as arrays
    """
    intensity, inclination, declination = arrays[0]
    magnetic_e, magnetic_n, magnetic_u = arrays[1]
    npt.assert_allclose(
        magnetic_vec_to_angles(magnetic_e, magnetic_n, magnetic_u),
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
