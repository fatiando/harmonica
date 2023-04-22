# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
import numpy as np
import numpy.testing as npt
import pytest

from .. import magnetic_ang_to_vec, magnetic_vec_to_ang

VECTORS = [
    [0.5, 0.5, -0.70710678],
    [0.5, 0.5, 0.70710678],
    [-0.5, 0.5, -0.70710678],
    [0, 0, -1],  # Over -z axis
    [1, 0, 0],  # Over east (y) axis
    [0, 1, 0],  # Over north (x) axis
]

ANGLES = [[1, 45, 45], [1, -45, 45.0], [1, 45, -45], [1, 90, 0], [1, 0, 90], [1, 0, 0]]


@pytest.fixture(name="data")
def create_angles_vectors():
    """
    Generate a tuple of angles and vectors
    """
    intensity, inclination, declination = np.vstack(ANGLES).T
    magnetic_e, magnetic_n, magnetic_u = np.vstack(VECTORS).T
    return intensity, inclination, declination, magnetic_e, magnetic_n, magnetic_u


def test_magnetic_ang_to_vec_float():
    """
    Check if the function returns the expected values for a given intensity
    inclination and declination as float
    """
    for i in range(len(ANGLES)):
        intensity, inclination, declination = ANGLES[i]
        magnetic_e, magnetic_n, magnetic_u = VECTORS[i]
        npt.assert_almost_equal(
            magnetic_ang_to_vec(intensity, inclination, declination),
            (magnetic_e, magnetic_n, magnetic_u),
        )


def test_magnetic_vec_to_ang_float():
    """
    Check if the function returns the expected values for a given magnetic
    vector as float
    """
    for i in range(len(VECTORS)):
        intensity, inclination, declination = ANGLES[i]
        magnetic_e, magnetic_n, magnetic_u = VECTORS[i]
        npt.assert_allclose(
            magnetic_vec_to_ang(magnetic_e, magnetic_n, magnetic_u),
            (intensity, inclination, declination),
        )


def test_magnetic_ang_to_vec_array(data):
    """
    Check if the function returns the expected values for a given intensity,
    inclination and declination a array
    """
    intensity, inclination, declination = data[:3]
    magnetic_e, magnetic_n, magnetic_u = data[3:]
    npt.assert_almost_equal(
        magnetic_ang_to_vec(intensity, inclination, declination),
        (magnetic_e, magnetic_n, magnetic_u),
    )


def test_magnetic_vec_to_ang_array(data):
    """
    Check if the function returns the expected values for the given magnentic
    vector as array
    """
    intensity, inclination, declination = data[:3]
    magnetic_e, magnetic_n, magnetic_u = data[3:]
    npt.assert_allclose(
        magnetic_vec_to_ang(magnetic_e, magnetic_n, magnetic_u),
        (intensity, inclination, declination),
    )


def test_unity(data):
    magnetic_e, magnetic_n, magnetic_u = data[3:]
    angles = magnetic_vec_to_ang(magnetic_e, magnetic_n, magnetic_u)
    npt.assert_almost_equal(
        magnetic_ang_to_vec(*angles), (magnetic_e, magnetic_n, magnetic_u)
    )
