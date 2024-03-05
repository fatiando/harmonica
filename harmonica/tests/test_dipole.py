# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test magnetic forward functions for dipoles
"""
import choclo
import numpy as np
import numpy.testing as npt
import pytest
import verde as vd
from choclo.dipole import magnetic_field

try:
    from numba_progress import ProgressBar
except ImportError:
    ProgressBar = None

from .. import dipole_magnetic
from .._forward.dipole import VALID_FIELDS
from .utils import run_only_with_numba


def test_invalid_field():
    "Check if passing an invalid field raises an error"
    coordinates = [0, 0, 0]
    dipoles = ([-100, 100], [-100, 100], [-200, -100])
    magnetic_moments = ([1, -1], [1, -2], [2, 3])
    with pytest.raises(ValueError, match="Invalid field"):
        dipole_magnetic(
            coordinates, dipoles, magnetic_moments, field="Not a valid field"
        )


@pytest.mark.parametrize("field", VALID_FIELDS)
def test_disable_checks(field):
    """Test if disabling checks works as expected"""
    coordinates = [0, 0, 0]
    dipoles = ([1, 1, 2], [-1, 0, 2], [-10, -2, -4])
    magnetic_moments = (
        [1, 4, 7],
        [2, 5, 8],
        [3, 6, 9],
    )
    dipole_magnetic(
        coordinates, dipoles, magnetic_moments, disable_checks=True, field=field
    )


@pytest.mark.use_numba
@pytest.mark.skipif(ProgressBar is None, reason="requires numba_progress")
@pytest.mark.parametrize("field", VALID_FIELDS)
def test_progress_bar(field):
    """
    Check if forward modelling results with and without progress bar match
    """
    dipoles = ([1, 1], [-1, 0], [-10, -2])
    magnetic_moments = ([1.0, 1.0], [1.0, -1.0], [1.0, 5.0])
    coordinates = vd.grid_coordinates(
        region=(-100, 100, -100, 100), spacing=20, extra_coords=10
    )
    result_progress_true = dipole_magnetic(
        coordinates, dipoles, magnetic_moments, field=field, progressbar=True
    )
    result_progress_false = dipole_magnetic(
        coordinates, dipoles, magnetic_moments, field=field, progressbar=False
    )
    npt.assert_allclose(result_progress_true, result_progress_false)


class TestInvalidMagneticMoments:
    @pytest.mark.parametrize("field", ("b", "b_e", "b_n", "b_u"))
    def test_magnetic_moments_and_dipoles(self, field):
        """
        Test error when dipoles and magnetic moments mismatch
        """
        coordinates = [0, 0, 0]
        dipoles = ([1, 1], [-1, 0], [-10, -2])
        # Define invalid magnetic moments (more moments than dipoles)
        magnetic_moments = (
            [1, 4, 7],
            [2, 5, 8],
            [3, 6, 9],
        )
        msg = "Number of elements in magnetic_moments"
        with pytest.raises(ValueError, match=msg):
            dipole_magnetic(coordinates, dipoles, magnetic_moments, field=field)

    @pytest.mark.parametrize("field", VALID_FIELDS)
    def test_magnetic_moments_field(self, field):
        """
        Test if error is raised when magnetic moments have != 3 elements
        """
        coordinates = [0, 0, 0]
        dipoles = ([1, 1], [-1, 0], [-10, -2])
        # Define invalid magnetic moments (more than three components)
        magnetic_moments = ([1, 4], [2, 5], [3, 6], [5, 7])
        msg = "Invalid magnetic moments with '4' elements."
        with pytest.raises(ValueError, match=msg):
            dipole_magnetic(coordinates, dipoles, magnetic_moments, field=field)


class TestSerialVsParallel:
    """
    Test serial vs parallel
    """

    @pytest.mark.parametrize("field", VALID_FIELDS)
    def test_dipoles_parallel_vs_serial_no_numba(self, field):
        """
        Check results of parallelized and serials runs
        Run a small problem with Numba disable to count for test coverage.
        """
        coordinates = ([0, 10], [0, 10], [0, 10])
        dipoles = ([-100, 0], [0, 100], [-20, -50])
        magnetic_moments = (
            [1.0, 1.0],
            [1.0, -1.0],
            [1.0, 5.0],
        )
        parallel = dipole_magnetic(
            coordinates, dipoles, magnetic_moments, field=field, parallel=True
        )
        serial = dipole_magnetic(
            coordinates, dipoles, magnetic_moments, field=field, parallel=False
        )
        npt.assert_allclose(parallel, serial)

    @run_only_with_numba
    @pytest.mark.parametrize("field", VALID_FIELDS)
    def test_dipoles_parallel_vs_serial(self, field):
        """
        Check results of parallelized and serials runs
        Run a large problem only with Numba enabled.
        """
        coordinates = vd.grid_coordinates(
            region=(-100, 100, -100, 100), spacing=20, extra_coords=10
        )
        dipoles = ([-100, 0], [0, 100], [-20, -50])
        magnetic_moments = [
            [1.0, 1.0],
            [1.0, -1.0],
            [1.0, 5.0],
        ]
        parallel = dipole_magnetic(
            coordinates, dipoles, magnetic_moments, field=field, parallel=True
        )
        serial = dipole_magnetic(
            coordinates, dipoles, magnetic_moments, field=field, parallel=False
        )
        npt.assert_allclose(parallel, serial)


class TestAgainstChoclo:
    """
    Test forward modelling functions against dumb Choclo runs
    """

    @pytest.fixture()
    def sample_dipoles(self):
        """
        Return four sample dipoles
        """
        dipoles = (
            np.array([-10.0, 0.0, 10.0, 5.0]),
            np.array([2.0, 0.0, -1.0, 1.5]),
            np.array([-10.0, -15.0, -5.0, -2.0]),
        )
        return dipoles

    @pytest.fixture()
    def sample_magnetic_moments(self):
        """
        Return sample magnetic moment vectors for the dipoles
        """
        magnetic_moments = (
            np.array([1.0, -1.0, 3.0, -1.5]),
            np.array([1.0, -0.5, -1.0, 2.0]),
            np.array([1.0, 2.0, -3.0, -2.0]),
        )
        return magnetic_moments

    @pytest.fixture()
    def sample_coordinates(self):
        """
        Return four sample observation points
        """
        easting = np.array([-5, 10, 0, 15], dtype=float)
        northing = np.array([14, -4, 11, 0], dtype=float)
        upward = np.array([9, 6, 6, 12], dtype=float)
        return (easting, northing, upward)

    def test_against_choclo(
        self, sample_coordinates, sample_dipoles, sample_magnetic_moments
    ):
        """
        Test dipole_magnetic against raw Choclo runs
        """
        easting, northing, upward = sample_coordinates
        easting_p, northing_p, upward_p = sample_dipoles
        # Compute expected results with dumb Choclo runs
        n_coords = easting.size
        n_dipoles = sample_dipoles[0].size
        expected_magnetic_e = np.zeros(n_coords, dtype=np.float64)
        expected_magnetic_n = np.zeros(n_coords, dtype=np.float64)
        expected_magnetic_u = np.zeros(n_coords, dtype=np.float64)
        for i in range(n_coords):
            for j in range(n_dipoles):
                b_e, b_n, b_u = magnetic_field(
                    easting[i],
                    northing[i],
                    upward[i],
                    easting_p[j],
                    northing_p[j],
                    upward_p[j],
                    sample_magnetic_moments[0][j],
                    sample_magnetic_moments[1][j],
                    sample_magnetic_moments[2][j],
                )
                expected_magnetic_e[i] += b_e
                expected_magnetic_n[i] += b_n
                expected_magnetic_u[i] += b_u
        # Convert to nT
        expected_magnetic_e *= 1e9
        expected_magnetic_n *= 1e9
        expected_magnetic_u *= 1e9
        # Compare with harmonica results
        b_e, b_n, b_u = dipole_magnetic(
            sample_coordinates, sample_dipoles, sample_magnetic_moments, field="b"
        )
        npt.assert_allclose(b_e, expected_magnetic_e)
        npt.assert_allclose(b_n, expected_magnetic_n)
        npt.assert_allclose(b_u, expected_magnetic_u)

    @pytest.mark.parametrize("field", ("b_e", "b_n", "b_u"))
    def test_component_against_choclo(
        self, sample_coordinates, sample_dipoles, sample_magnetic_moments, field
    ):
        """
        Test dipole_magnetic_component against raw Choclo runs
        """
        easting, northing, upward = sample_coordinates
        easting_p, northing_p, upward_p = sample_dipoles
        # Compute expected results with dumb Choclo runs
        n_coords = easting.size
        n_dipoles = sample_dipoles[0].size
        expected_result = np.zeros(n_coords, dtype=np.float64)
        forward_func = getattr(choclo.dipole, f"magnetic_{field[-1]}")
        for i in range(n_coords):
            for j in range(n_dipoles):
                expected_result[i] += forward_func(
                    easting[i],
                    northing[i],
                    upward[i],
                    easting_p[j],
                    northing_p[j],
                    upward_p[j],
                    sample_magnetic_moments[0][j],
                    sample_magnetic_moments[1][j],
                    sample_magnetic_moments[2][j],
                )
        # Convert to nT
        expected_result *= 1e9
        # Compare with harmonica results
        result = dipole_magnetic(
            sample_coordinates,
            sample_dipoles,
            sample_magnetic_moments,
            field=field,
        )
        npt.assert_allclose(result, expected_result)
