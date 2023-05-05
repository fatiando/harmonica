# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test forward functions for magnetic field of prisms
"""
import choclo
import numpy as np
import numpy.testing as npt
import pytest
import verde as vd
from choclo.prism import magnetic_field

try:
    from numba_progress import ProgressBar
except ImportError:
    ProgressBar = None

from .. import prism_magnetic, prism_magnetic_component
from .utils import run_only_with_numba


def test_invalid_component():
    "Check if passing an invalid component raises an error"
    prism = [-100, 100, -100, 100, -200, -100]
    magnetization = [1000, 1, 2]
    coordinates = [0, 0, 0]
    with pytest.raises(ValueError, match="Invalid component"):
        prism_magnetic_component(
            coordinates, prism, magnetization, component="Not a valid field"
        )


@pytest.mark.use_numba
@pytest.mark.skipif(ProgressBar is None, reason="requires numba_progress")
@pytest.mark.parametrize("component", (None, "easting", "northing", "upward"))
def test_progress_bar(component):
    """
    Check if forward modelling results with and without progress bar match
    """
    prisms = [
        [-100, 0, -100, 0, -10, 0],
        [0, 100, -100, 0, -10, 0],
    ]
    magnetizations = [
        [1.0, 1.0, 1.0],
        [1.0, -1.0, 5.0],
    ]
    coordinates = vd.grid_coordinates(
        region=(-100, 100, -100, 100), spacing=20, extra_coords=10
    )
    if component is None:
        result_progress_true = prism_magnetic(
            coordinates, prisms, magnetizations, progressbar=True
        )
        result_progress_false = prism_magnetic(
            coordinates, prisms, magnetizations, progressbar=False
        )
    else:
        result_progress_true = prism_magnetic_component(
            coordinates, prisms, magnetizations, component, progressbar=True
        )
        result_progress_false = prism_magnetic_component(
            coordinates, prisms, magnetizations, component, progressbar=False
        )
    npt.assert_allclose(result_progress_true, result_progress_false)


class TestSerialVsParallel:
    """
    Test serial vs parallel
    """

    @pytest.mark.parametrize("component", (None, "easting", "northing", "upward"))
    def test_prisms_parallel_vs_serial_no_numba(self, component):
        """
        Check results of parallelized and serials runs

        Run a small problem with Numba disable to count for test coverage.
        """
        prisms = [
            [-100, 0, -100, 0, -10, 0],
            [0, 100, -100, 0, -10, 0],
        ]
        magnetizations = [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, 5.0],
        ]
        coordinates = ([0, 10], [0, 10], [0, 10])
        if component is None:
            parallel = prism_magnetic(
                coordinates, prisms, magnetizations, parallel=True
            )
            serial = prism_magnetic(coordinates, prisms, magnetizations, parallel=False)
        else:
            parallel = prism_magnetic_component(
                coordinates, prisms, magnetizations, component, parallel=True
            )
            serial = prism_magnetic_component(
                coordinates, prisms, magnetizations, component, parallel=False
            )
        npt.assert_allclose(parallel, serial)

    @run_only_with_numba
    @pytest.mark.parametrize("component", (None, "easting", "northing", "upward"))
    def test_prisms_parallel_vs_serial(self, component):
        """
        Check results of parallelized and serials runs

        Run a large problem only with Numba enabled.
        """
        prisms = [
            [-100, 0, -100, 0, -10, 0],
            [0, 100, -100, 0, -10, 0],
            [-100, 0, 0, 100, -10, 0],
            [0, 100, 0, 100, -10, 0],
        ]
        magnetizations = [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, 5.0],
            [-2.0, 1.0, 3.0],
            [5.0, 4.0, 1.0],
        ]
        coordinates = vd.grid_coordinates(
            region=(-100, 100, -100, 100), spacing=20, extra_coords=10
        )
        if component is None:
            parallel = prism_magnetic(
                coordinates, prisms, magnetizations, parallel=True
            )
            serial = prism_magnetic(coordinates, prisms, magnetizations, parallel=False)
        else:
            parallel = prism_magnetic_component(
                coordinates, prisms, magnetizations, component, parallel=True
            )
            serial = prism_magnetic_component(
                coordinates, prisms, magnetizations, component, parallel=False
            )
        npt.assert_allclose(parallel, serial)


class TestInvalidPrisms:
    """
    Test forward modelling functions against invalid prisms
    """

    boundaries = [
        [100, -100, -100, 100, -200, -100],  # w > e
        [-100, 100, 100, -100, -200, -100],  # s > n
        [-100, 100, -100, -00, -100, -200],  # bottom > top
    ]

    @pytest.fixture()
    def sample_coordinates(self):
        """Return sample coordinates"""
        return [0, 0, 0]

    @pytest.fixture()
    def sample_magnetization(self):
        """Return sample magnetization"""
        return [1.0, 1.0, 1.0]

    @pytest.fixture(params=boundaries)
    def invalid_prism(self, request):
        """Return sample valid prism (with zero and non-zero volume)"""
        return np.atleast_2d(request.param)

    @pytest.mark.parametrize("component", (None, "easting", "northing", "upward"))
    def test_invalid_prisms(
        self, sample_coordinates, invalid_prism, sample_magnetization, component
    ):
        """
        Test forward modelling functions with invalid prisms

        It should raise any error
        """
        msg = "boundary can't be greater than the"
        if component is None:
            with pytest.raises(ValueError, match=msg):
                prism_magnetic(sample_coordinates, invalid_prism, sample_magnetization)
        else:
            with pytest.raises(ValueError, match=msg):
                prism_magnetic_component(
                    sample_coordinates, invalid_prism, sample_magnetization, component
                )

    @pytest.mark.parametrize("component", (None, "easting", "northing", "upward"))
    def test_disable_checks(self, invalid_prism, component):
        """Test if disabling checks doesn't raise errors on invalid prisms"""
        magnetization = [100, 10, -10]
        coordinates = [0, 0, 0]
        if component is None:
            prism_magnetic(
                coordinates, invalid_prism, magnetization, disable_checks=True
            )
        else:
            prism_magnetic_component(
                coordinates,
                invalid_prism,
                magnetization,
                component,
                disable_checks=True,
            )


class TestInvalidMagnetization:
    """
    Test errors after invalid magnetization arrays
    """

    @pytest.fixture()
    def sample_coordinates(self):
        """Return sample coordinates"""
        return [0, 0, 0]

    @pytest.fixture()
    def sample_prisms(self):
        """Return sample prisms"""
        prisms = [
            [-100, 0, -100, 0, -200, -100],
            [-100, 0, 0, 100, -200, -100],
            [0, 100, -100, 0, -200, -100],
            [0, 100, 0, 100, -200, -100],
        ]
        return prisms

    @pytest.mark.parametrize("component", (None, "easting", "northing", "upward"))
    def test_invalid_number_of_vectors(
        self, sample_coordinates, sample_prisms, component
    ):
        """Check error when magnetization has invalid number of vectors"""
        # Generate an array with only two magnetization vectors
        magnetizations = [
            [1.0, 1.0, 1.0],
            [-2.0, 3.0, -5.0],
        ]
        msg = "Number of magnetization vectors"
        if component is None:
            with pytest.raises(ValueError, match=msg):
                prism_magnetic(sample_coordinates, sample_prisms, magnetizations)
        else:
            with pytest.raises(ValueError, match=msg):
                prism_magnetic_component(
                    sample_coordinates, sample_prisms, magnetizations, component
                )

    @pytest.mark.parametrize("component", (None, "easting", "northing", "upward"))
    def test_invalid_number_of_elements(
        self, sample_coordinates, sample_prisms, component
    ):
        """Check error when magnetization has invalid number of elements"""
        # Generate an array with only two magnetization vectors
        magnetizations = [
            [1.0, 1.0, 1.0, 3.4],
            [-2.0, 3.0, -5.0, 3.4],
            [-2.0, 3.0, -5.0, 3.4],
            [-2.0, 3.0, -5.0, 3.4],
        ]
        msg = "Found magnetization vectors with"
        if component is None:
            with pytest.raises(ValueError, match=msg):
                prism_magnetic(sample_coordinates, sample_prisms, magnetizations)
        else:
            with pytest.raises(ValueError, match=msg):
                prism_magnetic_component(
                    sample_coordinates, sample_prisms, magnetizations, component
                )


class TestAgainstChoclo:
    """
    Test forward modelling functions against dumb Choclo runs
    """

    @pytest.fixture()
    def sample_prisms(self):
        """
        Return three sample prisms
        """
        prisms = np.array(
            [
                [-10, 10, -10, 0, -10, 0],
                [-10, 0, -10, 10, -20, -10],
                [5, 15, 5, 15, -15, -5],
                [-5, 5, -5, 5, -35, -20],
            ],
            dtype=float,
        )
        return prisms

    @pytest.fixture()
    def sample_magnetizations(self):
        """
        Return sample magnetization vectors for the prisms
        """
        magnetizations = np.array(
            [
                [1.0, 1.0, 1],
                [-1.0, -0.5, 2.0],
                [3, -1.0, -3.0],
                [-1.5, 2.0, -2.0],
            ],
            dtype=np.float64,
        )
        return magnetizations

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
        self, sample_coordinates, sample_prisms, sample_magnetizations
    ):
        """
        Test prism_magnetic against raw Choclo runs
        """
        easting, northing, upward = sample_coordinates
        # Compute expected results with dumb Choclo runs
        n_coords = easting.size
        n_prisms = sample_prisms.shape[0]
        expected_magnetic_e = np.zeros(n_coords, dtype=np.float64)
        expected_magnetic_n = np.zeros(n_coords, dtype=np.float64)
        expected_magnetic_u = np.zeros(n_coords, dtype=np.float64)
        for i in range(n_coords):
            for j in range(n_prisms):
                b_e, b_n, b_u = magnetic_field(
                    easting[i],
                    northing[i],
                    upward[i],
                    *sample_prisms[j, :],
                    *sample_magnetizations[j, :],
                )
                expected_magnetic_e[i] += b_e
                expected_magnetic_n[i] += b_n
                expected_magnetic_u[i] += b_u
        # Convert to nT
        expected_magnetic_e *= 1e9
        expected_magnetic_n *= 1e9
        expected_magnetic_u *= 1e9
        # Compare with harmonica results
        b_e, b_n, b_u = prism_magnetic(
            sample_coordinates, sample_prisms, sample_magnetizations
        )
        npt.assert_allclose(b_e, expected_magnetic_e)
        npt.assert_allclose(b_n, expected_magnetic_n)
        npt.assert_allclose(b_u, expected_magnetic_u)

    @pytest.mark.parametrize("component", ("easting", "northing", "upward"))
    def test_component_against_choclo(
        self, sample_coordinates, sample_prisms, sample_magnetizations, component
    ):
        """
        Test prism_magnetic_component against raw Choclo runs
        """
        easting, northing, upward = sample_coordinates
        # Compute expected results with dumb Choclo runs
        n_coords = easting.size
        n_prisms = sample_prisms.shape[0]
        expected_result = np.zeros(n_coords, dtype=np.float64)
        forward_func = getattr(choclo.prism, f"magnetic_{component[0]}")
        for i in range(n_coords):
            for j in range(n_prisms):
                expected_result[i] += forward_func(
                    easting[i],
                    northing[i],
                    upward[i],
                    *sample_prisms[j, :],
                    *sample_magnetizations[j, :],
                )
        # Convert to nT
        expected_result *= 1e9
        # Compare with harmonica results
        result = prism_magnetic_component(
            sample_coordinates,
            sample_prisms,
            sample_magnetizations,
            component,
        )
        npt.assert_allclose(result, expected_result)
