"""
Test magnetic forward functions for dipoles
"""
import pytest
import numpy as np
import numpy.testing as npt
import verde as vd
import choclo
from choclo.dipole import magnetic_field

from .utils import run_only_with_numba
from .. import dipole_magnetic, dipole_magnetic_component


def test_invalid_component():
    "Check if passing an invalid component raises an error"
    coordinates = [0, 0, 0]
    dipoles = [[-100, 100], [-100, 100], [-200, -100]]
    magnetic_moments = [[1, 1, 2], [-1, -2, 3]]
    with pytest.raises(ValueError, match="Invalid component"):
        dipole_magnetic_component(
            coordinates, dipoles, magnetic_moments, component="Not a valid field"
        )


class TestInvalidMagneticMoments:
    @pytest.mark.parametrize("component", ("all", "easting", "northing", "upward"))
    def test_magnetic_moments_and_dipoles(self, component):
        """
        Test error when dipoles and magnetic moments mismatch
        """
        coordinates = [0, 0, 0]
        dipoles = [[1, 1], [-1, 0], [-10, -2]]
        # Define invalid magnetic moments (more moments than dipoles)
        magnetic_moments = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
        msg = "Number of elements in magnetic_moments"
        if component == "all":
            with pytest.raises(ValueError, match=msg):
                dipole_magnetic(coordinates, dipoles, magnetic_moments)
        else:
            with pytest.raises(ValueError, match=msg):
                dipole_magnetic_component(
                    coordinates, dipoles, magnetic_moments, component
                )

    @pytest.mark.parametrize("component", ("all", "easting", "northing", "upward"))
    def test_magnetic_moments_components(self, component):
        """
        Test if error is raised when magnetic moments have != 3 components
        """
        coordinates = [0, 0, 0]
        dipoles = [[1, 1], [-1, 0], [-10, -2]]
        # Define invalid magnetic moments (more than three components)
        magnetic_moments = [
            [1, 2, 3, 5],
            [4, 5, 6, 7],
        ]
        msg = "Invalid magnetic moments with '4' elements."
        if component == "all":
            with pytest.raises(ValueError, match=msg):
                dipole_magnetic(coordinates, dipoles, magnetic_moments)
        else:
            with pytest.raises(ValueError, match=msg):
                dipole_magnetic_component(
                    coordinates, dipoles, magnetic_moments, component
                )

    @pytest.mark.parametrize("component", ("all", "easting", "northing", "upward"))
    def test_disable_checks(self, component):
        """
        Test disable checks on invalid magnetic moments
        """
        coordinates = [0, 0, 0]
        dipoles = [[1, 1], [-1, 0], [-10, -2]]
        # Define invalid magnetic moments (more than three components)
        magnetic_moments = [
            [1, 2, 3, 5],
            [4, 5, 6, 7],
        ]
        if component == "all":
            dipole_magnetic(coordinates, dipoles, magnetic_moments, disable_checks=True)
        else:
            dipole_magnetic_component(
                coordinates,
                dipoles,
                magnetic_moments,
                component,
                disable_checks=True,
            )


class TestSerialVsParallel:
    """
    Test serial vs parallel
    """

    @pytest.mark.parametrize("component", ("all", "easting", "northing", "upward"))
    def test_dipoles_parallel_vs_serial_no_numba(self, component):
        """
        Check results of parallelized and serials runs
        Run a small problem with Numba disable to count for test coverage.
        """
        coordinates = ([0, 10], [0, 10], [0, 10])
        dipoles = [[-100, 0], [0, 100], [-20, -50]]
        magnetic_moments = [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, 5.0],
        ]
        if component == "all":
            parallel = dipole_magnetic(
                coordinates, dipoles, magnetic_moments, parallel=True
            )
            serial = dipole_magnetic(
                coordinates, dipoles, magnetic_moments, parallel=False
            )
        else:
            parallel = dipole_magnetic_component(
                coordinates, dipoles, magnetic_moments, component, parallel=True
            )
            serial = dipole_magnetic_component(
                coordinates, dipoles, magnetic_moments, component, parallel=False
            )
        npt.assert_allclose(parallel, serial)

    @run_only_with_numba
    @pytest.mark.parametrize("component", ("all", "easting", "northing", "upward"))
    def test_dipoles_parallel_vs_serial(self, component):
        """
        Check results of parallelized and serials runs
        Run a large problem only with Numba enabled.
        """
        coordinates = vd.grid_coordinates(
            region=(-100, 100, -100, 100), spacing=20, extra_coords=10
        )
        dipoles = [[-100, 0], [0, 100], [-20, -50]]
        magnetic_moments = [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, 5.0],
        ]
        if component == "all":
            parallel = dipole_magnetic(
                coordinates, dipoles, magnetic_moments, parallel=True
            )
            serial = dipole_magnetic(
                coordinates, dipoles, magnetic_moments, parallel=False
            )
        else:
            parallel = dipole_magnetic_component(
                coordinates, dipoles, magnetic_moments, component, parallel=True
            )
            serial = dipole_magnetic_component(
                coordinates, dipoles, magnetic_moments, component, parallel=False
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
        magnetic_moments = np.array(
            [
                [1.0, 1.0, 1],
                [-1.0, -0.5, 2.0],
                [3, -1.0, -3.0],
                [-1.5, 2.0, -2.0],
            ],
            dtype=np.float64,
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
                    *sample_magnetic_moments[j, :],
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
            sample_coordinates, sample_dipoles, sample_magnetic_moments
        )
        npt.assert_allclose(b_e, expected_magnetic_e)
        npt.assert_allclose(b_n, expected_magnetic_n)
        npt.assert_allclose(b_u, expected_magnetic_u)

    @pytest.mark.parametrize("component", ("easting", "northing", "upward"))
    def test_component_against_choclo(
        self, sample_coordinates, sample_dipoles, sample_magnetic_moments, component
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
        forward_func = getattr(choclo.dipole, f"magnetic_{component[0]}")
        for i in range(n_coords):
            for j in range(n_dipoles):
                expected_result[i] += forward_func(
                    easting[i],
                    northing[i],
                    upward[i],
                    easting_p[j],
                    northing_p[j],
                    upward_p[j],
                    *sample_magnetic_moments[j, :],
                )
        # Convert to nT
        expected_result *= 1e9
        # Compare with harmonica results
        result = dipole_magnetic_component(
            sample_coordinates,
            sample_dipoles,
            sample_magnetic_moments,
            component,
        )
        npt.assert_allclose(result, expected_result)
