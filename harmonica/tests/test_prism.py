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
from choclo.prism import (
    gravity_e,
    gravity_ee,
    gravity_en,
    gravity_eu,
    gravity_n,
    gravity_nn,
    gravity_nu,
    gravity_pot,
    gravity_u,
    gravity_uu,
)

try:
    from numba_progress import ProgressBar
except ImportError:
    ProgressBar = None

from .. import bouguer_correction
from .._forward.prism_gravity import _check_prisms, _discard_null_prisms, prism_gravity
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
    # Create a set of 4 prisms
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
    npt.assert_allclose(prisms, np.array([[0.0, 10.0, -50.0, 33.0, -2e3, 150.0]]))
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
            ],
            dtype=float,
        )
        densities = np.array([400, -200, 300], dtype=float)
        return prisms, densities

    @pytest.fixture()
    def sample_coordinates(self):
        """
        Return four sample observation points
        """
        easting = np.array([-5, 10, 0, 15], dtype=float)
        northing = np.array([14, -4, 11, 0], dtype=float)
        upward = np.array([9, 6, 6, 12], dtype=float)
        return (easting, northing, upward)

    @pytest.mark.use_numba
    @pytest.mark.parametrize(
        "field, choclo_func",
        [
            ("potential", gravity_pot),
            ("g_e", gravity_e),
            ("g_n", gravity_n),
            ("g_z", gravity_u),
            ("g_ee", gravity_ee),
            ("g_nn", gravity_nn),
            ("g_zz", gravity_uu),
            ("g_en", gravity_en),
            ("g_ez", gravity_eu),
            ("g_nz", gravity_nu),
        ],
    )
    def test_against_choclo(
        self,
        field,
        choclo_func,
        sample_coordinates,
        sample_prisms,
    ):
        """
        Tests forward functions against dumb runs on Choclo
        """
        easting, northing, upward = sample_coordinates
        prisms, densities = sample_prisms
        # Compute expected results with dumb choclo calls
        expected_result = np.zeros_like(easting)
        for i in range(easting.size):
            for j in range(densities.size):
                expected_result[i] += choclo_func(
                    easting[i],
                    northing[i],
                    upward[i],
                    prisms[j, 0],
                    prisms[j, 1],
                    prisms[j, 2],
                    prisms[j, 3],
                    prisms[j, 4],
                    prisms[j, 5],
                    densities[j],
                )
        if field in ("g_z", "g_ez", "g_nz"):
            expected_result *= -1  # invert sign
        if field in ("g_e", "g_n", "g_z"):
            expected_result *= 1e5  # convert to mGal
        if field in ("g_ee", "g_nn", "g_zz", "g_en", "g_ez", "g_nz"):
            expected_result *= 1e9  # convert to Eotvos
        # Compare with Harmonica results
        result = prism_gravity(sample_coordinates, prisms, densities, field=field)
        npt.assert_allclose(result, expected_result)


@run_only_with_numba
def test_laplace():
    """
    Test if the diagonal components satisfy Laplace equation
    """
    region = (-10e3, 10e3, -10e3, 10e3)
    coords = vd.grid_coordinates(region, shape=(10, 10), extra_coords=300)
    prisms = [
        [1e3, 7e3, -5e3, 2e3, -1e3, -500],
        [-4e3, 1e3, 4e3, 10e3, -2e3, 200],
    ]
    densities = [2670.0, 2900.0]
    diagonal_components = {
        field: prism_gravity(coords, prisms, densities, field=field)
        for field in ("g_ee", "g_nn", "g_zz")
    }
    npt.assert_allclose(
        diagonal_components["g_ee"] + diagonal_components["g_nn"],
        -diagonal_components["g_zz"],
    )


@pytest.mark.use_numba
def test_prism_against_infinite_slab():
    """
    Test if g_z of a large prism matches the solution for an infinite slab
    """
    # Define an observation point at 1.5m above zero
    height = 1.5
    coordinates = (0, 0, height)
    # Define prisms with a top surface located at the same height as the
    # observation point. Each prisms will have a thickness of 10.5m, and
    # horizontal dimensions from 1e3 to 1e9m, and density of 2670 kg/m^3.
    top = height
    thickness = 10.5
    bottom = top - thickness
    sizes = np.logspace(3, 9, 7)
    density = 2670
    # Compute the gravity field generated by each prism
    # (from smaller to larger)
    results = np.zeros_like(sizes)
    for i, size in enumerate(sizes):
        prism = [-size / 2, size / 2, -size / 2, size / 2, bottom, top]
        results[i] = prism_gravity(coordinates, prism, density, field="g_z")
    # Check convergence: assert if as the prism size increases, the result gets
    # closer to the analytical solution for an infinite slab
    analytical = bouguer_correction(np.array(thickness), density)
    errors = abs(analytical - results)
    assert (errors[1:] < errors[:-1]).all()
    # Check if the largest size is close enough to the analytical solution
    npt.assert_allclose(analytical, results[-1])


@pytest.mark.use_numba
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
        region=(-100, 100, -100, 100), shape=(3, 3), extra_coords=10
    )
    for field in ("potential", "g_z"):
        result_parallel = prism_gravity(
            coordinates, prisms, densities, field=field, parallel=True
        )
        result_serial = prism_gravity(
            coordinates, prisms, densities, field=field, parallel=False
        )
        npt.assert_allclose(result_parallel, result_serial)


class TestProgressBar:
    @pytest.fixture
    def prisms(self):
        """Sample prisms"""
        prisms = [
            [-100, 0, -100, 0, -10, 0],
            [0, 100, -100, 0, -10, 0],
            [-100, 0, 0, 100, -10, 0],
            [0, 100, 0, 100, -10, 0],
        ]
        return prisms

    @pytest.fixture
    def densities(self):
        """Sample densities"""
        return [2000, 3000, 4000, 5000]

    @pytest.fixture
    def coordinates(self):
        """Sample coordinates"""
        coordinates = vd.grid_coordinates(
            region=(-100, 100, -100, 100), spacing=20, extra_coords=10
        )
        return coordinates

    @pytest.mark.skipif(ProgressBar is None, reason="requires numba_progress")
    @pytest.mark.use_numba
    @pytest.mark.parametrize(
        "field",
        [
            "potential",
            "g_e",
            "g_n",
            "g_z",
            "g_ee",
            "g_nn",
            "g_zz",
            "g_en",
            "g_ez",
            "g_nz",
        ],
    )
    def test_progress_bar(self, coordinates, prisms, densities, field):
        """
        Check if forward gravity results with and without progress bar match
        """
        result_progress_true = prism_gravity(
            coordinates, prisms, densities, field=field, progressbar=True
        )
        result_progress_false = prism_gravity(
            coordinates, prisms, densities, field=field, progressbar=False
        )
        npt.assert_allclose(result_progress_true, result_progress_false)

    @patch("harmonica._forward.utils.ProgressBar", None)
    def test_numba_progress_missing_error(self, coordinates, prisms, densities):
        """
        Check if error is raised when progresbar=True and numba_progress
        package is not installed.
        """
        # Check if error is raised
        with pytest.raises(ImportError):
            prism_gravity(
                coordinates, prisms, densities, field="potential", progressbar=True
            )


class TestSingularPoints:
    """
    Tests tensor components on singular points of the prism
    """

    @pytest.fixture
    def sample_prism(self):
        """Return a sample prism"""
        return np.array([-10.3, 5.4, 8.6, 14.3, -30.3, 2.4])

    @pytest.fixture
    def sample_density(self):
        """Return a sample density for the sample prism"""
        return np.array([2900.0])

    def get_vertices(self, prism):
        """
        Return the vertices of the prism as points
        """
        easting, northing, upward = tuple(
            c.ravel() for c in np.meshgrid(prism[:2], prism[2:4], prism[4:6])
        )
        return easting, northing, upward

    def get_easting_edges_center(self, prism):
        """
        Return points on the center of prism edges parallel to easting
        """
        easting_c = (prism[0] + prism[1]) / 2
        northing, upward = tuple(c.ravel() for c in np.meshgrid(prism[2:4], prism[4:6]))
        easting = np.full_like(northing, easting_c)
        return easting, northing, upward

    def get_northing_edges_center(self, prism):
        """
        Return points on the center of prism edges parallel to northing
        """
        northing_c = (prism[2] + prism[3]) / 2
        easting, upward = tuple(c.ravel() for c in np.meshgrid(prism[0:2], prism[4:6]))
        northing = np.full_like(easting, northing_c)
        return easting, northing, upward

    def get_upward_edges_center(self, prism):
        """
        Return points on the center of prism edges parallel to upward
        """
        upward_c = (prism[4] + prism[5]) / 2
        easting, northing = tuple(
            c.ravel() for c in np.meshgrid(prism[0:2], prism[2:4])
        )
        upward = np.full_like(easting, upward_c)
        return easting, northing, upward

    @pytest.mark.use_numba
    @pytest.mark.parametrize("field", ("g_ee", "g_nn", "g_zz", "g_en", "g_ez", "g_nz"))
    def test_on_vertices(self, sample_prism, sample_density, field):
        """
        Test tensor components when observation points fall on prism vertices
        """
        easting, northing, upward = self.get_vertices(sample_prism)
        for i in range(easting.size):
            msg = "Found observation point"
            with pytest.warns(UserWarning, match=msg):
                prism_gravity(
                    (easting[i], northing[i], upward[i]),
                    sample_prism,
                    sample_density,
                    field=field,
                )

    @pytest.mark.use_numba
    @pytest.mark.parametrize("field", ("g_nn", "g_zz", "g_nz"))
    def test_on_easting_edges(self, sample_prism, sample_density, field):
        """
        Test tensor components that have singular points on edges parallel to
        easting direction
        """
        easting, northing, upward = self.get_easting_edges_center(sample_prism)
        for i in range(easting.size):
            msg = "Found observation point"
            with pytest.warns(UserWarning, match=msg):
                prism_gravity(
                    (easting[i], northing[i], upward[i]),
                    sample_prism,
                    sample_density,
                    field=field,
                )

    @pytest.mark.use_numba
    @pytest.mark.parametrize("field", ("g_ee", "g_zz", "g_ez"))
    def test_on_northing_edges(self, sample_prism, sample_density, field):
        """
        Test tensor components that have singular points on edges parallel to
        easting direction
        """
        easting, northing, upward = self.get_northing_edges_center(sample_prism)
        for i in range(easting.size):
            msg = "Found observation point"
            with pytest.warns(UserWarning, match=msg):
                prism_gravity(
                    (easting[i], northing[i], upward[i]),
                    sample_prism,
                    sample_density,
                    field=field,
                )

    @pytest.mark.use_numba
    @pytest.mark.parametrize("field", ("g_ee", "g_nn", "g_en"))
    def test_on_upward_edges(self, sample_prism, sample_density, field):
        """
        Test tensor components that have singular points on edges parallel to
        easting direction
        """
        easting, northing, upward = self.get_upward_edges_center(sample_prism)
        for i in range(easting.size):
            msg = "Found observation point"
            with pytest.warns(UserWarning, match=msg):
                prism_gravity(
                    (easting[i], northing[i], upward[i]),
                    sample_prism,
                    sample_density,
                    field=field,
                )
