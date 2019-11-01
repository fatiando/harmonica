"""
Test forward modelling for point masses.
"""
import numpy as np
import numpy.testing as npt
import pytest

from ..constants import GRAVITATIONAL_CONST
from ..forward.point_mass import point_mass_gravity


def test_invalid_coordinate_system():
    "Check if invalid coordinate system is passed"
    coordinates = [0.0, 0.0, 0.0]
    point_mass = [0.0, 0.0, 0.0]
    mass = 1.0
    with pytest.raises(ValueError):
        point_mass_gravity(
            coordinates,
            point_mass,
            mass,
            "potential",
            "this-is-not-a-valid-coordinate-system",
        )


def test_invalid_field():
    "Check if an invalid gravitational field is passed as argument"
    coordinates = [0.0, 0.0, 0.0]
    point_mass = [0.0, 0.0, 0.0]
    mass = 1.0
    for coordinate_system in ("spherical", "cartesian"):
        with pytest.raises(ValueError):
            point_mass_gravity(
                coordinates,
                point_mass,
                mass,
                "this-field-does-not-exist",
                coordinate_system,
            )


def test_invalid_masses_array():
    "Check if error is raised when masses shape does not match points shape"
    # Create a set of 3 point masses
    points = [[-10, 0, 10], [-10, 0, 10], [-100, 0, 100]]
    # Generate a two element masses
    masses = [1000, 2000]
    coordinates = [0, 0, 250]
    with pytest.raises(ValueError):
        point_mass_gravity(
            coordinates,
            points,
            masses,
            field="potential",
            coordinate_system="cartesian",
        )


# ---------------------------
# Cartesian coordinates tests
# ---------------------------
@pytest.mark.use_numba
def test_potential_cartesian_known_values():
    """
    Compare the computed gravitational potential with reference values
    """
    # Define a single point mass
    point_mass = [0, 0, 0]
    mass = [5000]
    # Define a set of computation points
    northing = np.array(
        [
            -914.12,
            -652.35,
            633.73,
            -643.82,
            124.59,
            772.67,
            -849.78,
            397.73,
            -666.84,
            -150.87,
            -232.81,
            344.82,
            -891.76,
            -58.66,
            -326.17,
            544.99,
            931.25,
            -791.44,
            363.22,
            -315.11,
            -745.99,
            165.01,
            -972.92,
            238.07,
            596.43,
            195.3,
            -658.83,
            352.86,
            624.52,
            -895.8,
            418.84,
            190.65,
            -685.8,
            210.13,
            509.92,
            -260.09,
            -608.32,
            655.48,
            108.78,
            848.57,
            790.73,
            -817.8,
            814.28,
            -222.24,
            500.66,
            245.38,
            -925.32,
            -216.85,
            878.32,
            -930.67,
        ]
    )
    easting = np.array(
        [
            -731.48,
            -732.51,
            -189.75,
            -665.44,
            183.86,
            -325.31,
            198.43,
            846.9,
            848.24,
            -267.81,
            -661.11,
            138.29,
            56.59,
            382.3,
            -160.45,
            -5.61,
            -248.78,
            -244.93,
            25.01,
            784.87,
            -54.85,
            113.78,
            635.48,
            -594.94,
            -780.07,
            -710.5,
            398.46,
            -558.47,
            -695.67,
            483.55,
            -420.88,
            302.05,
            391.75,
            -191.12,
            -120.76,
            -896.24,
            -25.99,
            68.08,
            -544.59,
            398.22,
            661.75,
            157.35,
            -173.68,
            -825.65,
            -294.04,
            977.53,
            789.2,
            -614.47,
            -51.75,
            709.44,
        ]
    )
    upward = np.array(
        [
            166.38,
            578.28,
            756.02,
            -629.66,
            805.35,
            220.9,
            236.55,
            575.03,
            987.42,
            -405.96,
            -860.72,
            491.83,
            635.64,
            -690.93,
            -655.83,
            -661.36,
            919.24,
            -355.04,
            -358.92,
            256.77,
            -655.07,
            247.96,
            -406.07,
            -32.5,
            947.99,
            -761.66,
            298.92,
            804.15,
            -398.28,
            -712.59,
            414.7,
            -391.79,
            -482.91,
            -915.3,
            -728.96,
            -593.68,
            -42.72,
            318.34,
            -840.85,
            -696.34,
            567.13,
            -357.39,
            -593.51,
            -397.15,
            244.77,
            -948.18,
            -924.42,
            545.72,
            -841.45,
            710.26,
        ]
    )
    coordinates = [easting, northing, upward]
    reference_values = GRAVITATIONAL_CONST * np.array(
        [
            4.22824753,
            4.39113897,
            4.97719444,
            4.46537816,
            5.98505977,
            5.7671917,
            5.53016006,
            4.55281977,
            3.41858071,
            9.81927076,
            4.50449382,
            8.11190678,
            4.55965201,
            6.31456826,
            6.66818386,
            5.83433105,
            3.75386353,
            5.54727035,
            9.7799273,
            5.65686851,
            5.0286791,
            15.68189263,
            4.06181884,
            7.79267086,
            3.66330057,
            4.7180766,
            6.05369243,
            4.80445931,
            4.92041875,
            4.02381061,
            6.90366092,
            9.43092382,
            5.40109329,
            5.21724138,
            5.56937904,
            4.52058707,
            8.19172875,
            6.83184261,
            4.96184112,
            4.2818961,
            4.24889441,
            5.51726015,
            4.8900585,
            5.30351633,
            7.93518541,
            3.61331522,
            3.27307706,
            5.88272903,
            4.10697576,
            3.65253479,
        ]
    )
    # Compute potential gravity field on each computation point
    results = point_mass_gravity(
        coordinates, point_mass, mass, "potential", "cartesian"
    )
    npt.assert_allclose(results, reference_values)


@pytest.mark.use_numba
def test_potential_cartesian_symmetry():
    """
    Test if potential field of a point mass has symmetry in Cartesian coordinates
    """
    # Define a single point mass
    point_mass = [1.1, 1.2, 1.3]
    masses = [2670]
    # Define a set of computation points at a fixed distance from the point mass
    distance = 3.3
    easting = point_mass[0] * np.ones(6)
    northing = point_mass[1] * np.ones(6)
    upward = point_mass[2] * np.ones(6)
    easting[0] += distance
    easting[1] -= distance
    northing[2] += distance
    northing[3] -= distance
    upward[4] += distance
    upward[5] -= distance
    coordinates = [easting, northing, upward]
    # Compute potential gravity field on each computation point
    results = point_mass_gravity(
        coordinates, point_mass, masses, "potential", "cartesian"
    )
    npt.assert_allclose(*results)


@pytest.mark.use_numba
def test_g_z_symmetry():
    """
    Test if g_z field of a point mass has symmetry in Cartesian coordinates
    """
    # Define a single point mass
    point_mass = [1.1, 1.2, 1.3]
    masses = [2670]
    # Define a pair of computation points above and bellow the point mass
    distance = 3.3
    easting = point_mass[0] * np.ones(2)
    northing = point_mass[1] * np.ones(2)
    upward = point_mass[2] * np.ones(2)
    upward[0] += distance
    upward[1] -= distance
    coordinates = [easting, northing, upward]
    # Compute g_z gravity field on each computation point
    results = point_mass_gravity(coordinates, point_mass, masses, "g_z", "cartesian")
    npt.assert_allclose(results[0], -results[1])


# ---------------------------
# Spherical coordinates tests
# ---------------------------
@pytest.mark.use_numba
def test_point_mass_on_origin():
    "Check potential and g_z of point mass on origin in spherical coordinates"
    point_mass = [0.0, 0.0, 0.0]
    mass = 2.0
    radius = np.logspace(1, 8, 5)
    longitude = np.linspace(-180, 180, 37)
    latitude = np.linspace(-90, 90, 19)
    longitude, latitude, radius = np.meshgrid(longitude, latitude, radius)
    # Analytical solutions (accelerations are in mgal and tensor components in eotvos)
    analytical = {
        "potential": GRAVITATIONAL_CONST * mass / radius,
        "g_z": GRAVITATIONAL_CONST * mass / radius ** 2 * 1e5,
    }
    # Compare results with analytical solutions
    for field in analytical:
        npt.assert_allclose(
            point_mass_gravity(
                [longitude, latitude, radius], point_mass, mass, field, "spherical"
            ),
            analytical[field],
        )


@pytest.mark.use_numba
def test_point_mass_same_radial_direction():
    "Check potential and g_z of point mass and computation point on same radius"
    sphere_radius = 3.0
    mass = 2.0
    for longitude in np.linspace(-180, 180, 37):
        for latitude in np.linspace(-90, 90, 19):
            for height in np.logspace(0, 4, 5):
                point_mass = [longitude, latitude, sphere_radius]
                coordinates = [
                    np.array(longitude),
                    np.array(latitude),
                    np.array(height + sphere_radius),
                ]
                # Analytical solutions
                # (accelerations are in mgal and tensor components in eotvos)
                analytical = {
                    "potential": GRAVITATIONAL_CONST * mass / height,
                    "g_z": GRAVITATIONAL_CONST * mass / height ** 2 * 1e5,
                }
                # Compare results with analytical solutions
                for field in analytical:
                    npt.assert_allclose(
                        point_mass_gravity(
                            coordinates, point_mass, mass, field, "spherical"
                        ),
                        analytical[field],
                    )


@pytest.mark.use_numba
def test_point_mass_potential_on_equator():
    "Check potential field on equator and same radial coordinate"
    radius = 3.0
    mass = 2.0
    latitude = 0.0
    for longitude_p in np.linspace(0, 350, 36):
        point_mass = [longitude_p, latitude, radius]
        for longitude in np.linspace(0, 350, 36):
            if longitude != longitude_p:
                coordinates = [
                    np.array(longitude),
                    np.array(latitude),
                    np.array(radius),
                ]
                # Analytical solutions
                # (accelerations are in mgal and tensor components in eotvos)
                distance = (
                    2 * radius * np.sin(0.5 * np.radians(abs(longitude - longitude_p)))
                )
                analytical = {"potential": GRAVITATIONAL_CONST * mass / distance}
                # Compare results with analytical solutions
                npt.assert_allclose(
                    point_mass_gravity(
                        coordinates, point_mass, mass, "potential", "spherical"
                    ),
                    analytical["potential"],
                )


@pytest.mark.use_numba
def test_point_mass_potential_on_same_meridian():
    "Check potential field on same meridian and radial coordinate"
    radius = 3.0
    mass = 2.0
    longitude = 0.0
    for latitude_p in np.linspace(-90, 90, 19):
        point_mass = [longitude, latitude_p, radius]
        for latitude in np.linspace(-90, 90, 19):
            if latitude != latitude_p:
                coordinates = [
                    np.array(longitude),
                    np.array(latitude),
                    np.array(radius),
                ]
                # Analytical solutions
                # (accelerations are in mgal and tensor components in eotvos)
                distance = (
                    2 * radius * np.sin(0.5 * np.radians(abs(latitude - latitude_p)))
                )
                analytical = {"potential": GRAVITATIONAL_CONST * mass / distance}
                # Compare results with analytical solutions
                npt.assert_allclose(
                    point_mass_gravity(
                        coordinates, point_mass, mass, "potential", "spherical"
                    ),
                    analytical["potential"],
                )
