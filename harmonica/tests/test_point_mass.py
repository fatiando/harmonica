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


# ---------------------------
# Cartesian coordinates tests
# ---------------------------
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
def test_potential_versus_g_easting():
    """
    Test if the g_easting can be obtained numerically from potential
    """
    # Define a single point mass
    point_mass = [20, 54, -500.7]
    mass = [200]
    # Compute the easting component
    g_easting = point_mass_gravity(
        [[0], [0], [0]], point_mass, mass, "g_easting", "cartesian"
    )
    # Compute the numerical derivative of potential
    delta = 0.1
    easting = np.array([-delta, delta])
    northing = np.array([0, 0])
    upward = np.array([0, 0])
    coordinates = [easting, northing, upward]
    potential = point_mass_gravity(
        coordinates, point_mass, mass, "potential", "cartesian"
    )
    derivative_easting = 1e5 * (potential[1] - potential[0]) / (2.0 * delta)

    # Compare the results
    npt.assert_allclose(g_easting, derivative_easting)


@pytest.mark.use_numba
def test_potential_versus_g_northing():
    """
    Test if the g_northing can be obtained numerically from potential
    """
    # Define a single point mass
    point_mass = [-30, 10, -500.7]
    mass = [200]
    # Compute the northing component
    g_northing = point_mass_gravity(
        [[0], [0], [0]], point_mass, mass, "g_northing", "cartesian"
    )
    # Compute the numerical derivative of potential
    delta = 0.1
    easting = np.array([0, 0])
    northing = np.array([-delta, delta])
    upward = np.array([0, 0])
    coordinates = [easting, northing, upward]
    potential = point_mass_gravity(
        coordinates, point_mass, mass, "potential", "cartesian"
    )
    derivative_northing = 1e5 * (potential[1] - potential[0]) / (2.0 * delta)

    # Compare the results
    npt.assert_allclose(g_northing, derivative_northing)


@pytest.mark.use_numba
def test_potential_versus_g_z():
    """
    Test if the g_z can be obtained numerically from potential
    """
    # Define a single point mass
    point_mass = [-3, 51, -500.7]
    mass = [200]
    # Compute the z component
    g_z = point_mass_gravity([[0], [0], [0]], point_mass, mass, "g_z", "cartesian")
    # Compute the numerical derivative of potential
    delta = 0.1
    easting = np.array([0, 0])
    northing = np.array([0, 0])
    upward = np.array([-delta, delta])
    coordinates = [easting, northing, upward]
    potential = point_mass_gravity(
        coordinates, point_mass, mass, "potential", "cartesian"
    )
    derivative_z = 1e5 * (potential[1] - potential[0]) / (2.0 * delta)

    # Compare the results
    npt.assert_allclose(g_z, -derivative_z)


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


@pytest.mark.use_numba
def test_g_z_sign():
    """
    Test if g_z field of a positive point mass has the correct sign
    """
    # Define a single point mass
    point_mass = [-10, 100.2, -300.7]
    mass = [2670]
    # Define three computation points located above, at the same depth and
    # bellow the point mass
    easting = np.zeros(3)
    northing = np.zeros(3) + 52.3
    upward = np.array([100.11, -300.7, -400])
    coordinates = [easting, northing, upward]
    # Compute g_z gravity field on each computation point
    results = point_mass_gravity(coordinates, point_mass, mass, "g_z", "cartesian")
    assert np.sign(mass) == np.sign(results[0])
    npt.assert_allclose(results[1], 0)
    assert np.sign(mass) == -np.sign(results[2])


@pytest.mark.use_numba
def test_g_northing_symmetry():
    """
    Test if g_northing field of a point mass has symmetry in Cartesian coordinates
    """
    # Define a single point mass
    point_mass = [-7.9, 25, -130]
    masses = [2670]
    # Define a pair of computation points northward and southward the point mass
    distance = 6.1
    easting = point_mass[0] + np.zeros(2)
    northing = point_mass[1] + np.zeros(2)
    upward = point_mass[2] + np.zeros(2)
    northing[0] += distance
    northing[1] -= distance
    coordinates = [easting, northing, upward]
    # Compute g_northing gravity field on each computation point
    results = point_mass_gravity(
        coordinates, point_mass, masses, "g_northing", "cartesian"
    )
    npt.assert_allclose(results[0], -results[1])


@pytest.mark.use_numba
def test_g_northing_sign():
    """
    Test if g_northing field of a positive point mass has the correct sign
    """
    # Define a single point mass
    point_mass = [-10, 100.2, -300.7]
    mass = [2670]
    # Define three computation points located above the point mass, along the
    # north axis
    easting = np.zeros(3)
    northing = np.array([0, 100.2, 210.7])
    upward = np.zeros(3)
    coordinates = [easting, northing, upward]
    # Compute g_northing gravity field on each computation point
    results = point_mass_gravity(
        coordinates, point_mass, mass, "g_northing", "cartesian"
    )
    assert np.sign(mass) == np.sign(results[0])
    npt.assert_allclose(results[1], 0)
    assert np.sign(mass) == -np.sign(results[2])


@pytest.mark.use_numba
def test_g_easting_symmetry():
    """
    Test if g_easting field of a point mass has symmetry in Cartesian coordinates
    """
    # Define a single point mass
    point_mass = [191, -5, 0]
    masses = [2670]
    # Define a pair of computation points northward and southward the point mass
    distance = 4.6
    easting = point_mass[0] + np.zeros(2)
    northing = point_mass[1] + np.zeros(2)
    upward = point_mass[2] + np.zeros(2)
    easting[0] += distance
    easting[1] -= distance
    coordinates = [easting, northing, upward]
    # Compute g_easting gravity field on each computation point
    results = point_mass_gravity(
        coordinates, point_mass, masses, "g_easting", "cartesian"
    )
    npt.assert_allclose(results[0], -results[1])


@pytest.mark.use_numba
def test_g_easting_sign():
    """
    Test if g_easting field of a positive point mass has the correct sign
    """
    # Define a single point mass
    point_mass = [-10, 100.2, -300.7]
    mass = [2670]
    # Define three computation points located above the point mass, along the
    # east axis
    easting = np.array([-150.7, -10, 79])
    northing = np.zeros(3)
    upward = np.zeros(3)
    coordinates = [easting, northing, upward]
    # Compute g_easting gravity field on each computation point
    results = point_mass_gravity(
        coordinates, point_mass, mass, "g_easting", "cartesian"
    )
    assert np.sign(mass) == np.sign(results[0])
    npt.assert_allclose(results[1], 0)
    assert np.sign(mass) == -np.sign(results[2])


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
