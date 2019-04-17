"""
Test forward modellig for point masses.
"""
import numpy as np
import numpy.testing as npt
from pytest import raises

from harmonica.constants import GRAVITATIONAL_CONST
from ..point_mass import point_mass_gravity


def test_invalid_field():
    point_mass = [0.0, 0.0, 0.0]
    mass = 1.0
    longitude = np.array(0.0)
    latitude = np.array(0.0)
    height = np.array(0.0)
    with raises(ValueError):
        point_mass_gravity(
            [longitude, latitude, height], point_mass, mass, "this-field-does-not-exist"
        )


def test_point_mass_on_origin():
    "Check potential and gz of point mass on origin"
    point_mass = [0.0, 0.0, 0.0]
    mass = 1.0
    radius = np.logspace(1, 8, 5, dtype="float64")
    longitude = np.linspace(-180, 180, 37)
    latitude = np.linspace(-90, 90, 19)
    longitude, latitude, radius = np.meshgrid(longitude, latitude, radius)
    potential = point_mass_gravity(
        [longitude, latitude, radius], point_mass, mass, "potential"
    )
    potential_analytical = GRAVITATIONAL_CONST * mass / radius
    npt.assert_allclose(potential, potential_analytical)
    gz = point_mass_gravity([longitude, latitude, radius], point_mass, mass, "gz")
    gz_analytical = -GRAVITATIONAL_CONST * mass / radius ** 2
    # Convert to mGal
    gz_analytical *= 1e5
    npt.assert_allclose(gz, gz_analytical)


def test_point_mass_same_radial_direction():
    "Check potential, gz, gzz of point mass and computation point on same radius"
    sphere_radius = 1.0
    mass = 1.0
    for longitude in np.linspace(-180, 180, 37):
        for latitude in np.linspace(-90, 90, 19):
            for height in np.logspace(0, 4, 5):
                point_mass = [longitude, latitude, sphere_radius]
                coordinates = [
                    np.array(longitude),
                    np.array(latitude),
                    np.array(height + sphere_radius),
                ]
                potential = point_mass_gravity(
                    coordinates, point_mass, mass, "potential"
                )
                # Check potential
                potential_analytical = GRAVITATIONAL_CONST * mass / height
                npt.assert_allclose(potential, potential_analytical)
                # Check gz
                gz = point_mass_gravity(coordinates, point_mass, mass, "gz")
                gz_analytical = -GRAVITATIONAL_CONST * mass / height ** 2
                gz_analytical *= 1e5  # convert to mgal
                npt.assert_allclose(gz, gz_analytical)
                # Check gzz
                gzz = point_mass_gravity(coordinates, point_mass, mass, "gzz")
                gzz_analytical = 2 * GRAVITATIONAL_CONST * mass / height ** 3
                gzz_analytical *= 1e9  # convert to eotvos
                npt.assert_allclose(gzz, gzz_analytical)


def test_point_mass_potential_on_equator():
    "Check potential field on equator and same radial coordinate"
    radius = 1.0
    mass = 1.0
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
                potential = point_mass_gravity(
                    coordinates, point_mass, mass, "potential"
                )
                # Calculate analytical solution for this potential
                distance = (
                    2 * radius * np.sin(0.5 * np.radians(abs(longitude - longitude_p)))
                )
                potential_analytical = GRAVITATIONAL_CONST * mass / distance
                npt.assert_allclose(potential, potential_analytical)


def test_point_mass_potential_on_same_meridian():
    "Check potential field on same meridian and radial coordinate"
    radius = 1.0
    mass = 1.0
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
                potential = point_mass_gravity(
                    coordinates, point_mass, mass, "potential"
                )
                # Calculate analytical solution for this potential
                distance = (
                    2 * radius * np.sin(0.5 * np.radians(abs(latitude - latitude_p)))
                )
                potential_analytical = GRAVITATIONAL_CONST * mass / distance
                npt.assert_allclose(potential, potential_analytical)


def test_point_mass_gy_on_equator():
    "Check gy field on equator and same radial coordinate"
    radius = 1.0
    mass = 1.0
    latitude = 0.0
    longitude_p = 0.0
    point_mass = [longitude_p, latitude, radius]
    for delta_longitude in np.linspace(-90, 90, 19):
        longitude = longitude_p + delta_longitude
        if longitude != longitude_p:
            coordinates = [np.array(longitude), np.array(latitude), np.array(radius)]
            gy = point_mass_gravity(coordinates, point_mass, mass, "gy")
            # Calculate analytical solution for gy
            distance = 2 * radius * np.sin(0.5 * np.radians(abs(delta_longitude)))
            gy_analytical = (
                GRAVITATIONAL_CONST
                * mass
                / distance ** 2
                * np.cos(0.5 * np.radians(delta_longitude))
                * np.sign(-delta_longitude)
            )
            gy_analytical *= 1e5
            npt.assert_allclose(gy, gy_analytical)


def test_point_mass_gx_on_same_meridian():
    "Check gx field on same meridian and radial coordinate"
    radius = 1.0
    mass = 1.0
    longitude = 0.0
    latitude_p = 0.0
    point_mass = [longitude, latitude_p, radius]
    for delta_latitude in np.linspace(-90, 90, 19):
        latitude = latitude_p + delta_latitude
        if latitude != latitude_p:
            coordinates = [np.array(longitude), np.array(latitude), np.array(radius)]
            gx = point_mass_gravity(coordinates, point_mass, mass, "gx")
            # Calculate analytical solution for gy
            distance = 2 * radius * np.sin(0.5 * np.radians(abs(delta_latitude)))
            gx_analytical = (
                GRAVITATIONAL_CONST
                * mass
                / distance ** 2
                * np.cos(0.5 * np.radians(delta_latitude))
                * np.sign(-delta_latitude)
            )
            gx_analytical *= 1e5
            npt.assert_allclose(gx, gx_analytical)
