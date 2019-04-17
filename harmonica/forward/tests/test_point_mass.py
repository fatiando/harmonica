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
    "Check potential, gz and gzz of point mass on origin"
    point_mass = [0.0, 0.0, 0.0]
    mass = 1.0
    radius = np.logspace(1, 8, 5, dtype="float64")
    longitude = np.linspace(-180, 180, 37)
    latitude = np.linspace(-90, 90, 19)
    longitude, latitude, radius = np.meshgrid(longitude, latitude, radius)
    # Analytical solutions (accelerations are in mgal and tensor components in eotvos)
    analytical = {
        "potential": GRAVITATIONAL_CONST * mass / radius,
        "gz": -GRAVITATIONAL_CONST * mass / radius ** 2 * 1e5,
        "gzz": 2 * GRAVITATIONAL_CONST * mass / radius ** 3 * 1e9,
    }
    # Compare results with analytical solutions
    for field in analytical:
        npt.assert_allclose(
            point_mass_gravity([longitude, latitude, radius], point_mass, mass, field),
            analytical[field],
        )


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
                # Analytical solutions
                # (accelerations are in mgal and tensor components in eotvos)
                analytical = {
                    "potential": GRAVITATIONAL_CONST * mass / height,
                    "gz": -GRAVITATIONAL_CONST * mass / height ** 2 * 1e5,
                    "gzz": 2 * GRAVITATIONAL_CONST * mass / height ** 3 * 1e9,
                }
                # Compare results with analytical solutions
                for field in analytical:
                    npt.assert_allclose(
                        point_mass_gravity(coordinates, point_mass, mass, field),
                        analytical[field],
                    )


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
                # Analytical solutions
                # (accelerations are in mgal and tensor components in eotvos)
                distance = (
                    2 * radius * np.sin(0.5 * np.radians(abs(longitude - longitude_p)))
                )
                analytical = {"potential": GRAVITATIONAL_CONST * mass / distance}
                # Compare results with analytical solutions
                npt.assert_allclose(
                    point_mass_gravity(coordinates, point_mass, mass, "potential"),
                    analytical["potential"],
                )


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
                # Analytical solutions
                # (accelerations are in mgal and tensor components in eotvos)
                distance = (
                    2 * radius * np.sin(0.5 * np.radians(abs(latitude - latitude_p)))
                )
                analytical = {"potential": GRAVITATIONAL_CONST * mass / distance}
                # Compare results with analytical solutions
                npt.assert_allclose(
                    point_mass_gravity(coordinates, point_mass, mass, "potential"),
                    analytical["potential"],
                )


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
            # Analytical solutions
            # (accelerations are in mgal and tensor components in eotvos)
            distance = 2 * radius * np.sin(0.5 * np.radians(abs(delta_longitude)))
            analytical = {
                "gy": (
                    GRAVITATIONAL_CONST
                    * mass
                    / distance ** 2
                    * np.cos(0.5 * np.radians(delta_longitude))
                    * np.sign(-delta_longitude)
                    * 1e5
                )
            }
            # Compare results with analytical solutions
            npt.assert_allclose(
                point_mass_gravity(coordinates, point_mass, mass, "gy"),
                analytical["gy"],
            )


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
            # Analytical solutions
            # (accelerations are in mgal and tensor components in eotvos)
            distance = 2 * radius * np.sin(0.5 * np.radians(abs(delta_latitude)))
            analytical = {
                "gx": (
                    GRAVITATIONAL_CONST
                    * mass
                    / distance ** 2
                    * np.cos(0.5 * np.radians(delta_latitude))
                    * np.sign(-delta_latitude)
                    * 1e5
                )
            }
            # Compare results with analytical solutions
            npt.assert_allclose(
                point_mass_gravity(coordinates, point_mass, mass, "gx"),
                analytical["gx"],
            )
