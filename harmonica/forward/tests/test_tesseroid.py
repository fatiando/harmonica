"""
Test forward modellig for point masses.
"""
import numpy as np
import numpy.testing as npt
from pytest import raises

from ..tesseroid import (
    _distance_tesseroid_point,
    _tesseroid_dimensions,
    _split_tesseroid,
    adaptive_discretization,
)


def test_distance_tesseroid_point():
    "Test distance between tesseroid and computation point"
    longitude_p, latitude_p, radius_p = 0.0, 0.0, 1.0
    d_lon, d_lat, d_radius = 2.0, 2.0, 1.0
    tesseroid = [
        longitude_p - d_lon / 2,
        longitude_p + d_lon / 2,
        latitude_p - d_lat / 2,
        latitude_p + d_lat / 2,
        radius_p - d_radius / 2,
        radius_p + d_radius / 2,
    ]
    # Computation point on the center of the tesseroid
    point = [longitude_p, latitude_p, radius_p]
    npt.assert_allclose(_distance_tesseroid_point(point, tesseroid), 0.0)
    # Computation point and tesseroid center with same longitude and latitude
    radius = radius_p + 1.0
    point = [longitude_p, latitude_p, radius]
    npt.assert_allclose(_distance_tesseroid_point(point, tesseroid), 1.0)
    # Computation point and tesseroid center on equator and same radius
    longitude = 3.0
    point = [longitude, latitude_p, radius_p]
    distance = 2 * radius_p * np.sin(0.5 * np.radians(abs(longitude - longitude_p)))
    npt.assert_allclose(_distance_tesseroid_point(point, tesseroid), distance)
    # Computation point and tesseroid center on same meridian and radius
    latitude = 3.0
    point = [longitude_p, latitude, radius_p]
    distance = 2 * radius_p * np.sin(0.5 * np.radians(abs(latitude - latitude_p)))
    npt.assert_allclose(_distance_tesseroid_point(point, tesseroid), distance)


def test_tesseroid_dimensions():
    "Test calculation of tesseroid dimensions"
    # Tesseroid on equator
    w, e, s, n, bottom, top = -1.0, 1.0, -1.0, 1.0, 0.5, 1.5
    tesseroid = [w, e, s, n, bottom, top]
    L_lon, L_lat = top * np.radians(abs(e - w)), top * np.radians(abs(n - s))
    L_r = top - bottom
    npt.assert_allclose((L_lon, L_lat, L_r), _tesseroid_dimensions(tesseroid))


def test_split_tesseroid_only_longitude():
    "Test splitting of a tesseroid only on longitude"
    lon_indexes = [0, 1]
    lat_indexes = [2, 3]
    radial_indexes = [4, 5]
    w, e, s, n, bottom, top = -10.0, 10.0, -10.0, 10.0, 1.0, 10.0
    tesseroid = [w, e, s, n, bottom, top]
    # Split only on longitude
    stack = np.zeros((8, 6))
    stack_top = -1
    stack_top = _split_tesseroid(
        tesseroid,
        split_lon=True,
        split_lat=False,
        split_radial=False,
        stack=stack,
        stack_top=stack_top,
    )
    splitted = np.array([tess for tess in stack if not np.all(tess == 0)])
    assert splitted.shape[0] == 2
    assert splitted.shape[0] == stack_top + 1
    # Check if the tesseroid hasn't been split on latitudinal and radial direction
    assert (splitted[0, lat_indexes] == splitted[:, lat_indexes]).all()
    assert (splitted[0, radial_indexes] == splitted[:, radial_indexes]).all()
    # Check if the tesseroid has been correctly split on longitudinal direction
    lon_splitted = splitted[:, lon_indexes]
    lon_splitted.sort(axis=0)
    npt.assert_allclose(lon_splitted[0], [w, (e + w) / 2])
    npt.assert_allclose(lon_splitted[1], [(e + w) / 2, e])


def test_split_tesseroid_only_latitude():
    "Test splitting of a tesseroid only on latitude"
    lon_indexes = [0, 1]
    lat_indexes = [2, 3]
    radial_indexes = [4, 5]
    w, e, s, n, bottom, top = -10.0, 10.0, -10.0, 10.0, 1.0, 10.0
    tesseroid = [w, e, s, n, bottom, top]
    # Split only on longitude
    stack = np.zeros((8, 6))
    stack_top = -1
    stack_top = _split_tesseroid(
        tesseroid,
        split_lon=False,
        split_lat=True,
        split_radial=False,
        stack=stack,
        stack_top=stack_top,
    )
    splitted = np.array([tess for tess in stack if not np.all(tess == 0)])
    assert splitted.shape[0] == 2
    assert splitted.shape[0] == stack_top + 1
    # Check if the tesseroid hasn't been split on longitudinal and radial direction
    assert (splitted[0, lon_indexes] == splitted[:, lon_indexes]).all()
    assert (splitted[0, radial_indexes] == splitted[:, radial_indexes]).all()
    # Check if the tesseroid has been correctly split on longitudinal direction
    lat_splitted = splitted[:, lat_indexes]
    lat_splitted.sort(axis=0)
    npt.assert_allclose(lat_splitted[0], [s, (n + s) / 2])
    npt.assert_allclose(lat_splitted[1], [(n + s) / 2, n])


def test_split_tesseroid_only_radius():
    "Test splitting of a tesseroid only on radius"
    lon_indexes = [0, 1]
    lat_indexes = [2, 3]
    radial_indexes = [4, 5]
    w, e, s, n, bottom, top = -10.0, 10.0, -10.0, 10.0, 1.0, 10.0
    tesseroid = [w, e, s, n, bottom, top]
    # Split only on longitude
    stack = np.zeros((8, 6))
    stack_top = -1
    stack_top = _split_tesseroid(
        tesseroid,
        split_lon=False,
        split_lat=False,
        split_radial=True,
        stack=stack,
        stack_top=stack_top,
    )
    splitted = np.array([tess for tess in stack if not np.all(tess == 0)])
    assert splitted.shape[0] == 2
    assert splitted.shape[0] == stack_top + 1
    # Check if the tesseroid hasn't been split on longitudinal and latitudinal direction
    assert (splitted[0, lon_indexes] == splitted[:, lon_indexes]).all()
    assert (splitted[0, lat_indexes] == splitted[:, lat_indexes]).all()
    # Check if the tesseroid has been correctly split on longitudinal direction
    radial_splitted = splitted[:, radial_indexes]
    radial_splitted.sort(axis=0)
    npt.assert_allclose(radial_splitted[0], [bottom, (top + bottom) / 2])
    npt.assert_allclose(radial_splitted[1], [(top + bottom) / 2, top])


def test_split_tesseroid_only_horizontal():
    "Test splitting of a tesseroid on horizontal directions"
    radial_indexes = [4, 5]
    tesseroid = [-10.0, 10.0, -10.0, 10.0, 1.0, 10.0]
    # Split only on longitude
    stack = np.zeros((8, 6))
    stack_top = -1
    stack_top = _split_tesseroid(
        tesseroid,
        split_lon=True,
        split_lat=True,
        split_radial=False,
        stack=stack,
        stack_top=stack_top,
    )
    splitted = np.array([tess for tess in stack if not np.all(tess == 0)])
    assert splitted.shape[0] == 2 ** 2
    assert splitted.shape[0] == stack_top + 1
    # Check if the tesseroid hasn't been split on radial direction
    assert (splitted[0, radial_indexes] == splitted[:, radial_indexes]).all()


def test_split_tesseroid():
    "Test splitting of a tesseroid on every direction"
    lon_indexes = [0, 1]
    lat_indexes = [2, 3]
    radial_indexes = [4, 5]
    tesseroid = [-10.0, 10.0, -10.0, 10.0, 1.0, 10.0]
    # Split only on longitude
    stack = np.zeros((8, 6))
    stack_top = -1
    stack_top = _split_tesseroid(
        tesseroid,
        split_lon=True,
        split_lat=True,
        split_radial=True,
        stack=stack,
        stack_top=stack_top,
    )
    splitted = np.array([tess for tess in stack if not np.all(tess == 0)])
    assert splitted.shape[0] == 2 ** 3
    assert splitted.shape[0] == stack_top + 1
    # Check if the tesseroid hasn't been split on each direction
    assert not (splitted[0, lon_indexes] == splitted[:, lon_indexes]).all()
    assert not (splitted[0, lat_indexes] == splitted[:, lat_indexes]).all()
    assert not (splitted[0, radial_indexes] == splitted[:, radial_indexes]).all()


def test_adaptive_discretization_on_radii():
    "Test if closer computation points increase the tesseroid discretization"
    for radial_discretization in [True, False]:
        tesseroid = [-10.0, 10.0, -10.0, 10.0, 1.0, 10.0]
        radii = [10.5, 12.0, 13.0, 15.0, 20.0, 30.0]
        # Only if 2D adaptive discretization set point on the surface of the tesseroid
        if radial_discretization:
            radii.insert(0, 10.1)
        else:
            radii.insert(0, 10.0)
        number_of_splits = []
        for radius in radii:
            coordinates = [0.0, 0.0, radius]
            smaller_tesseroids = adaptive_discretization(
                coordinates,
                tesseroid,
                distance_size_ratio=10.0,
                radial_discretization=radial_discretization,
            )
            number_of_splits.append(smaller_tesseroids.shape[0])
        for i in range(1, len(number_of_splits)):
            assert number_of_splits[i - 1] >= number_of_splits[i]


def test_adaptive_discretization_on_distance_size_ratio():
    "Test if higher distance-size-ratio increase the tesseroid discretization"
    for radial_discretization in [True, False]:
        tesseroid = [-10.0, 10.0, -10.0, 10.0, 1.0, 10.0]
        coordinates = [0.0, 0.0, 10.2]
        distance_size_ratii = np.linspace(1, 10, 10)
        number_of_splits = []
        for distance_size_ratio in distance_size_ratii:
            smaller_tesseroids = adaptive_discretization(
                coordinates,
                tesseroid,
                distance_size_ratio=distance_size_ratio,
                radial_discretization=radial_discretization,
            )
            number_of_splits.append(smaller_tesseroids.shape[0])
        for i in range(1, len(number_of_splits)):
            assert number_of_splits[i - 1] <= number_of_splits[i]


def test_stack_overflow():
    "Test if adaptive discretization raises OverflowError on stack overflow"
    tesseroid = [-10.0, 10.0, -10.0, 10.0, 0.5, 1.0]
    coordinates = [0.0, 0.0, 1.0]
    with raises(OverflowError):
        adaptive_discretization(coordinates, tesseroid, 10.0, stack_size=2)


def test_two_dimensional_adaptive_discretization():
    "Test if the 2D adaptive discretization produces no splits on radial direction"
    bottom, top = 1.0, 10.0
    tesseroid = [-10.0, 10.0, -10.0, 10.0, bottom, top]
    coordinates = [0.0, 0.0, top]
    small_tesseroids = adaptive_discretization(coordinates, tesseroid, 10.0)
    for tess in small_tesseroids:
        assert tess[-2] == bottom
        assert tess[-1] == top
