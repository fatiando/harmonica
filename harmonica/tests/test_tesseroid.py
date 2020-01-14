"""
Test forward modelling for tesseroids.
"""
import numpy as np
import numpy.testing as npt
import pytest
from verde import grid_coordinates
import boule

from .utils import require_numba
from ..constants import GRAVITATIONAL_CONST
from ..forward.tesseroid import (
    tesseroid_gravity,
    _check_tesseroids,
    _check_points_outside_tesseroids,
    _distance_tesseroid_point,
    _tesseroid_dimensions,
    _split_tesseroid,
    _adaptive_discretization,
    _longitude_continuity,
    STACK_SIZE,
    MAX_DISCRETIZATIONS,
)

# Define the accuracy threshold for tesseroids (0.1%) as a
# relative error (0.001)
ACCURACY_THRESHOLD = 1e-3


@pytest.mark.use_numba
def test_single_tesseroid():
    "Test single tesseroid for achieving coverage when Numba is disabled"
    ellipsoid = boule.WGS84
    top = ellipsoid.mean_radius
    bottom = top - 1e3
    tesseroid = np.array([-10.0, 10.0, -10.0, 10.0, bottom, top])
    density = 1000.0
    coordinates = [0.0, 0.0, top + 100]
    for field in ("potential", "g_z"):
        for radial_discretization in (True, False):
            tesseroid_gravity(
                coordinates,
                tesseroid,
                density,
                radial_adaptive_discretization=radial_discretization,
                field=field,
            )


# ------------------
# Test error raising
# ------------------
def test_invalid_field():
    "Check if passing an invalid field raises an error"
    tesseroid = [-10, 10, -10, 10, 100, 200]
    density = 1000
    coordinates = [0, 0, 250]
    with pytest.raises(ValueError):
        tesseroid_gravity(coordinates, tesseroid, density, field="Not a valid field")


def test_invalid_distance_size_ratii():
    """
    Check if distance_size_ratii argument is well handled by tesseroid_gravity
    """
    tesseroid = [-10, 10, -10, 10, 100, 200]
    density = 1000
    coordinates = [0, 0, 250]
    # Check empty distance_size_ratii dictionary
    distance_size_ratii = {}
    for field in ("potential", "g_z"):
        with pytest.raises(ValueError):
            tesseroid_gravity(
                coordinates,
                tesseroid,
                density,
                distance_size_ratii=distance_size_ratii,
                field=field,
            )


def test_invalid_density_array():
    """
    Check if error is raised when density shape does not match tesseroids shape
    """
    # Create a set of 4 tesseroids
    tesseroids = [
        [-10, 0, -10, 0, 100, 200],
        [-10, 0, 0, 10, 100, 200],
        [0, 10, -10, 0, 100, 200],
        [0, 10, 0, 10, 100, 200],
    ]
    # Generate a two element density
    density = [1000, 2000]
    coordinates = [0, 0, 250]
    with pytest.raises(ValueError):
        tesseroid_gravity(coordinates, tesseroids, density, field="potential")


def test_valid_tesseroid():
    "Check if no valid tesseroid is caught as invalid by _check_tesseroids"
    # Define some tesseroid boundaries
    w, e, s, n, bottom, top = -10, 10, -10, 10, 100, 200
    # Check if some valid tesseroids are not caught
    _check_tesseroids(np.atleast_2d([w, e, s, n, bottom, top]))
    # Check if valid tesseroids with zero volume aren't caught
    _check_tesseroids(np.atleast_2d([w, w, s, n, bottom, top]))
    _check_tesseroids(np.atleast_2d([w, e, s, s, bottom, top]))
    _check_tesseroids(np.atleast_2d([w, e, s, n, bottom, bottom]))
    # Check if valid tesseroid with west > east is not caught
    _check_tesseroids(np.atleast_2d([350, 10, s, n, bottom, top]))
    # Check if tesseroids on [-180, 180) are not caught
    _check_tesseroids(np.atleast_2d([-70, -60, s, n, bottom, top]))
    _check_tesseroids(np.atleast_2d([-10, 10, s, n, bottom, top]))
    _check_tesseroids(np.atleast_2d([-150, 150, s, n, bottom, top]))
    # Check if tesseroids around the globe aren't caught
    _check_tesseroids(np.atleast_2d([0, 360, s, n, bottom, top]))
    _check_tesseroids(np.atleast_2d([-180, 180, s, n, bottom, top]))


def test_invalid_tesseroid():
    "Check if invalid tesseroid boundaries are caught by _check_tesseroids"
    # Define some tesseroid boundaries
    w, e, s, n, bottom, top = -10, 10, -10, 10, 100, 200
    # Test invalid latitudinal boundaries
    with pytest.raises(ValueError):
        _check_tesseroids(np.atleast_2d([w, e, n, s, bottom, top]))
    with pytest.raises(ValueError):
        _check_tesseroids(np.atleast_2d([w, e, s, -100, bottom, top]))
    with pytest.raises(ValueError):
        _check_tesseroids(np.atleast_2d([w, e, s, 100, bottom, top]))
    with pytest.raises(ValueError):
        _check_tesseroids(np.atleast_2d([w, e, -100, n, bottom, top]))
    with pytest.raises(ValueError):
        _check_tesseroids(np.atleast_2d([w, e, 100, n, bottom, top]))
    # Test invalid radial boundaries
    with pytest.raises(ValueError):
        _check_tesseroids(np.atleast_2d([w, e, s, n, top, bottom]))
    with pytest.raises(ValueError):
        _check_tesseroids(np.atleast_2d([w, e, s, n, bottom, -1]))
    with pytest.raises(ValueError):
        _check_tesseroids(np.atleast_2d([w, e, s, n, -1, top]))
    # Test invalid longitudinal boundaries
    # longitudinal boundaries outside the [-180, 360] interval
    with pytest.raises(ValueError):
        _check_tesseroids(np.atleast_2d([-200, e, s, n, bottom, top]))
    with pytest.raises(ValueError):
        _check_tesseroids(np.atleast_2d([400, e, s, n, bottom, top]))
    with pytest.raises(ValueError):
        _check_tesseroids(np.atleast_2d([w, -200, s, n, bottom, top]))
    with pytest.raises(ValueError):
        _check_tesseroids(np.atleast_2d([w, 400, s, n, bottom, top]))
    # west > east (even after longitude continuity)
    with pytest.raises(ValueError):
        _check_tesseroids(np.atleast_2d([30, 0, s, n, bottom, top]))
    with pytest.raises(ValueError):
        _check_tesseroids(np.atleast_2d([-60, -70, s, n, bottom, top]))
    with pytest.raises(ValueError):
        _check_tesseroids(np.atleast_2d([300, -150, s, n, bottom, top]))
    with pytest.raises(ValueError):
        _check_tesseroids(np.atleast_2d([350, 340, s, n, bottom, top]))
    # more than one turn around the globe
    with pytest.raises(ValueError):
        _check_tesseroids(np.atleast_2d([-150, 300, s, n, bottom, top]))


@pytest.mark.use_numba
def test_disable_checks():
    "Check if the disable_checks flag works properly"
    valid_tesseroid = [0.0, 10.0, 0.0, 10.0, 10.0, 20.0]
    invalid_tesseroid = [0.0, 10.0, 0.0, 10.0, 20.0, 10.0]
    density = 100.0
    coordinates = [0.0, 0.0, 10.0]
    # By default, an error should be raised for invalid input
    with pytest.raises(ValueError):
        tesseroid_gravity(coordinates, invalid_tesseroid, density, field="potential")
    # Check if an invalid tesseroid doesn't raise an error with the
    # disable_checks flag set to True
    invalid_result = tesseroid_gravity(
        coordinates, invalid_tesseroid, density, field="potential", disable_checks=True
    )
    # Check if the invalid tesseroid generates a potential field inverse
    # than the one generated by the valid tesseroid
    valid_result = tesseroid_gravity(
        coordinates, valid_tesseroid, density, field="potential"
    )
    npt.assert_allclose(invalid_result, -valid_result)


def test_point_inside_tesseroid():
    "Check if a computation point inside the tesseroid is caught"
    tesseroids = np.atleast_2d([-10, 10, -10, 10, 100, 200])
    # Test if outside point is not caught
    points = [
        np.atleast_2d([0, 0, 250]).T,  # outside point on radius
        np.atleast_2d([20, 0, 150]).T,  # outside point on longitude
        np.atleast_2d([0, 20, 150]).T,  # outside point on latitude
        np.atleast_2d([0, 0, 200]).T,  # point on top surface
        np.atleast_2d([0, 0, 100]).T,  # point on bottom surface
        np.atleast_2d([-10, 0, 150]).T,  # point on western surface
        np.atleast_2d([10, 0, 150]).T,  # point on eastern surface
        np.atleast_2d([0, -10, 150]).T,  # point on southern surface
        np.atleast_2d([0, 10, 150]).T,  # point on northern surface
    ]
    for coordinates in points:
        _check_points_outside_tesseroids(coordinates, tesseroids)
    # Test if computation point is inside the tesseroid
    coordinates = np.atleast_2d([0, 0, 150]).T
    with pytest.raises(ValueError):
        _check_points_outside_tesseroids(coordinates, tesseroids)
    # Test if computation point with phased longitude is inside the tesseroid
    coordinates = np.atleast_2d([360, 0, 150]).T
    with pytest.raises(ValueError):
        _check_points_outside_tesseroids(coordinates, tesseroids)
    tesseroids = np.atleast_2d([260, 280, -10, 10, 100, 200])
    coordinates = np.atleast_2d([-90, 0, 150]).T
    with pytest.raises(ValueError):
        _check_points_outside_tesseroids(coordinates, tesseroids)


@pytest.mark.use_numba
def test_stack_overflow_on_adaptive_discretization():
    "Test if _adaptive_discretization raises OverflowError on stack overflow"
    tesseroid = np.array([-10.0, 10.0, -10.0, 10.0, 0.5, 1.0])
    coordinates = np.array([0.0, 0.0, 1.0])
    distance_size_ratio = 10
    # Test stack overflow
    stack = np.empty((2, 6))
    small_tesseroids = np.empty((MAX_DISCRETIZATIONS, 6))
    with pytest.raises(OverflowError):
        _adaptive_discretization(
            coordinates, tesseroid, distance_size_ratio, stack, small_tesseroids
        )
    # Test small_tesseroids overflow
    stack = np.empty((STACK_SIZE, 6))
    small_tesseroids = np.empty((2, 6))
    with pytest.raises(OverflowError):
        _adaptive_discretization(
            coordinates, tesseroid, distance_size_ratio, stack, small_tesseroids
        )


# --------------------------------------
# Test tesseroid distance and dimensions
# --------------------------------------
@pytest.mark.use_numba
def test_distance_tesseroid_point():
    "Test distance between tesseroid and computation point"
    ellipsoid = boule.WGS84
    longitude_p, latitude_p, radius_p = 0.0, 0.0, ellipsoid.mean_radius
    d_lon, d_lat, d_radius = 2.0, 2.0, 1.3
    tesseroid = np.array(
        [
            longitude_p - d_lon / 2,
            longitude_p + d_lon / 2,
            latitude_p - d_lat / 2,
            latitude_p + d_lat / 2,
            radius_p - d_radius / 2,
            radius_p + d_radius / 2,
        ]
    )
    # Computation point on the center of the tesseroid
    point = np.array([longitude_p, latitude_p, radius_p])
    npt.assert_allclose(_distance_tesseroid_point(point, tesseroid), 0.0)
    # Computation point and tesseroid center with same longitude and latitude
    radius = radius_p + 1.0
    point = np.array([longitude_p, latitude_p, radius])
    npt.assert_allclose(_distance_tesseroid_point(point, tesseroid), 1.0)
    # Computation point and tesseroid center on equator and same radius
    longitude = 3.0
    point = np.array([longitude, latitude_p, radius_p])
    distance = 2 * radius_p * np.sin(0.5 * np.radians(abs(longitude - longitude_p)))
    npt.assert_allclose(_distance_tesseroid_point(point, tesseroid), distance)
    # Computation point and tesseroid center on same meridian and radius
    latitude = 3.0
    point = np.array([longitude_p, latitude, radius_p])
    distance = 2 * radius_p * np.sin(0.5 * np.radians(abs(latitude - latitude_p)))
    npt.assert_allclose(_distance_tesseroid_point(point, tesseroid), distance)


@pytest.mark.use_numba
def test_tesseroid_dimensions():
    "Test calculation of tesseroid dimensions"
    # Tesseroid on equator
    w, e, s, n, bottom, top = -1.0, 1.0, -1.0, 1.0, 0.5, 1.5
    tesseroid = np.array([w, e, s, n, bottom, top])
    l_lon, l_lat = top * np.radians(abs(e - w)), top * np.radians(abs(n - s))
    l_rad = top - bottom
    npt.assert_allclose((l_lon, l_lat, l_rad), _tesseroid_dimensions(tesseroid))


# -------------------------
# Test longitude continuity
# -------------------------
def test_longitude_continuity():
    "Check if longitude_continuity works as expected"
    # Tesseroid on the [-180, 180) interval
    tesseroid = np.atleast_2d([-10, 10, -10, 10, 1, 2])
    tesseroid = _longitude_continuity(tesseroid)
    assert tesseroid[0, 0] == -10
    assert tesseroid[0, 1] == 10
    tesseroid = np.atleast_2d([-70, -60, -10, 10, 1, 2])
    tesseroid = _longitude_continuity(tesseroid)
    assert tesseroid[0, 0] == -70
    assert tesseroid[0, 1] == -60
    # Tesseroid on the [0, 360) interval
    tesseroid = np.atleast_2d([350, 10, -10, 10, 1, 2])
    tesseroid = _longitude_continuity(tesseroid)
    assert tesseroid[0, 0] == -10
    assert tesseroid[0, 1] == 10


def test_longitude_continuity_equivalent_tesseroids():
    "Check if two equivalent tesseroids generate the same gravity field"
    ellipsoid = boule.WGS84
    top = ellipsoid.mean_radius
    bottom = top - 1e4
    w, e, s, n = -10, 10, -10, 10
    tesseroid = [w, e, s, n, bottom, top]
    density = 1e3
    coordinates = [0, 0, ellipsoid.mean_radius + 1e3]
    for field in ("potential", "g_z"):
        result = tesseroid_gravity(coordinates, tesseroid, density, field=field)
        # Change longitudinal boundaries of tesseroid but defining the same one
        tesseroid = [350, 10, s, n, bottom, top]
        npt.assert_allclose(
            result, tesseroid_gravity(coordinates, tesseroid, density, field=field)
        )


# ---------------------
# Test tesseroid splits
# ---------------------
@pytest.mark.use_numba
def test_split_tesseroid():
    "Test splitting of a tesseroid on every direction"
    lon_indexes, lat_indexes, radial_indexes = [0, 1], [2, 3], [4, 5]
    tesseroid = np.array([-10.0, 10.0, -10.0, 10.0, 1.0, 10.0])
    # Split only on longitude
    stack = np.zeros((8, 6))
    stack_top = -1
    stack_top = _split_tesseroid(
        tesseroid, n_lon=2, n_lat=2, n_rad=2, stack=stack, stack_top=stack_top
    )
    splitted = np.array([tess for tess in stack if not np.all(tess == 0)])
    assert splitted.shape[0] == 2 ** 3
    assert splitted.shape[0] == stack_top + 1
    # Check if the tesseroid hasn't been split on each direction
    assert not (splitted[0, lon_indexes] == splitted[:, lon_indexes]).all()
    assert not (splitted[0, lat_indexes] == splitted[:, lat_indexes]).all()
    assert not (splitted[0, radial_indexes] == splitted[:, radial_indexes]).all()


@pytest.mark.use_numba
def test_split_tesseroid_only_longitude():
    "Test splitting of a tesseroid only on longitude"
    lon_indexes, lat_indexes, radial_indexes = [0, 1], [2, 3], [4, 5]
    w, e, s, n, bottom, top = -10.0, 10.0, -10.0, 10.0, 1.0, 10.0
    tesseroid = np.array([w, e, s, n, bottom, top])
    # Split only on longitude
    stack = np.zeros((8, 6))
    stack_top = -1
    stack_top = _split_tesseroid(
        tesseroid, n_lon=2, n_lat=1, n_rad=1, stack=stack, stack_top=stack_top
    )
    splitted = np.array([tess for tess in stack if not np.all(tess == 0)])
    assert splitted.shape[0] == 2
    assert splitted.shape[0] == stack_top + 1
    # Check if the tesseroid hasn't been split on latitudinal and
    # radial direction
    assert (splitted[0, lat_indexes] == splitted[:, lat_indexes]).all()
    assert (splitted[0, radial_indexes] == splitted[:, radial_indexes]).all()
    # Check if the tesseroid has been correctly split on longitudinal direction
    lon_splitted = splitted[:, lon_indexes]
    lon_splitted.sort(axis=0)
    npt.assert_allclose(lon_splitted[0], [w, (e + w) / 2])
    npt.assert_allclose(lon_splitted[1], [(e + w) / 2, e])


@pytest.mark.use_numba
def test_split_tesseroid_only_latitude():
    "Test splitting of a tesseroid only on latitude"
    lon_indexes, lat_indexes, radial_indexes = [0, 1], [2, 3], [4, 5]
    w, e, s, n, bottom, top = -10.0, 10.0, -10.0, 10.0, 1.0, 10.0
    tesseroid = np.array([w, e, s, n, bottom, top])
    # Split only on longitude
    stack = np.zeros((8, 6))
    stack_top = -1
    stack_top = _split_tesseroid(
        tesseroid, n_lon=1, n_lat=2, n_rad=1, stack=stack, stack_top=stack_top
    )
    splitted = np.array([tess for tess in stack if not np.all(tess == 0)])
    assert splitted.shape[0] == 2
    assert splitted.shape[0] == stack_top + 1
    # Check if the tesseroid hasn't been split on longitudinal and
    # radial direction
    assert (splitted[0, lon_indexes] == splitted[:, lon_indexes]).all()
    assert (splitted[0, radial_indexes] == splitted[:, radial_indexes]).all()
    # Check if the tesseroid has been correctly split on longitudinal direction
    lat_splitted = splitted[:, lat_indexes]
    lat_splitted.sort(axis=0)
    npt.assert_allclose(lat_splitted[0], [s, (n + s) / 2])
    npt.assert_allclose(lat_splitted[1], [(n + s) / 2, n])


@pytest.mark.use_numba
def test_split_tesseroid_only_radius():
    "Test splitting of a tesseroid only on radius"
    lon_indexes, lat_indexes, radial_indexes = [0, 1], [2, 3], [4, 5]
    w, e, s, n, bottom, top = -10.0, 10.0, -10.0, 10.0, 1.0, 10.0
    tesseroid = np.array([w, e, s, n, bottom, top])
    # Split only on longitude
    stack = np.zeros((8, 6))
    stack_top = -1
    stack_top = _split_tesseroid(
        tesseroid, n_lon=1, n_lat=1, n_rad=2, stack=stack, stack_top=stack_top
    )
    splitted = np.array([tess for tess in stack if not np.all(tess == 0)])
    assert splitted.shape[0] == 2
    assert splitted.shape[0] == stack_top + 1
    # Check if the tesseroid hasn't been split on longitudinal and latitudinal
    # direction
    assert (splitted[0, lon_indexes] == splitted[:, lon_indexes]).all()
    assert (splitted[0, lat_indexes] == splitted[:, lat_indexes]).all()
    # Check if the tesseroid has been correctly split on longitudinal direction
    radial_splitted = splitted[:, radial_indexes]
    radial_splitted.sort(axis=0)
    npt.assert_allclose(radial_splitted[0], [bottom, (top + bottom) / 2])
    npt.assert_allclose(radial_splitted[1], [(top + bottom) / 2, top])


@pytest.mark.use_numba
def test_split_tesseroid_only_horizontal():
    "Test splitting of a tesseroid on horizontal directions"
    radial_indexes = [4, 5]
    tesseroid = np.array([-10.0, 10.0, -10.0, 10.0, 1.0, 10.0])
    # Split only on longitude
    stack = np.zeros((8, 6))
    stack_top = -1
    stack_top = _split_tesseroid(
        tesseroid, n_lon=2, n_lat=2, n_rad=1, stack=stack, stack_top=stack_top
    )
    splitted = np.array([tess for tess in stack if not np.all(tess == 0)])
    assert splitted.shape[0] == 2 ** 2
    assert splitted.shape[0] == stack_top + 1
    # Check if the tesseroid hasn't been split on radial direction
    assert (splitted[0, radial_indexes] == splitted[:, radial_indexes]).all()


# ----------------------------
# Test adaptive discretization
# ----------------------------
@require_numba
def test_adaptive_discretization_on_radii():
    "Test if closer computation points increase the tesseroid discretization"
    tesseroid = np.array([-10.0, 10.0, -10.0, 10.0, 1.0, 10.0])
    distance_size_ratio = 10
    stack = np.empty((STACK_SIZE, 6))
    small_tesseroids = np.empty((MAX_DISCRETIZATIONS, 6))
    for radial_discretization in [True, False]:
        radii = [10.5, 12.0, 13.0, 15.0, 20.0, 30.0]
        # Only if 2D adaptive discretization set point on the surface of the
        # tesseroid
        if radial_discretization:
            radii.insert(0, 10.1)
        else:
            radii.insert(0, 10.0)
        number_of_splits = []
        for radius in radii:
            coordinates = np.array([0.0, 0.0, radius])
            n_splits = _adaptive_discretization(
                coordinates,
                tesseroid,
                distance_size_ratio,
                stack,
                small_tesseroids,
                radial_discretization=radial_discretization,
            )
            number_of_splits.append(n_splits)
        for i in range(1, len(number_of_splits)):
            assert number_of_splits[i - 1] >= number_of_splits[i]


@require_numba
def test_adaptive_discretization_vs_distance_size_ratio():
    "Test if higher distance-size-ratio increase the tesseroid discretization"
    tesseroid = np.array([-10.0, 10.0, -10.0, 10.0, 1.0, 10.0])
    coordinates = np.array([0.0, 0.0, 10.2])
    distance_size_ratii = np.linspace(1, 10, 10)
    stack = np.empty((STACK_SIZE, 6))
    small_tesseroids = np.empty((MAX_DISCRETIZATIONS, 6))
    for radial_discretization in [True, False]:
        number_of_splits = []
        for distance_size_ratio in distance_size_ratii:
            n_splits = _adaptive_discretization(
                coordinates,
                tesseroid,
                distance_size_ratio,
                stack,
                small_tesseroids,
                radial_discretization=radial_discretization,
            )
            number_of_splits.append(n_splits)
        for i in range(1, len(number_of_splits)):
            assert number_of_splits[i - 1] <= number_of_splits[i]


@require_numba
def test_two_dimensional_adaptive_discretization():
    """
    Test if 2D adaptive discretization produces no splits on radial direction
    """
    bottom, top = 1.0, 10.0
    tesseroid = np.array([-10.0, 10.0, -10.0, 10.0, bottom, top])
    coordinates = np.array([0.0, 0.0, top])
    stack = np.empty((STACK_SIZE, 6))
    small_tesseroids = np.empty((MAX_DISCRETIZATIONS, 6))
    distance_size_ratio = 10
    n_splits = _adaptive_discretization(
        coordinates, tesseroid, distance_size_ratio, stack, small_tesseroids
    )
    small_tesseroids = small_tesseroids[:n_splits]
    for tess in small_tesseroids:
        assert tess[-2] == bottom
        assert tess[-1] == top


# ------------------------------------------------------------------
# Compare numerical result vs analytical solution of spherical shell
# ------------------------------------------------------------------
def spherical_shell_analytical(top, bottom, density, radius):
    """
    Compute analytical solution of gravity fields for an
    homogeneous spherical shell
    """
    potential = (
        4
        / 3
        * np.pi
        * GRAVITATIONAL_CONST
        * density
        * (top ** 3 - bottom ** 3)
        / radius
    )
    analytical = {
        "potential": potential,
        # Accelerations are converted from SI to mGal
        "g_z": 1e5 * potential / radius,
    }
    return analytical


@require_numba
def test_spherical_shell_two_dim_adaptive_discret():  # pylint: disable=too-many-locals
    """
    Compare numerical result with analytical solution for
    2D adaptive discretization
    """
    # Define computation point located on the equator at the mean Earth radius
    ellipsoid = boule.WGS84
    radius = ellipsoid.mean_radius
    coordinates = grid_coordinates([0, 350, -90, 90], spacing=10, extra_coords=radius)
    # Define lon and lat coordinates of spherical shell model made of
    # tesseroids
    shape = (12, 6)
    longitude = np.linspace(0, 360, shape[0] + 1)
    latitude = np.linspace(-90, 90, shape[1] + 1)
    west, east = longitude[:-1], longitude[1:]
    south, north = latitude[:-1], latitude[1:]
    # Define a density for the shell
    density = 1000
    # Define different values for the spherical shell thickness
    thicknesses = np.logspace(1, 5, 5)
    for thickness in thicknesses:
        # Create list of tesseroids for the spherical shell model
        tesseroids = []
        # Define boundary coordinates of each tesseroid
        top = ellipsoid.mean_radius
        bottom = top - thickness
        for w, e in zip(west, east):
            for s, n in zip(south, north):
                tesseroids.append([w, e, s, n, bottom, top])
        # Get analytical solutions
        analytical = spherical_shell_analytical(top, bottom, density, radius)
        # Assert analytical and numerical solution are bellow the accuracy
        # threshold
        for field in analytical:
            npt.assert_allclose(
                analytical[field],
                tesseroid_gravity(
                    coordinates, tesseroids, density * np.ones(shape), field=field
                ),
                rtol=ACCURACY_THRESHOLD,
            )


@require_numba
def test_spherical_shell_three_dim_adaptive_discret():  # pylint: disable=too-many-locals
    """
    Compare numerical result with analytical solution for
    3D adaptive discretization
    """
    # Define computation point located on the equator at 1km above mean Earth
    # radius
    ellipsoid = boule.WGS84
    radius = ellipsoid.mean_radius + 1e3
    coordinates = [0, 0, radius]
    # Define shape of spherical shell model made of tesseroids
    shape = (6, 6)
    # Define a density for the shell
    density = 1000
    # Define different values for the spherical shell thickness
    thicknesses = np.logspace(1, 5, 5)
    for thickness in thicknesses:
        # Create list of tesseroids for the spherical shell model
        tesseroids = []
        # Define boundary coordinates of each tesseroid
        top = ellipsoid.mean_radius
        bottom = top - thickness
        longitude = np.linspace(0, 360, shape[0] + 1)
        latitude = np.linspace(-90, 90, shape[1] + 1)
        west, east = longitude[:-1], longitude[1:]
        south, north = latitude[:-1], latitude[1:]
        for w, e in zip(west, east):
            for s, n in zip(south, north):
                tesseroids.append([w, e, s, n, bottom, top])
        # Get analytical solutions
        analytical = spherical_shell_analytical(top, bottom, density, radius)
        # Assert analytical and numerical solution are bellow the accuracy
        # threshold
        for field in analytical:
            npt.assert_allclose(
                analytical[field],
                tesseroid_gravity(
                    coordinates, tesseroids, density * np.ones(shape), field=field
                ),
                rtol=ACCURACY_THRESHOLD,
            )
