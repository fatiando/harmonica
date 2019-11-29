"""
Test forward modelling for point masses.
"""
import numpy as np
import numpy.testing as npt
import pytest

from ..constants import GRAVITATIONAL_CONST
from ..forward.point_mass import point_mass_gravity
from ..forward.utils import distance_cartesian


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
            931.10,
            -91.03,
            -849.97,
            365.09,
            -276.50,
            232.36,
            312.70,
            -134.88,
            409.50,
            -346.05,
            557.02,
            -932.30,
            -300.25,
            -110.90,
            -583.77,
            -546.05,
            -95.70,
            -734.09,
            290.47,
            993.40,
            684.20,
            437.78,
            -677.84,
            -304.01,
            -807.34,
            -376.36,
            900.98,
            -955.56,
            576.78,
            223.98,
            -13.06,
            830.86,
            339.04,
            -285.78,
            366.23,
            174.36,
            48.05,
            820.27,
            -108.76,
            -38.23,
            -484.50,
            312.16,
            -380.09,
            -334.75,
            77.81,
            902.84,
            847.66,
            450.64,
            791.34,
            622.71,
        ]
    )
    easting = np.array(
        [
            -131.70,
            -576.22,
            190.04,
            143.74,
            509.58,
            -166.58,
            -102.59,
            465.18,
            -417.79,
            950.07,
            -224.93,
            114.93,
            -968.02,
            -591.11,
            855.88,
            -844.75,
            940.84,
            69.15,
            669.43,
            867.80,
            -316.07,
            -630.61,
            109.24,
            777.67,
            -213.65,
            116.96,
            304.36,
            -902.88,
            -989.56,
            629.47,
            -434.78,
            35.17,
            109.05,
            414.24,
            -517.95,
            491.98,
            3.43,
            764.76,
            -546.27,
            -324.06,
            -512.52,
            26.20,
            709.40,
            119.21,
            -49.98,
            -645.65,
            497.76,
            599.79,
            854.61,
            82.29,
        ]
    )
    upward = np.array(
        [
            -893.20,
            -786.09,
            520.99,
            -968.22,
            -242.84,
            834.79,
            584.30,
            577.25,
            44.58,
            -231.41,
            -749.64,
            482.39,
            979.16,
            -681.41,
            975.01,
            371.44,
            579.73,
            490.48,
            206.78,
            19.65,
            -790.08,
            654.36,
            924.03,
            -615.07,
            962.74,
            -22.40,
            -136.43,
            -690.57,
            -303.72,
            693.82,
            940.05,
            137.74,
            478.93,
            -732.37,
            -439.04,
            337.17,
            544.70,
            -994.25,
            681.77,
            317.31,
            261.09,
            285.83,
            -892.43,
            -543.00,
            -532.01,
            -800.70,
            510.58,
            577.19,
            519.46,
            868.59,
        ]
    )
    coordinates = [easting, northing, upward]
    reference_values = GRAVITATIONAL_CONST * np.array(
        [
            3.855176919424,
            5.107752104793,
            4.926658868677,
            4.786054709408,
            7.954613392000,
            5.666420314717,
            7.455942163459,
            6.635464230025,
            8.522128914471,
            4.820334195373,
            5.204875535728,
            4.734941247442,
            3.548003019775,
            5.501387595948,
            3.514542301199,
            4.663049497953,
            4.507569002018,
            5.646058534931,
            6.592272557173,
            3.790156391155,
            4.579153404808,
            4.956798165253,
            4.343342386884,
            4.821298569285,
            3.923152681360,
            12.666216468561,
            5.204344878310,
            3.367042200379,
            4.219521966453,
            5.190941850742,
            4.827149429419,
            5.931663138166,
            8.377501825344,
            5.626754456652,
            6.481205423461,
            8.046436783272,
            9.143676674289,
            3.336126623489,
            5.679435198661,
            10.985375376975,
            6.648460108563,
            11.790711842140,
            4.160678338057,
            7.704923214409,
            9.259465081419,
            3.653330841953,
            4.513884876958,
            5.282294418322,
            3.920614966133,
            4.664576954878,
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
    Test if potential field of a point mass has symmetry in Cartesian coords
    """
    # Define a single point mass
    point_mass = [1.1, 1.2, 1.3]
    masses = [2670]
    # Define a set of computation points at a fixed distance from the point
    # mass
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


@pytest.mark.use_numba
def test_g_z_relative_error():
    """
    Test the relative error in computing the g_z component
    """
    # Define a single point mass
    point_mass = (1, -67, -300.7)
    mass = 250
    coordinates_p = (0, -39, -13)
    # Compute the z component
    exact_deriv = point_mass_gravity(
        coordinates_p, point_mass, mass, "g_z", "cartesian"
    )
    # Compute the numerical derivative of potential
    delta = 0.1
    easting = np.zeros(2) + coordinates_p[0]
    northing = np.zeros(2) + coordinates_p[1]
    upward = np.array([coordinates_p[2] - delta, coordinates_p[2] + delta])
    coordinates = (easting, northing, upward)
    potential = point_mass_gravity(
        coordinates, point_mass, mass, "potential", "cartesian"
    )
    # Remember that the ``g_z`` field returns the downward component of the
    # gravitational acceleration. As a consequence, the numerical
    # derivativative is multiplied by -1.
    approximated_deriv = -1e5 * (potential[1] - potential[0]) / (2.0 * delta)

    # Compute the relative error
    relative_error = np.abs((approximated_deriv - exact_deriv) / exact_deriv)

    # Bound value
    distance = distance_cartesian(coordinates_p, point_mass)
    bound_value = 1.5 * (delta / distance) ** 2

    # Compare the results
    npt.assert_array_less(relative_error, bound_value)


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
    Test if g_northing field of a point mass has symmetry in Cartesian
    coordinates
    """
    # Define a single point mass
    point_mass = [-7.9, 25, -130]
    masses = [2670]
    # Define a pair of computation points northward and southward the point
    # mass
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
def test_g_northing_relative_error():
    """
    Test the relative error in computing the g_northing component
    """
    # Define a single point mass
    point_mass = (1, -67, -300.7)
    mass = 250
    coordinates_p = (0, -39, -13)
    # Compute the northing component
    exact_deriv = point_mass_gravity(
        coordinates_p, point_mass, mass, "g_northing", "cartesian"
    )
    # Compute the numerical derivative of potential
    delta = 0.1
    easting = np.zeros(2) + coordinates_p[0]
    northing = np.array([coordinates_p[1] - delta, coordinates_p[1] + delta])
    upward = np.zeros(2) + coordinates_p[2]
    coordinates = (easting, northing, upward)
    potential = point_mass_gravity(
        coordinates, point_mass, mass, "potential", "cartesian"
    )
    approximated_deriv = 1e5 * (potential[1] - potential[0]) / (2.0 * delta)

    # Compute the relative error
    relative_error = np.abs((approximated_deriv - exact_deriv) / exact_deriv)

    # Bound value
    distance = distance_cartesian(coordinates_p, point_mass)
    bound_value = 1.5 * (delta / distance) ** 2

    # Compare the results
    npt.assert_array_less(relative_error, bound_value)


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
    Test if g_easting field of a point mass has symmetry in Cartesian
    coordinates
    """
    # Define a single point mass
    point_mass = [191, -5, 0]
    masses = [2670]
    # Define a pair of computation points northward and southward the point
    # mass
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
def test_g_easting_relative_error():
    """
    Test the relative error in computing the g_easting component
    """
    # Define a single point mass
    point_mass = (20, 54, -500.7)
    mass = 200
    coordinates_p = (-3, 24, -10)
    # Compute the easting component
    exact_deriv = point_mass_gravity(
        coordinates_p, point_mass, mass, "g_easting", "cartesian"
    )
    # Compute the numerical derivative of potential
    delta = 0.1
    easting = np.array([coordinates_p[0] - delta, coordinates_p[0] + delta])
    northing = np.zeros(2) + coordinates_p[1]
    upward = np.zeros(2) + coordinates_p[2]
    coordinates = (easting, northing, upward)
    potential = point_mass_gravity(
        coordinates, point_mass, mass, "potential", "cartesian"
    )
    approximated_deriv = 1e5 * (potential[1] - potential[0]) / (2.0 * delta)

    # Compute the relative error
    relative_error = np.abs((approximated_deriv - exact_deriv) / exact_deriv)

    # Bound value
    distance = distance_cartesian(coordinates_p, point_mass)
    bound_value = 1.5 * (delta / distance) ** 2

    # Compare the results
    npt.assert_array_less(relative_error, bound_value)


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
    # Analytical solutions (accelerations are in mgal and tensor components in
    # eotvos)
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
    """
    Check potential and g_z of point mass and computation point on same radius
    """
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
