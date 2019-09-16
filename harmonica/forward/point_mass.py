"""
Forward modelling for point masses
"""
import numpy as np
from numba import jit

from ..constants import GRAVITATIONAL_CONST
from .utils import (
    check_coordinate_system,
    distance_cartesian,
    distance_spherical,
    distance_spherical_core,
)


def point_mass_gravity(
    coordinates, points, masses, field, coordinate_system="cartesian", dtype="float64"
):
    r"""
    Compute gravitational fields of point masses.

    It can compute the gravitational fields of point masses on a set of computation
    points defined either in Cartesian or geocentric spherical coordinates.

    The potential gravity field generated by a point mass with mass :math:`m` located at
    a point :math:`Q` on a computation point :math:`P` can be computed as:

    .. math::

        V(P) = \frac{G m}{l},

    where :math:`G` is the gravitational constant and :math:`l` is the Euclidean
    distance between :math:`P` and :math:`Q` [Blakely1995]_.

    In Cartesian coordinates, the points :math:`P` and :math:`Q` are given by :math:`x`,
    :math:`y` and :math:`z` coordinates, which can be translated into ``northing``,
    ``easting`` and ``upward``, respectively.
    If :math:`P` is located at :math:`(x, y, z)`, and :math:`Q` at :math:`(x_p, y_p,
    z_p)`, the distance :math:`l` can be computed as:

    .. math::

        l = \sqrt{ (x - x_p)^2 + (y - y_p)^2 + (z - z_p)^2 }.

    The gradient of the potential, also known as the gravity acceleration vector
    :math:`\vec{g}`, is defined as:

    .. math::

        \vec{g} = \nabla V.

    Therefore, the :math:`z` component of :math:`\vec{g}` at the point :math:`P` can be
    computed as (remember that :math:`z` points upwards):

    .. math::

        g_{up}(P) = - \frac{G m}{l^3} (z - z_p).

    We define the downward component of the gravity acceleration as the opposite of
    :math:`g_{up}`:

    .. math::

        g_{z}(P) = \frac{G m}{l^3} (z - z_p).

    On a geocentric spherical coordinate system, the points :math:`P` and :math:`Q` are
    given by the ``longitude``, ``latitude`` and ``radius`` coordinates, i.e.
    :math:`\lambda`, :math:`\varphi` and :math:`r`, respectively. On this coordinate
    system, the Euclidean distance between :math:`P(r, \varphi, \lambda)` and
    :math:`Q(r_p, \varphi_p, \lambda_p)` can be calculated  as follows [Grombein2013]_:

    .. math::

        l = \sqrt{ r^2 + r_p^2 - 2 r r_p \cos \Psi },

    where

    .. math::

        \cos \Psi = \sin \varphi \sin \varphi_p +
        \cos \varphi \cos \varphi_p \cos(\lambda - \lambda_p).

    The radial component of the acceleration vector on a local North-oriented
    system whose origin is located on the point :math:`P(r, \varphi, \lambda)`
    is given by [Grombein2013]_:

    .. math::

        g_r(P) =  - \frac{G m}{l^3} (r - r_p \cos \Psi).

    We define the downward component of the gravity acceleration :math:`g_z` as the
    opposite of the radial component:

    .. math::

        g_z(P) = \frac{G m}{l^3} (r - r_p \cos \Psi).

    .. warning::

        When working in Cartesian coordinates, the **z direction points upwards**, i.e.
        positive and negative values of ``upward`` represent points above and below the
        surface, respectively. But remember that the ``g_z`` field returns the downward
        component of the gravity acceleration.

    .. warning::

        When working in geocentric spherical coordinates, remember that the ``g_z``
        field returns the downward component of the gravity acceleration on the local
        North oriented coordinate system. It is equivalent to the opposite of the radial
        component, therefore it's positive if the acceleration vector points inside the
        spheroid.


    Parameters
    ----------
    coordinates : list or array
        List or array containing the coordinates of computation points in the following
        order: ``easting``, ``northing`` and ``upward`` (if coordinates given in
        Cartesian coordiantes), or ``longitude``, ``latitude`` and ``radius`` (if given
        on a spherical geocentric coordinate system).
        All ``easting``, ``northing`` and ``upward`` should be in meters.
        Both ``longitude`` and ``latitude`` should be in degrees and ``radius`` in
        meters.
    points : list or array
        List or array containing the coordinates of the point masses in the following
        order: ``easting``, ``northing`` and ``upward`` (if coordinates given in
        Cartesian coordiantes), or ``longitude``, ``latitude`` and ``radius`` (if given
        on a spherical geocentric coordinate system).
        All ``easting``, ``northing`` and ``upward`` should be in meters.
        Both ``longitude`` and ``latitude`` should be in degrees and ``radius`` in
        meters.
    masses : list or array
        List or array containing the mass of each point mass in kg.
    field : str
        Gravitational field that wants to be computed.
        The available fields coordinates are:

        - Gravitational potential: ``potential``
        - Downward acceleration: ``g_z``

    coordinate_system : str (optional)
        Coordinate system of the coordinates of the computation points and the point
        masses. Available coordinates systems: ``cartesian``, ``spherical``.
        Default ``cartesian``.
    dtype : data-type (optional)
        Data type assigned to resulting gravitational field, and coordinates of point
        masses and computation points. Default to ``np.float64``.


    Returns
    -------
    result : array
        Gravitational field generated by the ``point_mass`` on the computation points
        defined in ``coordinates``.
        The potential is given in SI units, the accelerations in mGal and the Marussi
        tensor components in Eotvos.
    """
    # Organize dispatchers and kernel functions inside dictionaries
    dispatchers = {
        "cartesian": jit_point_mass_cartesian,
        "spherical": jit_point_mass_spherical,
    }
    kernels = {
        "cartesian": {
            "potential": kernel_potential_cartesian,
            "g_z": kernel_g_z_cartesian,
        },
        "spherical": {
            "potential": kernel_potential_spherical,
            "g_z": kernel_g_z_spherical,
        },
    }
    # Sanity checks for coordinate_system and field
    check_coordinate_system(coordinate_system)
    if field not in kernels[coordinate_system]:
        raise ValueError("Gravity field {} not recognized".format(field))
    # Figure out the shape and size of the output array
    cast = np.broadcast(*coordinates[:3])
    result = np.zeros(cast.size, dtype=dtype)
    # Prepare arrays to be passed to the jitted functions
    coordinates = tuple(np.atleast_1d(i).ravel().astype(dtype) for i in coordinates[:3])
    points = tuple(np.atleast_1d(i).ravel().astype(dtype) for i in points[:3])
    masses = np.atleast_1d(masses).astype(dtype).ravel()
    # Sanity checks
    if masses.size != points[0].size:
        raise ValueError(
            "Masses array must have the same size as number of point masses."
        )
    # Compute gravitational field
    dispatchers[coordinate_system](
        *coordinates, *points, masses, result, kernels[coordinate_system][field]
    )
    result *= GRAVITATIONAL_CONST
    # Convert to more convenient units
    if field == "g_z":
        result *= 1e5  # SI to mGal
    return result.reshape(cast.shape)


@jit(nopython=True)
def jit_point_mass_cartesian(
    easting, northing, upward, easting_p, northing_p, upward_p, masses, out, kernel
):  # pylint: disable=invalid-name
    """
    Compute gravity field of point masses on computation points in Cartesian coordinates

    Parameters
    ----------
    easting, northing, upward : 1d-arrays
        Coordinates of computation points in Cartesian coordinate system.
    easting_p, northing_p, upward_p : 1d-arrays
        Coordinates of point masses in Cartesian coordinate system.
    masses : 1d-array
        Mass of each point mass in SI units.
    out : 1d-array
        Array where the gravitational field on each computation point will be appended.
        It must have the same size of ``easting``, ``northing`` and ``upward``.
    kernel : func
        Kernel function that will be used to compute the gravity field on the
        computation points.
    """
    for l in range(easting.size):
        for m in range(easting_p.size):
            out[l] += masses[m] * kernel(
                easting[l],
                northing[l],
                upward[l],
                easting_p[m],
                northing_p[m],
                upward_p[m],
            )


@jit(nopython=True)
def kernel_potential_cartesian(
    easting, northing, upward, easting_p, northing_p, upward_p
):
    """
    Kernel function for potential gravity field in Cartesian coordinates
    """
    return 1 / distance_spherical(
        (easting, northing, upward), (easting_p, northing_p, upward_p)
    )


@jit(nopython=True)
def kernel_g_z_cartesian(easting, northing, upward, easting_p, northing_p, upward_p):
    """
    Kernel function for downward component of gravity gradient in Cartesian coordinates
    """
    distance = distance_cartesian(
        [easting, northing, upward], [easting_p, northing_p, upward_p]
    )
    return (upward - upward_p) / distance ** 3


@jit(nopython=True)
def jit_point_mass_spherical(
    longitude, latitude, radius, longitude_p, latitude_p, radius_p, masses, out, kernel
):  # pylint: disable=invalid-name
    """
    Compute gravity field of point masses on computation points in spherical coordinates

    Parameters
    ----------
    longitude, latitude, radius : 1d-arrays
        Coordinates of computation points in spherical geocentric coordinate system.
    longitude_p, latitude_p, radius_p : 1d-arrays
        Coordinates of point masses in spherical geocentric coordinate system.
    masses : 1d-array
        Mass of each point mass in SI units.
    out : 1d-array
        Array where the gravitational field on each computation point will be appended.
        It must have the same size of ``longitude``, ``latitude`` and ``radius``.
    kernel : func
        Kernel function that will be used to compute the gravity field on the
        computation points.
    """
    # Compute quantities related to computation point
    longitude = np.radians(longitude)
    latitude = np.radians(latitude)
    cosphi = np.cos(latitude)
    sinphi = np.sin(latitude)
    # Compute quantities related to point masses
    longitude_p = np.radians(longitude_p)
    latitude_p = np.radians(latitude_p)
    cosphi_p = np.cos(latitude_p)
    sinphi_p = np.sin(latitude_p)
    # Compute gravity field
    for l in range(longitude.size):
        for m in range(longitude_p.size):
            out[l] += masses[m] * kernel(
                longitude[l],
                cosphi[l],
                sinphi[l],
                radius[l],
                longitude_p[m],
                cosphi_p[m],
                sinphi_p[m],
                radius_p[m],
            )


@jit(nopython=True)
def kernel_potential_spherical(
    longitude, cosphi, sinphi, radius, longitude_p, cosphi_p, sinphi_p, radius_p
):
    """
    Kernel function for potential gravity field in spherical coordinates
    """
    distance, _, _ = distance_spherical_core(
        longitude, cosphi, sinphi, radius, longitude_p, cosphi_p, sinphi_p, radius_p
    )
    return 1 / distance


@jit(nopython=True)
def kernel_g_z_spherical(
    longitude, cosphi, sinphi, radius, longitude_p, cosphi_p, sinphi_p, radius_p
):
    """
    Kernel function for downward component of gravity gradient in spherical coordinates
    """
    distance, cospsi, _ = distance_spherical_core(
        longitude, cosphi, sinphi, radius, longitude_p, cosphi_p, sinphi_p, radius_p
    )
    delta_z = radius - radius_p * cospsi
    return delta_z / distance ** 3
