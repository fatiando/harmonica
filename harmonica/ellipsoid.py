"""
Module for defining and setting the reference ellipsoid for normal gravity calculations
and coordinate transformations.
"""
import math

import attr


ELLIPSOID = []
KNOWN_ELLIPSOIDS = {
    "WGS84": dict(
        name="WGS84",
        long_name="World Geodetic System 1984",
        semimajor_axis=6378137,
        inverse_flattening=298.257223563,
        geocentric_grav_const=3986004.418e+8,
        angular_velocity=7292115e-11,
    ),
    "GRS80": dict(
        name="GRS80",
        long_name="Geodetic Reference System 1980",
        semimajor_axis=6378137,
        inverse_flattening=298.257222101,
        geocentric_grav_const=3986005.0e+8,
        angular_velocity=7292115e-11,
    ),
}


# Don't let ellipsoid parameters be changed to avoid messing up calculations
# accidentally.
@attr.s(frozen=True)
class Ellipsoid:
    """
    A reference ellipsoid for coordinate manipulations and normal gravity calculations.

    The ellipsoid is oblate and spins around it's minor axis. It is defined by four
    parameters and offers other derived quantities as read-only properties. In fact, all
    attributes of this class are read-only and cannot be changed after instantiation.

    All ellipsoid parameters are in SI units.

    Use :func:`harmonica.set_ellipsoid` to set the default ellipsoid for all
    calculations in harmonica and :func:`harmonica.get_ellipsoid` to retrieve the
    currently set ellipsoid.

    Parameters
    ----------
    name : str
        A short name for the ellipsoid, for example ``'WGS84'``.
    semimajor_axis : float
        The semi-major axis of the ellipsoid (equatorial radius), usually represented by
        "a" [meters].
    inverse_flattening : float
        The inverse of the flattening (1/f) [adimensional].
    geocentric_grav_const : float
        The geocentric gravitational constant (GM) [m^3 s^-2].
    angular_velocity : float
        The angular velocity of the rotating ellipsoid (omega) [rad s^-1].
    long_name : float or None
        A long name for the ellipsoid, for example ``"World Geodetic System 1984"``
        (optional).

    Examples
    --------

    We can create the WGS84 ellipsoid using the values given in
    [Hofmann-WellenhofMoritz2006]_. This class offers derived attributes that can be
    used for other purposes. Note that the ellipsoid gravity at the pole differs from
    [Hofmann-WellenhofMoritz2006]_ on the last digit. This is sufficiently small as to
    not be a cause for concern.

    >>> wgs84 = Ellipsoid(name="WGS84", long_name="World Geodetic System 1984",
    ...                   semimajor_axis=6378137, inverse_flattening=298.257223563,
    ...                   geocentric_grav_const=3986004.418e+8,
    ...                   angular_velocity=7292115e-11)
    >>> print(wgs84) # doctest: +ELLIPSIS
    Ellipsoid(name='WGS84', ... long_name='World Geodetic System 1984')
    >>> print("{:.4f}".format(wgs84.semiminor_axis))
    6356752.3142
    >>> print("{:.7f}".format(wgs84.flattening))
    0.0033528
    >>> print("{:.13e}".format(wgs84.linear_eccentricity))
    5.2185400842339e+05
    >>> print("{:.13e}".format(wgs84.second_eccentricity))
    8.2094437949696e-02
    >>> print("{:.14f}".format(wgs84.emm))
    0.00344978650684
    >>> print("{:.10f}".format(wgs84.gravity_equator))
    9.7803253359
    >>> print("{:.10f}".format(wgs84.gravity_pole))
    9.8321849379

    """

    name = attr.ib()
    semimajor_axis = attr.ib()
    inverse_flattening = attr.ib()
    geocentric_grav_const = attr.ib()
    angular_velocity = attr.ib()
    long_name = attr.ib(default=None)

    @property
    def flattening(self):
        "The flattening of the ellipsoid [adimensional]"
        return 1 / self.inverse_flattening

    @property
    def semiminor_axis(self):
        "The small (polar) axis of the ellipsoid [meters]"
        return self.semimajor_axis * (1 - self.flattening)

    @property
    def linear_eccentricity(self):
        "The linear eccentricity [meters]"
        return math.sqrt(self.semimajor_axis ** 2 - self.semiminor_axis ** 2)

    @property
    def second_eccentricity(self):
        "The second eccentricity [adimensional]"
        return self.linear_eccentricity / self.semiminor_axis

    @property
    def emm(self):
        r"Auxiliary quantity :math:`m = \omega^2 a^2 b / (GM)`"
        return (
            self.angular_velocity ** 2
            * self.semimajor_axis ** 2
            * self.semiminor_axis
            / self.geocentric_grav_const
        )

    @property
    def gravity_equator(self):
        "The norm of the gravity vector at the equator on the ellipsoid"
        ratio = self.semiminor_axis / self.linear_eccentricity
        arctan = math.atan2(self.linear_eccentricity, self.semiminor_axis)
        aux = (
            self.second_eccentricity
            * (3 * (1 + ratio ** 2) * (1 - ratio * arctan) - 1)
            / (3 * ((1 + 3 * ratio ** 2) * arctan - 3 * ratio))
        )
        axis_mul = self.semimajor_axis * self.semiminor_axis
        result = self.geocentric_grav_const * (1 - self.emm - self.emm * aux) / axis_mul
        return result

    @property
    def gravity_pole(self):
        "The norm of the gravity vector at the poles on the ellipsoid"
        ratio = self.semiminor_axis / self.linear_eccentricity
        arctan = math.atan2(self.linear_eccentricity, self.semiminor_axis)
        aux = (
            self.second_eccentricity
            * (3 * (1 + ratio ** 2) * (1 - ratio * arctan) - 1)
            / (1.5 * ((1 + 3 * ratio ** 2) * arctan - 3 * ratio))
        )
        result = (
            self.geocentric_grav_const * (1 + self.emm * aux) / self.semimajor_axis ** 2
        )
        return result


def set_ellipsoid(ellipsoid="WGS84"):
    """
    """
    if ellipsoid in KNOWN_ELLIPSOIDS:
        ellipsoid = Ellipsoid(**KNOWN_ELLIPSOIDS[ellipsoid])
    return EllipsoidContext(ellipsoid)


def get_ellipsoid():
    """
    """
    return ELLIPSOID[-1]


class EllipsoidContext:
    """
    """

    def __init__(self, ellipsoid):
        ELLIPSOID.append(ellipsoid)

    def __enter__(self):
        """
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        """
        ELLIPSOID.pop()


# Set the default ellipsoid for all calculations
set_ellipsoid()
