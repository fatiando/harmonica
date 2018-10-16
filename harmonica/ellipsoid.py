"""
Module for defining and setting the reference ellipsoid for normal gravity calculations
and coordinate transformations.
"""
import attr


ELLIPSOID = None
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


@attr.s(frozen=True)
class Ellipsoid:
    """
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
        return 1/self.inverse_flattening

    @property
    def semiminor_axis(self):
        "The small (polar) axis of the ellipsoid [meters]"
        return self.semimajor_axis*(1 - self.flattening)


def set_ellipsoid(ellipsoid):
    if ellipsoid in KNOWN_ELLIPSOIDS:
        ellipsoid = KNOWN_ELLIPSOIDS[ellipsoid]
    return EllipsoidContext(ellipsoid)


def get_ellipsoid():
    return ELLIPSOID


class EllipsoidContext:
    def __init__(self, ellipsoid):
        global ELLIPSOID
        self.backup = ELLIPSOID
        ELLIPSOID = ellipsoid

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global ELLIPSOID
        ELLIPSOID = self.backup


set_ellipsoid("WGS84")
