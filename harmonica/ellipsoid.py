"""
Module for defining and setting the reference ellipsoid for normal gravity calculations
and coordinate transformations.
"""
import math

import attr


# Use a list to hold the current ellipsoid because it is mutable and therefore doesn't
# need the "global" keyword. The current ellipsoid will always be the last element of
# this list. See get_ellipsoid(). Starts off empty because we have to define
# set_ellipsoid, etc, first. See the last line of this file for the set_ellipsoid call.
ELLIPSOID = []
# Dict with the known ellipsoids and their names. Used when setting the current
# ellipsoid by name in set_ellipsoid().
KNOWN_ELLIPSOIDS = {
    # From Hofmann-WellenhofMoritz2006
    "WGS84": dict(
        name="WGS84",
        long_name="World Geodetic System 1984",
        semimajor_axis=6378137,
        inverse_flattening=298.257223563,
        geocentric_grav_const=3986004.418e8,
        angular_velocity=7292115e-11,
    ),
    # From Hofmann-WellenhofMoritz2006
    "GRS80": dict(
        name="GRS80",
        long_name="Geodetic Reference System 1980",
        semimajor_axis=6378137,
        inverse_flattening=298.257222101,
        geocentric_grav_const=3986005.0e8,
        angular_velocity=7292115e-11,
    ),
}


# Don't let ellipsoid parameters be changed to avoid messing up calculations
# accidentally.
@attr.s(frozen=True)
class ReferenceEllipsoid:
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

    >>> wgs84 = ReferenceEllipsoid(
    ...     name="WGS84",
    ...     long_name="World Geodetic System 1984",
    ...     semimajor_axis=6378137,
    ...     inverse_flattening=298.257223563,
    ...     geocentric_grav_const=3986004.418e+8,
    ...     angular_velocity=7292115e-11
    ... )
    >>> print(wgs84) # doctest: +ELLIPSIS
    ReferenceEllipsoid(name='WGS84', ... long_name='World Geodetic System 1984')
    >>> print("{:.4f}".format(wgs84.semiminor_axis))
    6356752.3142
    >>> print("{:.7f}".format(wgs84.flattening))
    0.0033528
    >>> print("{:.13e}".format(wgs84.linear_eccentricity))
    5.2185400842339e+05
    >>> print("{:.13e}".format(wgs84.first_eccentricity))
    8.1819190842621e-02
    >>> print("{:.13e}".format(wgs84.second_eccentricity))
    8.2094437949696e-02
    >>> print("{:.4f}".format(wgs84.mean_radius))
    6371008.7714
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
    def first_eccentricity(self):
        "The first eccentricity [adimensional]"
        return self.linear_eccentricity / self.semimajor_axis

    @property
    def second_eccentricity(self):
        "The second eccentricity [adimensional]"
        return self.linear_eccentricity / self.semiminor_axis

    @property
    def mean_radius(self):
        """
        The arithmetic mean radius :math:`R_1 = (2a + b) /3` [Moritz2000]_ [meters]
        """
        return 1 / 3 * (2 * self.semimajor_axis + self.semiminor_axis)

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
        "The norm of the gravity vector at the equator on the ellipsoid [m/s^2]"
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
        "The norm of the gravity vector at the poles on the ellipsoid [m/s^2]"
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


def print_ellipsoids(**kwargs):
    """
    Print all available ellipsoids.

    These are the ones that are hard-coded into the library. You can always set your own
    ellipsoid using :func:`harmonica.set_ellipsoid` and
    :class:`harmonica.ReferenceEllipsoid`.

    Any keyword arguments given to this function will be passed to :func:`print`.

    Examples
    --------

    >>> print_ellipsoids() # doctest: +ELLIPSIS
    ReferenceEllipsoid(name='GRS80', ... long_name='Geodetic Reference System 1980')
    ReferenceEllipsoid(name='WGS84', ... long_name='World Geodetic System 1984')

    """
    for ellipsoid in sorted(KNOWN_ELLIPSOIDS):
        print(ReferenceEllipsoid(**KNOWN_ELLIPSOIDS[ellipsoid]), **kwargs)


def set_ellipsoid(ellipsoid="WGS84"):
    """
    Set the reference ellipsoid used throughout the library.

    See :func:`harmonica.print_ellipsoids` for a list of all available ellipsoids. You
    always set your own ellipsoid by passing a :class:`harmonica.ReferenceEllipsoid`
    to this function instead.

    If used in a ``with`` block, will only set the ellipsoid for the context of the
    block. Upon exit, will restore the previously used ellipsoid.

    Parameters
    ----------
    ellipsoid : str or :class:`~harmonica.ReferenceEllipsoid`
        The reference ellipsoid to use throughout the library. Either the name of a
        built-in ellipsoid (see :func:`harmonica.print_ellipsoids`) or a
        :class:`~harmonica.ReferenceEllipsoid`.

    See also
    --------

    get_ellipsoid
    print_ellipsoids
    ReferenceEllipsoid

    Examples
    --------

    Set the ellipsoid globally (for an entire script or notebook):

    >>> # WGS84 is the default
    >>> print(get_ellipsoid().name)
    WGS84
    >>> # Set the ellipsoid globally
    >>> set_ellipsoid("GRS80") # doctest: +ELLIPSIS
    EllipsoidManager(...)
    >>> print(get_ellipsoid().name)
    GRS80
    >>> # Return to the default
    >>> set_ellipsoid() # doctest: +ELLIPSIS
    EllipsoidManager(...)
    >>> print(get_ellipsoid().name)
    WGS84

    To set the ellipsoid in a limited context, use a ``with`` block:

    >>> with set_ellipsoid("GRS80"):
    ...     print(get_ellipsoid().name)
    GRS80
    >>> print(get_ellipsoid().name)
    WGS84
    >>> # Blocks can be nested
    >>> with set_ellipsoid("GRS80"):
    ...     with set_ellipsoid("WGS84"):
    ...         print("Inner: ", get_ellipsoid().name)
    ...     print("Middle:", get_ellipsoid().name)
    Inner:  WGS84
    Middle: GRS80
    >>> print("Outer: ", get_ellipsoid().name)
    Outer:  WGS84

    Use :class:`harmonica.ReferenceEllipsoid` to set a custom ellipsoid:

    >>> myell = ReferenceEllipsoid(name="TINY", semimajor_axis=1, inverse_flattening=1,
    ...                            geocentric_grav_const=10, angular_velocity=1)
    >>> with set_ellipsoid(myell):
    ...     print(get_ellipsoid().name)
    TINY
    >>> print(get_ellipsoid().name)
    WGS84

    """
    if ellipsoid in KNOWN_ELLIPSOIDS:
        ellipsoid = ReferenceEllipsoid(**KNOWN_ELLIPSOIDS[ellipsoid])
    return EllipsoidManager(ellipsoid).set()


def get_ellipsoid():
    """
    Get the current reference ellipsoid.

    Returns
    -------
    ellipsoid : :class:`~harmonica.ReferenceEllipsoid`
        The currently set reference ellipsoid.

    See also
    --------
    set_ellipsoid
    print_ellipsoids
    ReferenceEllipsoid

    Examples
    --------

    >>> ell = get_ellipsoid()
    >>> print(ell) # doctest: +ELLIPSIS
    ReferenceEllipsoid(name='WGS84', ...)

    """
    return ELLIPSOID[-1]


@attr.s
class EllipsoidManager:
    """
    A context manager to handle setting and resetting the current ellipsoid.
    """

    ellipsoid = attr.ib()
    # Allow passing in the ELLIPSOID in the constructor so we can use another list to
    # test this class without messing with the global ellipsoid.
    global_context = attr.ib(default=ELLIPSOID)
    _enabled = attr.ib(init=False, repr=False, default=False)
    _index = attr.ib(init=False, repr=False, default=None)

    def set(self):
        """
        Set the global ellipsoid to the one given to this context.

        Returns
        -------
        self
        """
        if not self._enabled:
            self.global_context.append(self.ellipsoid)
            self._enabled = True
            self._index = len(self.global_context) - 1
        return self

    def reset(self):
        """
        Reset the global ellipsoid to what it was before this context was set.
        """
        if self._enabled:
            self.global_context.pop(self._index)
            self._enabled = False
            self._index = None

    def __enter__(self):
        "Enter context manager by calling ``set`` to enable the given ellipsoid."
        return self.set()

    def __exit__(self, exc_type, exc_val, exc_tb):
        "Reset the ellipsoid to the previous one."
        self.reset()


# Set the default ellipsoid for all calculations
set_ellipsoid()
