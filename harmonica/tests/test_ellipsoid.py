"""
Testing the setting of different ellipsoids for calculations.
"""

from ..ellipsoid import (
    get_ellipsoid,
    set_ellipsoid,
    ReferenceEllipsoid,
    EllipsoidManager,
)


def test_set_ellipsoid():
    "Set the ellipsoid for a script"
    assert get_ellipsoid().name == "WGS84"
    ellie = ReferenceEllipsoid(
        name="Ellie",
        semimajor_axis=1,
        inverse_flattening=1,
        geocentric_grav_const=1,
        angular_velocity=1,
    )
    set_ellipsoid(ellie)
    assert get_ellipsoid().name == "Ellie"
    set_ellipsoid()
    assert get_ellipsoid().name == "WGS84"


def test_set_ellipsoid_by_name():
    "Set the ellipsoid for a script using the name"
    assert get_ellipsoid().name == "WGS84"
    set_ellipsoid("GRS80")
    assert get_ellipsoid().name == "GRS80"
    set_ellipsoid()
    assert get_ellipsoid().name == "WGS84"


def test_set_ellipsoid_by_name_context():
    "Set the ellipsoid in a context using the name"
    assert get_ellipsoid().name == "WGS84"
    with set_ellipsoid("GRS80"):
        assert get_ellipsoid().name == "GRS80"
    assert get_ellipsoid().name == "WGS84"


def test_ellipsoid_context():
    "Make sure setup and tear down actions work properly"
    ctx = EllipsoidManager("myell", global_context=[])
    # Set it globally
    ctx.set()
    assert ctx.global_context == ["myell"]
    ctx.reset()
    assert ctx.global_context == []
    # Use it as a context manager
    with ctx:
        assert ctx.global_context == ["myell"]
    assert ctx.global_context == []
    # Set before entering the context
    ctx.set()
    assert ctx.global_context == ["myell"]
    with ctx:
        assert ctx.global_context == ["myell"]
    assert ctx.global_context == []
    # Reset before leaving the context
    with ctx:
        assert ctx.global_context == ["myell"]
        ctx.reset()
        assert ctx.global_context == []
    assert ctx.global_context == []
