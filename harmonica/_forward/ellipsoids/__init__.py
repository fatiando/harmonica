"""
Forward modelling of ellipsoids.
"""

from .ellipsoids import (
    ProlateEllipsoid,
    OblateEllipsoid,
    Sphere,
    TriaxialEllipsoid,
    create_ellipsoid,
)
from .gravity import ellipsoid_gravity
from .magnetic import ellipsoid_magnetic
