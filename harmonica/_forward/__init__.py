from .create_ellipsoid import OblateEllipsoid, ProlateEllipsoid, TriaxialEllipsoid
from .ellipsoid_gravity import (
    _get_ABC,
    _get_gravity_array,
    _get_gravity_oblate,
    _get_gravity_prolate,
    _get_gravity_triaxial,
    _get_internal_g,
    ellipsoid_gravity,
)
from .ellipsoid_magnetics import ellipsoid_magnetics
from .utils_ellipsoids import (
    _calculate_lambda,
    _generate_basic_ellipsoid,
    _get_V_as_Euler,
    _global_to_local,
)
