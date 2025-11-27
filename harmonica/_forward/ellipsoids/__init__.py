# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Forward modelling of ellipsoids.
"""

from .ellipsoids import (
    # OblateEllipsoid,
    # ProlateEllipsoid,
    # Sphere,
    # TriaxialEllipsoid,
    # create_ellipsoid,
    Ellipsoid,
)
from .gravity import ellipsoid_gravity
from .magnetic import ellipsoid_magnetic
