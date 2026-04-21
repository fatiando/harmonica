# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Forward modelling of rectangular prisms.
"""

from .gravity import prism_gravity
from .magnetic import prism_magnetic
from .layer import DatasetAccessorPrismLayer, prism_layer
