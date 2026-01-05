# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Custom type hints for harmonica.
"""

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

Coordinates: TypeAlias = Sequence[npt.NDArray[np.float64]] | Sequence[float]
"""
Type alias to represent 3D coordinates.
"""
