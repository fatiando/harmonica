# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Custom type hints for harmonica.
"""

import numpy.typing as npt
from typing_extensions import Protocol


class Ellipsoid(Protocol):
    """Protocol to define an Ellipsoid object."""

    a: float
    b: float
    c: float
    yaw: float
    pitch: float
    roll: float
    center: tuple[float, float, float]
    density: float | None
    susceptibility: float | npt.NDArray | None
    remanent_mag: npt.NDArray | None

    @property
    def rotation_matrix(self) -> npt.NDArray:
        """Rotation matrix for the ellipsoid."""
        raise NotImplementedError
