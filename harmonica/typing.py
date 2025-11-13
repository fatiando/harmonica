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

    @property
    def a(self) -> float:
        """Length of the first semiaxis."""
        raise NotImplementedError

    @property
    def b(self) -> float:
        """Length of the second semiaxis."""
        raise NotImplementedError

    @property
    def c(self) -> float:
        """Length of the third semiaxis."""
        raise NotImplementedError

    @property
    def yaw(self) -> float:
        """Yaw angle in degrees."""
        raise NotImplementedError

    @property
    def pitch(self) -> float:
        """Pitch angle in degrees."""
        raise NotImplementedError

    @property
    def roll(self) -> float:
        """Roll angle in degrees."""
        raise NotImplementedError

    @property
    def center(self) -> tuple[float, float, float]:
        """Coordinates of ellipsoid's center."""
        raise NotImplementedError

    @property
    def density(self) -> float | None:
        """Density of the ellipsoid in :math:`kg/m^3`."""
        raise NotImplementedError

    @property
    def susceptibility(self) -> float | npt.NDArray | None:
        """Magnetic susceptibility of the ellipsoid in SI units."""
        raise NotImplementedError

    @property
    def remanent_mag(self) -> npt.NDArray | None:
        """Remanent magnetization of the ellipsoid in A/m."""
        raise NotImplementedError

    @property
    def rotation_matrix(self) -> npt.NDArray:
        """Rotation matrix for the ellipsoid."""
        raise NotImplementedError
