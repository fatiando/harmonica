# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Classes to define ellipsoids.
"""
from abc import ABC, abstractmethod

import numpy.typing as npt

from .utils import get_rotation_matrix


class BaseEllipsoid(ABC):
    """
    Base class for ellipsoids.

    .. important::

        This class is not meant to be instantiated.
    """

    @property
    @abstractmethod
    def a(self) -> float:
        """First semiaxes length."""

    @property
    @abstractmethod
    def b(self) -> float:
        """Second semiaxes length."""

    @property
    @abstractmethod
    def c(self) -> float:
        """Third semiaxes length."""

    @property
    @abstractmethod
    def pitch(self) -> float:
        """Pitch angle in degrees."""

    @property
    @abstractmethod
    def roll(self) -> float:
        """Roll angle in degrees."""

    @property
    @abstractmethod
    def yaw(self) -> float:
        """Yaw angle in degrees."""

    @property
    def rotation_matrix(self) -> npt.NDArray:
        """
        Create a rotation matrix for the ellipsoid.

        Use this matrix to rotate from the local coordinate system (centered in the
        ellipsoid center) in to the global coordinate system
        (easting, northing, upward).

        Returns
        -------
        rotation_matrix : (3, 3) array
            Rotation matrix that transforms coordinates from the local ellipsoid-aligned
            coordinate system to the global coordinate system.

        Notes
        -----
        Generate the rotation matrix from Tait-Bryan intrinsic angles: yaw, pitch, and
        roll. The rotations are applied in the following order: (ZŶX). Yaw (Z) and roll
        (X) rotations are done using the right-hand rule. Rotations for the pitch (Ŷ)
        are carried out in the opposite direction, so positive pitch *lifts* the easting
        axis.
        """
        return get_rotation_matrix(self.yaw, self.pitch, self.roll)


class TriaxialEllipsoid(BaseEllipsoid):
    """
    Triaxial ellipsoid with arbitrary orientation.

    Define a triaxial ellipsoid whose semi-axes lengths are ``a > b > c``.

    Parameters
    ----------
    a, b, c : float
        Semi-axis lengths of the ellipsoid. Must satisfy the condition: ``a > b > c``.
    yaw : float
        Rotation angle about the upward axis, in degrees.
    pitch : float
        Rotation angle about the northing axis (after yaw rotation), in degrees.
        A positive pitch angle *lifts* the side of the ellipsoid pointing in easting
        direction.
    roll : float
        Rotation angle about the easting axis (after yaw and pitch rotation), in
        degrees.
    center : tuple of float
        Coordinates of the center of the ellipsoid in the following order: `easting`,
        `northing`, `upward`.

    Notes
    -----
    The three semi-axes ``a``, ``b``, and ``c`` are defined parallel to the ``easting``,
    ``northing`` and ``upward`` directions, respectively, before applying any rotation.

    Rotations directed by ``yaw`` and ``roll`` are applied using the right-hand rule
    across their respective axes. Pitch rotations are carried out in the opposite
    direction, so a positive ``pitch`` *lifts* the side of the ellipsoid pointing in the
    easting direction.

    """

    def __init__(self, a, b, c, yaw, pitch, roll, center):
        self._check_semiaxes_lenghts(a, b, c)

        # semiaxes
        self._a = a  # major_axis
        self._b = b  # intermediate_axis
        self._c = c  # minor_axis

        # euler angles
        self._yaw = yaw
        self._pitch = pitch
        self._roll = roll

        # Center of ellipsoid
        self.center = center

    def _check_semiaxes_lenghts(self, a, b, c):
        if not (a > b > c):
            msg = (
                "Invalid ellipsoid axis lengths for triaxial ellipsoid: "
                f"expected a > b > c but got a = {a}, b = {b}, c = {c}"
            )
            raise ValueError(msg)

    @property
    def a(self) -> float:
        """First semiaxes length."""
        return self._a

    @a.setter
    def a(self, value: float):
        self._check_semiaxes_lenghts(value, self.b, self.c)
        self._a = value

    @property
    def b(self) -> float:
        """Second semiaxes length."""
        return self._b

    @b.setter
    def b(self, value: float):
        self._check_semiaxes_lenghts(self.a, value, self.c)
        self._b = value

    @property
    def c(self) -> float:
        """Third semiaxes length."""
        return self._c

    @c.setter
    def c(self, value: float):
        self._check_semiaxes_lenghts(self.a, self.b, value)
        self._c = value

    @property
    def pitch(self) -> float:
        """Pitch angle in degrees."""
        return self._pitch

    @pitch.setter
    def pitch(self, value: float):
        self._pitch = value

    @property
    def roll(self) -> float:
        """Roll angle in degrees."""
        return self._roll

    @roll.setter
    def roll(self, value: float):
        self._roll = value

    @property
    def yaw(self) -> float:
        """Yaw angle in degrees."""
        return self._yaw

    @yaw.setter
    def yaw(self, value: float):
        self._yaw = value


class ProlateEllipsoid(BaseEllipsoid):
    """
    Prolate ellipsoid with arbitrary orientation.

    Define a prolate ellipsoid whose semi-axes lengths are ``a > b = c``.

    Parameters
    ----------
    a, b : float
        Semi-axis lengths of the ellipsoid. Must satisfy the condition: ``a > b = c``.
    yaw : float
        Rotation angle about the upward axis, in degrees.
    pitch : float
        Rotation angle about the northing axis (after yaw rotation), in degrees.
        A positive pitch angle *lifts* the side of the ellipsoid pointing in easting
        direction.
    center : tuple of float
        Coordinates of the center of the ellipsoid in the following order: `easting`,
        `northing`, `upward`.

    Attributes
    ----------
    c : float
        Equal to ``b`` by definition.
    roll : float
        Set always equal to zero. Roll rotations have no effect on
        :class:`harmonica.ProlateEllipsoid``s due to symmetry.

    Notes
    -----
    The three semi-axes ``a``, ``b``, and ``c`` are defined parallel to the ``easting``,
    ``northing`` and ``upward`` directions, respectively, before applying any rotation.

    Rotations directed by ``yaw`` are applied using the right-hand rule across the
    upward axis. Pitch rotations are carried out in the opposite direction, so
    a positive ``pitch`` *lifts* the side of the ellipsoid pointing in the easting
    direction.

    Roll rotations are not enabled in the prolate ellipsoid, since they don't have any
    effect due to symmetry. Hence, ``roll`` is always equal to zero for the
    :class:`harmonica.ProlateEllipsoid`.
    """

    def __init__(self, a, b, yaw, pitch, center):
        self._check_semiaxes_lenghts(a, b)

        # semiaxes
        self._a = a  # major_axis
        self._b = b  # minor axis

        # euler angles
        self.yaw = yaw
        self.pitch = pitch

        # center of ellipsoid
        self.center = center

    def _check_semiaxes_lenghts(self, a, b):
        if not (a > b):
            msg = (
                "Invalid ellipsoid axis lengths for prolate ellipsoid: "
                f"expected a > b (= c ) but got a = {a}, b = {b}"
            )
            raise ValueError(msg)

    @property
    def a(self) -> float:
        """First semiaxes length."""
        return self._a

    @a.setter
    def a(self, value: float):
        self._check_semiaxes_lenghts(value, self.b)
        self._a = value

    @property
    def b(self) -> float:
        """Second semiaxes length."""
        return self._b

    @b.setter
    def b(self, value: float):
        self._check_semiaxes_lenghts(self.a, value)
        self._b = value

    @property
    def c(self):
        """Length of the third semiaxis, equal to ``b`` by definition."""
        return self.b

    @property
    def pitch(self) -> float:
        """Pitch angle in degrees."""
        return self._pitch

    @pitch.setter
    def pitch(self, value: float):
        self._pitch = value

    @property
    def roll(self):
        """Roll angle, equal to zero."""
        return 0.0

    @property
    def yaw(self) -> float:
        """Yaw angle in degrees."""
        return self._yaw

    @yaw.setter
    def yaw(self, value: float):
        self._yaw = value


class OblateEllipsoid(BaseEllipsoid):
    """
    Oblate ellipsoid with arbitrary orientation.

    Define a prolate ellipsoid whose semi-axes lengths are ``a < b = c``.

    Parameters
    ----------
    a, b : float
        Semi-axis lengths of the ellipsoid. Must satisfy the condition: ``a < b = c``.
    yaw : float
        Rotation angle about the upward axis, in degrees.
    pitch : float
        Rotation angle about the northing axis (after yaw rotation), in degrees.
        A positive pitch angle *lifts* the side of the ellipsoid pointing in easting
        direction.
    center : tuple of float
        Coordinates of the center of the ellipsoid in the following order: `easting`,
        `northing`, `upward`.

    Attributes
    ----------
    c : float
        Equal to ``b`` by definition.
    roll : float
        Set always equal to zero. Roll rotations have no effect on
        :class:`harmonica.OblateEllipsoid``s due to symmetry.

    Notes
    -----
    The three semi-axes ``a``, ``b``, and ``c`` are defined parallel to the ``easting``,
    ``northing`` and ``upward`` directions, respectively, before applying any rotation.

    Rotations directed by ``yaw`` are applied using the right-hand rule across the
    upward axis. Pitch rotations are carried out in the opposite direction, so
    a positive ``pitch`` *lifts* the side of the ellipsoid pointing in the easting
    direction.

    Roll rotations are not enabled in the prolate ellipsoid, since they don't have any
    effect due to symmetry. Hence, ``roll`` is always equal to zero for the
    :class:`harmonica.OblateEllipsoid`.
    """

    def __init__(self, a, b, yaw, pitch, center):
        self._check_semiaxes_lenghts(a, b)

        # semiaxes
        self._a = a  # minor ais
        self._b = b  # major axis

        # euler angles
        self._yaw = yaw
        self._pitch = pitch

        # center of ellipsoid
        self.center = center

    def _check_semiaxes_lenghts(self, a, b):
        if not (a < b):
            msg = (
                "Invalid ellipsoid axis lengths for oblate ellipsoid: "
                f"expected a < b (= c ) but got a = {a}, b = {b}"
            )
            raise ValueError(msg)

    @property
    def a(self) -> float:
        """First semiaxes length."""
        return self._a

    @a.setter
    def a(self, value: float):
        self._check_semiaxes_lenghts(value, self.b)
        self._a = value

    @property
    def b(self) -> float:
        """Second semiaxes length."""
        return self._b

    @b.setter
    def b(self, value: float):
        self._check_semiaxes_lenghts(self.a, value)
        self._b = value

    @property
    def c(self):
        """Length of the third semiaxis, equal to ``b`` by definition."""
        return self.b

    @property
    def pitch(self) -> float:
        """Pitch angle in degrees."""
        return self._pitch

    @pitch.setter
    def pitch(self, value: float):
        self._pitch = value

    @property
    def roll(self):
        """Roll angle, equal to zero."""
        return 0.0

    @property
    def yaw(self) -> float:
        """Yaw angle in degrees."""
        return self._yaw

    @yaw.setter
    def yaw(self, value: float):
        self._yaw = value
