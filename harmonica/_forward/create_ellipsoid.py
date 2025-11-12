# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Classes to define ellipsoids.
"""


class TriaxialEllipsoid:
    """
    Triaxial ellipsoid with arbitrary orientation.

    Define a triaxial ellipsoid whose semi-axes lengths are ``a > b > c``.

    Parameters
    ----------
    a, b, c : floats
        Semi-axis lengths of the ellipsoid. Must satisfy the condition: ``a > b > c``.
    yaw : float
        Rotation angle about the upward axis, in degrees.
    pitch : float
        Rotation angle about the northing axis (after yaw rotation), in degrees.
        A positive pitch angle _lifts_ the side of the ellipsoid pointing in easting
        direction.
    roll : float
        Rotation angle about the easting axis (after yaw and pitch rotation), in
        degrees.
    center : tuple of floats
        Coordinates of the center of the ellipsoid in the following order: _easting_,
        _northing_, _upward_.

    Notes
    -----
    The three semi-axes ``a``, ``b``, and ``c`` are defined parallel to the ``easting``,
    ``northing`` and ``upward`` directions, respectively, before applying any rotation.

    Rotations directed by ``yaw`` and ``roll`` are applied using the right-hand rule
    across their respective axes. Pitch rotations are carried out in the opposite
    direction, so a positive ``pitch`` _lifts_ the side of the ellipsoid pointing in the
    easting direction.

    """

    def __init__(self, a, b, c, yaw, pitch, roll, center):
        if not (a > b > c):
            msg = (
                "Invalid ellipsoid axis lengths for triaxial ellipsoid: "
                f"expected a > b > c but got a = {a}, b = {b}, c = {c}"
            )
            raise ValueError(msg)

        # semiaxes
        self.a = a  # major_axis
        self.b = b  # intermediate_axis
        self.c = c  # minor_axis

        # euler angles
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll

        # Center of ellipsoid
        self.center = center


class ProlateEllipsoid:
    """
    Prolate ellipsoid with arbitrary orientation.

    Define a prolate ellipsoid whose semi-axes lengths are ``a > b = c``.

    Parameters
    ----------
    a, b : floats
        Semi-axis lengths of the ellipsoid. Must satisfy the condition: ``a > b = c``.
    yaw : float
        Rotation angle about the upward axis, in degrees.
    pitch : float
        Rotation angle about the northing axis (after yaw rotation), in degrees.
        A positive pitch angle _lifts_ the side of the ellipsoid pointing in easting
        direction.
    center : tuple of floats
        Coordinates of the center of the ellipsoid in the following order: _easting_,
        _northing_, _upward_.

    Properties
    ----------
    c : float
        Equal to ``b`` by definition.
    roll : float
        Set always equal to zero. Roll rotations have no effect on ``ProlateEllipsoid``s
        due to symmetry.

    Notes
    -----
    The three semi-axes ``a``, ``b``, and ``c`` are defined parallel to the ``easting``,
    ``northing`` and ``upward`` directions, respectively, before applying any rotation.

    Rotations directed by ``yaw`` are applied using the right-hand rule across the
    upward axis. Pitch rotations are carried out in the opposite direction, so
    a positive ``pitch`` _lifts_ the side of the ellipsoid pointing in the easting
    direction.

    Roll rotations are not enabled in the prolate ellipsoid, since they don't have any
    effect due to symmetry. Hence, ``roll`` is always equal to zero for the
    ``ProlateEllipsoid``.
    """

    def __init__(self, a, b, yaw, pitch, center):
        if not (a > b):
            msg = (
                "Invalid ellipsoid axis lengths for prolate ellipsoid: "
                f"expected a > b (= c ) but got a = {a}, b = {b}"
            )
            raise ValueError(msg)

        # semiaxes
        self.a = a  # major_axis
        self.b = b  # minor axis

        # euler angles
        self.yaw = yaw
        self.pitch = pitch

        # center of ellipsoid
        self.center = center

    @property
    def c(self):
        return self.b

    @property
    def roll(self):
        return 0.0


class OblateEllipsoid:
    """
    Oblate ellipsoid with arbitrary orientation.

    Define a prolate ellipsoid whose semi-axes lengths are ``a < b = c``.

    Parameters
    ----------
    a, b : floats
        Semi-axis lengths of the ellipsoid. Must satisfy the condition: ``a < b = c``.
    yaw : float
        Rotation angle about the upward axis, in degrees.
    pitch : float
        Rotation angle about the northing axis (after yaw rotation), in degrees.
        A positive pitch angle _lifts_ the side of the ellipsoid pointing in easting
        direction.
    center : tuple of floats
        Coordinates of the center of the ellipsoid in the following order: _easting_,
        _northing_, _upward_.

    Properties
    ----------
    c : float
        Equal to ``b`` by definition.
    roll : float
        Set always equal to zero. Roll rotations have no effect on ``OblateEllipsoid``s
        due to symmetry.

    Notes
    -----
    The three semi-axes ``a``, ``b``, and ``c`` are defined parallel to the ``easting``,
    ``northing`` and ``upward`` directions, respectively, before applying any rotation.

    Rotations directed by ``yaw`` are applied using the right-hand rule across the
    upward axis. Pitch rotations are carried out in the opposite direction, so
    a positive ``pitch`` _lifts_ the side of the ellipsoid pointing in the easting
    direction.

    Roll rotations are not enabled in the prolate ellipsoid, since they don't have any
    effect due to symmetry. Hence, ``roll`` is always equal to zero for the
    ``OblateEllipsoid``.
    """

    def __init__(self, a, b, yaw, pitch, center):
        if not (a < b):
            msg = (
                "Invalid ellipsoid axis lengths for oblate ellipsoid: "
                f"expected a < b (= c ) but got a = {a}, b = {b}"
            )
            raise ValueError(msg)

        # semiaxes
        self.a = a  # minor ais
        self.b = b  # major axis

        # euler angles
        self.yaw = yaw
        self.pitch = pitch

        # center of ellipsoid
        self.center = center

    @property
    def c(self):
        return self.b

    @property
    def roll(self):
        return 0.0
