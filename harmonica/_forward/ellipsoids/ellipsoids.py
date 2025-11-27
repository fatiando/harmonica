# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Classes to define ellipsoids.
"""

from collections.abc import Sequence
from numbers import Real

import numpy as np
import numpy.typing as npt

from ..utils import get_rotation_matrix

try:
    import pyvista
except ImportError:
    pyvista = None


class Ellipsoid:
    """
    Ellipsoidal body with arbitrary rotation.

    Parameters
    ----------
    a, b, c : float
        Semi-axis lengths of the ellipsoid in meters.
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
        `northing`, `upward`, in meters.
    density : float or None, optional
        Density of the ellipsoid in :math:`kg/m^3`.
    susceptibility : float, (3, 3) array or None, optional
        Magnetic susceptibility of the ellipsoid in SI units.
        A single float represents isotropic susceptibility in the body.
        A (3, 3) array represents the susceptibility tensor to account for anisotropy.
        If None, zero susceptibility will be assigned to the ellipsoid.
    remanent_mag : (3,) array or None, optional
        Remanent magnetization vector of the ellipsoid in A/m units. Its components
        are defined in the easting-northing-upward coordinate system and should be
        passed in that order.
        If None, no remanent magnetization will be assigned to the ellipsoid.

    Notes
    -----
    The three semi-axes ``a``, ``b``, and ``c`` are defined parallel to the ``easting``,
    ``northing`` and ``upward`` directions, respectively, before applying any rotation.

    Rotations directed by ``yaw`` and ``roll`` are applied using the right-hand rule
    across their respective axes. Pitch rotations are carried out in the opposite
    direction, so a positive ``pitch`` *lifts* the side of the ellipsoid pointing in the
    easting direction.
    """

    def __init__(
        self,
        a: float,
        b: float,
        c: float,
        yaw: float,
        pitch: float,
        roll: float,
        center: tuple[float, float, float],
        *,
        density: float | None = None,
        susceptibility: float | npt.NDArray | None = None,
        remanent_mag: npt.NDArray | None = None,
    ):
        # Sanity checks on semiaxes
        for value, semiaxis in zip((a, b, c), ("a", "b", "c"), strict=True):
            self._check_positive_semiaxis(value, semiaxis)
        self._a, self._b, self._c = a, b, c

        # Angles and center
        self.yaw, self.pitch, self.roll = yaw, pitch, roll
        self.center = center

        # Physical properties of the ellipsoid
        self.density = density
        self.susceptibility = susceptibility
        self.remanent_mag = remanent_mag

    @property
    def a(self) -> float:
        """Length of the first semiaxis in meters."""
        return self._a

    @a.setter
    def a(self, value: float) -> None:
        self._check_positive_semiaxis(value, "a")
        self._a = value

    @property
    def b(self) -> float:
        """Length of the second semiaxis in meters."""
        return self._b

    @b.setter
    def b(self, value: float):
        self._check_positive_semiaxis(value, "b")
        self._b = value

    @property
    def c(self) -> float:
        """Length of the third semiaxis in meters."""
        return self._c

    @c.setter
    def c(self, value: float):
        self._check_positive_semiaxis(value, "c")
        self._c = value

    @property
    def density(self) -> float | None:
        """Density of the ellipsoid in :math:`kg/m^3`."""
        return self._density

    @density.setter
    def density(self, value) -> None:
        if not isinstance(value, Real) and value is not None:
            msg = (
                f"Invalid 'density' of type '{type(value).__name__}'. "
                "It must be a float or None."
            )
            raise TypeError(msg)
        if isinstance(value, Real):
            value = float(value)
        self._density = value

    @property
    def susceptibility(self) -> float | npt.NDArray | None:
        """Magnetic susceptibility of the ellipsoid in SI units."""
        return self._susceptibility

    @susceptibility.setter
    def susceptibility(self, value) -> None:
        if isinstance(value, np.ndarray):
            if value.shape != (3, 3):
                msg = (
                    f"Invalid 'susceptibility' as an array with shape '{value.shape}'. "
                    "It must be a (3, 3) array, a single float or None."
                )
                raise ValueError(msg)
        elif not isinstance(value, Real) and value is not None:
            msg = (
                f"Invalid 'susceptibility' of type '{type(value).__name__}'. "
                "It must be a (3, 3) array, a single float or None."
            )
            raise TypeError(msg)
        if isinstance(value, Real):
            value = float(value)
        self._susceptibility = value

    @property
    def remanent_mag(self) -> npt.NDArray | None:
        """Remanent magnetization of the ellipsoid in A/m."""
        return self._remanent_mag

    @remanent_mag.setter
    def remanent_mag(self, value) -> None:
        if isinstance(value, Sequence):
            value = np.asarray(value)
        if isinstance(value, np.ndarray):
            if value.shape != (3,):
                msg = (
                    f"Invalid 'remanent_mag' with shape '{value.shape}'. "
                    "It must be a (3,) array or None."
                )
                raise ValueError(msg)
        elif value is not None:
            msg = (
                f"Invalid 'remanent_mag' of type '{type(value).__name__}'. "
                "It must be a (3,) array or None."
            )
            raise TypeError(msg)
        self._remanent_mag = value

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

    def _check_positive_semiaxis(self, value: float, semiaxis: str):
        """
        Raise error if passed semiaxis length value is not positive.

        Parameters
        ----------
        value : float
            Value of the semiaxis length.
        semiaxis : {"a", "b", "c"}
            Semiaxis name. Used to generate the error message.
        """
        if value <= 0:
            msg = f"Invalid value of '{semiaxis}' equal to '{value}'. It must be positive."
            raise ValueError(msg)

    def to_pyvista(self, **kwargs):
        """
        Export ellipsoid to a :class:`pyvista.PolyData` object.

        .. important::

            The :mod:`pyvista` optional dependency must be installed to use this method.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments passed to :func:`pyvista.ParametricEllipsoid`.

        Returns
        -------
        ellipsoid : pyvista.PolyData
            A PyVista's parametric ellipsoid.
        """
        if pyvista is None:
            msg = (
                "Missing optional dependency 'pyvista' required for "
                "exporting ellipsoids to PyVista."
            )
            raise ImportError(msg)
        ellipsoid = pyvista.ParametricEllipsoid(
            xradius=self.a, yradius=self.b, zradius=self.c, **kwargs
        )
        ellipsoid.rotate(rotation=self.rotation_matrix, inplace=True)
        ellipsoid.translate(self.center, inplace=True)
        return ellipsoid


# def create_ellipsoid(
#     a: float,
#     b: float,
#     c: float,
#     *,
#     center: tuple[float, float, float],
#     yaw: float = 0.0,
#     pitch: float = 0.0,
#     roll: float = 0.0,
#     **kwargs,
# ) -> "BaseEllipsoid":
#     """
#     Create an ellipsoid given its three semiaxes lenghts.
#
#     This function returns an ellipsoid object of the correct type based on the values of
#     the three semiaxes lenghts. Semiaxes lengths can be passed in any order.
#
#     Parameters
#     ----------
#     a, b, c: float
#         Semiaxes lenghts in meters. They can be passed in any particular order. They
#         must all be positive values.
#     center : tuple of float
#         Coordinates of the center of the ellipsoid in the following order: `easting`,
#         `northing`, `upward`.
#     yaw : float, optional
#         Rotation angle about the upward axis, in degrees.
#     pitch : float, optional
#         Rotation angle about the northing axis (after yaw rotation), in degrees.
#         A positive pitch angle *lifts* the side of the ellipsoid pointing in easting
#         direction.
#     roll : float, optional
#         Rotation angle about the easting axis (after yaw and pitch rotation), in
#         degrees.
#     **kwargs : dict
#         Extra arguments passed to the ellipsoid classes. They can be physical properties
#         like ``density``, ``susceptibility``, and ``remanent_mag``.
#
#     Returns
#     -------
#     ellipsoid : harmonica.typing.Ellipsoid
#         Instance of one of the ellipsoid classes: :class:`harmonica.OblateEllipsoid`,
#         :class:`harmonica.ProlateEllipsoid`, :class:`harmonica.Sphere`, or
#         :class:`harmonica.TriaxialEllipsoid`.
#
#     Examples
#     --------
#     Define a prolate, oblate, triaxial or spherical ellipsoid by passing its semiaxes in
#     any order:
#
#     >>> ellipsoid = create_ellipsoid(1, 1, 2, center=(1., 2., 3.))
#     >>> print(type(ellipsoid).__name__)
#     ProlateEllipsoid
#     >>> ellipsoid.a
#     2
#     >>> ellipsoid.b
#     1
#     >>> ellipsoid.c
#     1
#
#     >>> ellipsoid = create_ellipsoid(1, 2, 2, center=(1., 2., 3.))
#     >>> print(type(ellipsoid).__name__)
#     OblateEllipsoid
#     >>> ellipsoid.a
#     1
#     >>> ellipsoid.b
#     2
#     >>> ellipsoid.c
#     2
#
#     >>> ellipsoid = create_ellipsoid(1, 2, 3, center=(1., 2., 3.))
#     >>> print(type(ellipsoid).__name__)
#     TriaxialEllipsoid
#     >>> ellipsoid.a
#     3
#     >>> ellipsoid.b
#     2
#     >>> ellipsoid.c
#     1
#
#     We can specify rotation angles while creating the ellipsoid:
#
#     >>> ellipsoid = create_ellipsoid(1, 1, 2, yaw=85, pitch=32, center=(1., 2., 3.))
#     >>> ellipsoid.yaw
#     85
#     >>> ellipsoid.pitch
#     32
#
#     The function can also take physical properties:
#
#     >>> density, susceptibility = -200.0, 0.4
#     >>> ellipsoid = create_ellipsoid(
#     ...     1, 1, 2, center=(1., 2., 3.),
#     ...     density=density,
#     ...     susceptibility=susceptibility,
#     ... )
#     """
#     # Sanity checks
#     for semiaxis, value in zip(("a", "b", "c"), (a, b, c), strict=True):
#         if value <= 0:
#             msg = (
#                 f"Invalid value of '{semiaxis}' equal to '{value}'. "
#                 "It must be positive."
#             )
#             raise ValueError(msg)
#
#     if a == b == c:
#         return Sphere(a, center=center, **kwargs)
#
#     # Sort semiaxes so a >= b >= c
#     c, b, a = sorted((a, b, c))
#
#     if a == b:
#         a = c  # set `a` as the smallest semiaxis (`c`)
#         ellipsoid = OblateEllipsoid(a, b, yaw=yaw, pitch=pitch, center=center, **kwargs)
#     elif b == c:
#         ellipsoid = ProlateEllipsoid(
#             a, b, yaw=yaw, pitch=pitch, center=center, **kwargs
#         )
#     else:
#         ellipsoid = TriaxialEllipsoid(
#             a, b, c, yaw=yaw, pitch=pitch, roll=roll, center=center, **kwargs
#         )
#
#     return ellipsoid
#
#
# class BaseEllipsoid:
#     """
#     Base class for ellipsoids.
#
#     .. important::
#
#         This class is not meant to be instantiated.
#     """
#
#     @property
#     def density(self) -> float | None:
#         """Density of the ellipsoid in :math:`kg/m^3`."""
#         return self._density
#
#     @density.setter
#     def density(self, value) -> None:
#         if not isinstance(value, Real) and value is not None:
#             msg = (
#                 f"Invalid 'density' of type '{type(value).__name__}'. "
#                 "It must be a float or None."
#             )
#             raise TypeError(msg)
#         if isinstance(value, Real):
#             value = float(value)
#         self._density = value
#
#     @property
#     def susceptibility(self) -> float | npt.NDArray | None:
#         """Magnetic susceptibility of the ellipsoid in SI units."""
#         return self._susceptibility
#
#     @susceptibility.setter
#     def susceptibility(self, value) -> None:
#         if isinstance(value, np.ndarray):
#             if value.shape != (3, 3):
#                 msg = (
#                     f"Invalid 'susceptibility' as an array with shape '{value.shape}'. "
#                     "It must be a (3, 3) array, a single float or None."
#                 )
#                 raise ValueError(msg)
#         elif not isinstance(value, Real) and value is not None:
#             msg = (
#                 f"Invalid 'susceptibility' of type '{type(value).__name__}'. "
#                 "It must be a (3, 3) array, a single float or None."
#             )
#             raise TypeError(msg)
#         if isinstance(value, Real):
#             value = float(value)
#         self._susceptibility = value
#
#     @property
#     def remanent_mag(self) -> npt.NDArray | None:
#         """Remanent magnetization of the ellipsoid in A/m."""
#         return self._remanent_mag
#
#     @remanent_mag.setter
#     def remanent_mag(self, value) -> None:
#         if isinstance(value, Sequence):
#             value = np.asarray(value)
#         if isinstance(value, np.ndarray):
#             if value.shape != (3,):
#                 msg = (
#                     f"Invalid 'remanent_mag' with shape '{value.shape}'. "
#                     "It must be a (3,) array or None."
#                 )
#                 raise ValueError(msg)
#         elif value is not None:
#             msg = (
#                 f"Invalid 'remanent_mag' of type '{type(value).__name__}'. "
#                 "It must be a (3,) array or None."
#             )
#             raise TypeError(msg)
#         self._remanent_mag = value
#
#     @property
#     def rotation_matrix(self) -> npt.NDArray:
#         """
#         Create a rotation matrix for the ellipsoid.
#
#         Use this matrix to rotate from the local coordinate system (centered in the
#         ellipsoid center) in to the global coordinate system
#         (easting, northing, upward).
#
#         Returns
#         -------
#         rotation_matrix : (3, 3) array
#             Rotation matrix that transforms coordinates from the local ellipsoid-aligned
#             coordinate system to the global coordinate system.
#
#         Notes
#         -----
#         Generate the rotation matrix from Tait-Bryan intrinsic angles: yaw, pitch, and
#         roll. The rotations are applied in the following order: (ZŶX). Yaw (Z) and roll
#         (X) rotations are done using the right-hand rule. Rotations for the pitch (Ŷ)
#         are carried out in the opposite direction, so positive pitch *lifts* the easting
#         axis.
#         """
#         return get_rotation_matrix(self.yaw, self.pitch, self.roll)
#
#     def _check_positive_semiaxis(self, value: float, semiaxis: str):
#         """
#         Raise error if passed semiaxis length value is not positive.
#
#         Parameters
#         ----------
#         value : float
#             Value of the semiaxis length.
#         semiaxis : {"a", "b", "c"}
#             Semiaxis name. Used to generate the error message.
#         """
#         if value <= 0:
#             msg = f"Invalid value of '{semiaxis}' equal to '{value}'. It must be positive."
#             raise ValueError(msg)
#
#     def to_pyvista(self, **kwargs):
#         """
#         Export ellipsoid to a :class:`pyvista.PolyData` object.
#
#         .. important::
#
#             The :mod:`pyvista` optional dependency must be installed to use this method.
#
#         Parameters
#         ----------
#         kwargs : dict
#             Keyword arguments passed to :func:`pyvista.ParametricEllipsoid`.
#
#         Returns
#         -------
#         ellipsoid : pyvista.PolyData
#             A PyVista's parametric ellipsoid.
#         """
#         if pyvista is None:
#             msg = (
#                 "Missing optional dependency 'pyvista' required for "
#                 "exporting ellipsoids to PyVista."
#             )
#             raise ImportError(msg)
#         ellipsoid = pyvista.ParametricEllipsoid(
#             xradius=self.a, yradius=self.b, zradius=self.c, **kwargs
#         )
#         ellipsoid.rotate(rotation=self.rotation_matrix, inplace=True)
#         ellipsoid.translate(self.center, inplace=True)
#         return ellipsoid
#
#
# class TriaxialEllipsoid(BaseEllipsoid):
#     """
#     Triaxial ellipsoid with arbitrary orientation.
#
#     Define a triaxial ellipsoid whose semi-axes lengths are ``a > b > c``.
#
#     Parameters
#     ----------
#     a, b, c : float
#         Semi-axis lengths of the ellipsoid. Must satisfy the condition: ``a > b > c``.
#     yaw : float
#         Rotation angle about the upward axis, in degrees.
#     pitch : float
#         Rotation angle about the northing axis (after yaw rotation), in degrees.
#         A positive pitch angle *lifts* the side of the ellipsoid pointing in easting
#         direction.
#     roll : float
#         Rotation angle about the easting axis (after yaw and pitch rotation), in
#         degrees.
#     center : tuple of float
#         Coordinates of the center of the ellipsoid in the following order: `easting`,
#         `northing`, `upward`.
#     density : float or None, optional
#         Density of the ellipsoid in :math:`kg/m^3`.
#     susceptibility : float, (3, 3) array or None, optional
#         Magnetic susceptibility of the ellipsoid in SI units.
#         A single float represents isotropic susceptibility in the body.
#         A (3, 3) array represents the susceptibility tensor to account for anisotropy.
#         If None, zero susceptibility will be assigned to the ellipsoid.
#     remanent_mag : (3,) array or None, optional
#         Remanent magnetization vector of the ellipsoid in A/m units. Its components
#         are defined in the easting-northing-upward coordinate system and should be
#         passed in that order.
#         If None, no remanent magnetization will be assigned to the ellipsoid.
#
#     Notes
#     -----
#     The three semi-axes ``a``, ``b``, and ``c`` are defined parallel to the ``easting``,
#     ``northing`` and ``upward`` directions, respectively, before applying any rotation.
#
#     Rotations directed by ``yaw`` and ``roll`` are applied using the right-hand rule
#     across their respective axes. Pitch rotations are carried out in the opposite
#     direction, so a positive ``pitch`` *lifts* the side of the ellipsoid pointing in the
#     easting direction.
#
#     """
#
#     def __init__(
#         self,
#         a: float,
#         b: float,
#         c: float,
#         yaw: float,
#         pitch: float,
#         roll: float,
#         center: tuple[float, float, float],
#         *,
#         density: float | None = None,
#         susceptibility: float | npt.NDArray | None = None,
#         remanent_mag: npt.NDArray | None = None,
#     ):
#         # Sanity checks on semiaxes
#         for value, semiaxis in zip((a, b, c), ("a", "b", "c"), strict=True):
#             self._check_positive_semiaxis(value, semiaxis)
#         self._check_semiaxes_lenghts(a, b, c)
#         self._a, self._b, self._c = a, b, c
#
#         # Angles and center
#         self.yaw, self.pitch, self.roll = yaw, pitch, roll
#         self.center = center
#
#         # Physical properties of the ellipsoid
#         self.density = density
#         self.susceptibility = susceptibility
#         self.remanent_mag = remanent_mag
#
#     @property
#     def a(self) -> float:
#         """Length of the first semiaxis."""
#         return self._a
#
#     @a.setter
#     def a(self, value: float) -> None:
#         self._check_positive_semiaxis(value, "a")
#         self._check_semiaxes_lenghts(value, self.b, self.c)
#         self._a = value
#
#     @property
#     def b(self) -> float:
#         """Length of the second semiaxis."""
#         return self._b
#
#     @b.setter
#     def b(self, value: float):
#         self._check_positive_semiaxis(value, "b")
#         self._check_semiaxes_lenghts(self.a, value, self.c)
#         self._b = value
#
#     @property
#     def c(self) -> float:
#         """Length of the third semiaxis."""
#         return self._c
#
#     @c.setter
#     def c(self, value: float):
#         self._check_positive_semiaxis(value, "c")
#         self._check_semiaxes_lenghts(self.a, self.b, value)
#         self._c = value
#
#     def _check_semiaxes_lenghts(self, a, b, c):
#         if not (a > b > c):
#             msg = (
#                 "Invalid ellipsoid axis lengths for triaxial ellipsoid: "
#                 f"expected a > b > c but got a = {a}, b = {b}, c = {c}"
#             )
#             raise ValueError(msg)
#
#
# class ProlateEllipsoid(BaseEllipsoid):
#     """
#     Prolate ellipsoid with arbitrary orientation.
#
#     Define a prolate ellipsoid whose semi-axes lengths are ``a > b = c``.
#
#     Parameters
#     ----------
#     a, b : float
#         Semi-axis lengths of the ellipsoid. Must satisfy the condition: ``a > b = c``.
#     yaw : float
#         Rotation angle about the upward axis, in degrees.
#     pitch : float
#         Rotation angle about the northing axis (after yaw rotation), in degrees.
#         A positive pitch angle *lifts* the side of the ellipsoid pointing in easting
#         direction.
#     center : tuple of float
#         Coordinates of the center of the ellipsoid in the following order: `easting`,
#         `northing`, `upward`.
#     density : float or None, optional
#         Density of the ellipsoid in :math:`kg/m^3`.
#     susceptibility : float or None, optional
#         Magnetic susceptibility of the ellipsoid in SI units.
#         A single float represents isotropic susceptibility in the body.
#         A (3, 3) array represents the susceptibility tensor to account for anisotropy.
#         If None, zero susceptibility will be assigned to the ellipsoid.
#     remanent_mag : (3,) array or None, optional
#         Remanent magnetization vector of the ellipsoid in A/m units. Its components
#         are defined in the easting-northing-upward coordinate system and should be
#         passed in that order.
#         If None, no remanent magnetization will be assigned to the ellipsoid.
#
#     Notes
#     -----
#     The three semi-axes ``a``, ``b``, and ``c`` are defined parallel to the ``easting``,
#     ``northing`` and ``upward`` directions, respectively, before applying any rotation.
#
#     Rotations directed by ``yaw`` are applied using the right-hand rule across the
#     upward axis. Pitch rotations are carried out in the opposite direction, so
#     a positive ``pitch`` *lifts* the side of the ellipsoid pointing in the easting
#     direction.
#
#     Roll rotations are not enabled in the prolate ellipsoid, since they don't have any
#     effect due to symmetry. Hence, ``roll`` is always equal to zero for the
#     :class:`harmonica.ProlateEllipsoid`.
#     """
#
#     def __init__(
#         self,
#         a: float,
#         b: float,
#         yaw: float,
#         pitch: float,
#         center: tuple[float, float, float],
#         *,
#         density: float | None = None,
#         susceptibility: float | npt.NDArray | None = None,
#         remanent_mag: npt.NDArray | None = None,
#     ):
#         # Sanity checks on semiaxes
#         for value, semiaxis in zip((a, b), ("a", "b"), strict=True):
#             self._check_positive_semiaxis(value, semiaxis)
#         self._check_semiaxes_lenghts(a, b)
#         self._a, self._b = a, b
#
#         # Angles and center
#         self.yaw, self.pitch = yaw, pitch
#         self.center = center
#
#         # Physical properties of the ellipsoid
#         self.density = density
#         self.susceptibility = susceptibility
#         self.remanent_mag = remanent_mag
#
#     @property
#     def a(self) -> float:
#         """Length of the first semiaxis."""
#         return self._a
#
#     @a.setter
#     def a(self, value: float):
#         self._check_positive_semiaxis(value, "a")
#         self._check_semiaxes_lenghts(value, self.b)
#         self._a = value
#
#     @property
#     def b(self) -> float:
#         """Length of the second semiaxis."""
#         return self._b
#
#     @b.setter
#     def b(self, value: float):
#         self._check_positive_semiaxis(value, "b")
#         self._check_semiaxes_lenghts(self.a, value)
#         self._b = value
#
#     @property
#     def c(self):
#         """Length of the third semiaxis, equal to ``b`` by definition."""
#         return self.b
#
#     @property
#     def roll(self):
#         """Roll angle, equal to zero."""
#         return 0.0
#
#     def _check_semiaxes_lenghts(self, a, b):
#         if not (a > b):
#             msg = (
#                 "Invalid ellipsoid axis lengths for prolate ellipsoid: "
#                 f"expected a > b (= c ) but got a = {a}, b = {b}"
#             )
#             raise ValueError(msg)
#
#
# class OblateEllipsoid(BaseEllipsoid):
#     """
#     Oblate ellipsoid with arbitrary orientation.
#
#     Define a prolate ellipsoid whose semi-axes lengths are ``a < b = c``.
#
#     Parameters
#     ----------
#     a, b : float
#         Semi-axis lengths of the ellipsoid. Must satisfy the condition: ``a < b = c``.
#     yaw : float
#         Rotation angle about the upward axis, in degrees.
#     pitch : float
#         Rotation angle about the northing axis (after yaw rotation), in degrees.
#         A positive pitch angle *lifts* the side of the ellipsoid pointing in easting
#         direction.
#     center : tuple of float
#         Coordinates of the center of the ellipsoid in the following order: `easting`,
#         `northing`, `upward`.
#     density : float or None, optional
#         Density of the ellipsoid in :math:`kg/m^3`.
#     susceptibility : float or None, optional
#         Magnetic susceptibility of the ellipsoid in SI units.
#         A single float represents isotropic susceptibility in the body.
#         A (3, 3) array represents the susceptibility tensor to account for anisotropy.
#         If None, zero susceptibility will be assigned to the ellipsoid.
#     remanent_mag : (3,) array or None, optional
#         Remanent magnetization vector of the ellipsoid in A/m units. Its components
#         are defined in the easting-northing-upward coordinate system and should be
#         passed in that order.
#         If None, no remanent magnetization will be assigned to the ellipsoid.
#
#     Notes
#     -----
#     The three semi-axes ``a``, ``b``, and ``c`` are defined parallel to the ``easting``,
#     ``northing`` and ``upward`` directions, respectively, before applying any rotation.
#
#     Rotations directed by ``yaw`` are applied using the right-hand rule across the
#     upward axis. Pitch rotations are carried out in the opposite direction, so
#     a positive ``pitch`` *lifts* the side of the ellipsoid pointing in the easting
#     direction.
#
#     Roll rotations are not enabled in the prolate ellipsoid, since they don't have any
#     effect due to symmetry. Hence, ``roll`` is always equal to zero for the
#     :class:`harmonica.OblateEllipsoid`.
#     """
#
#     def __init__(
#         self,
#         a: float,
#         b: float,
#         yaw: float,
#         pitch: float,
#         center: tuple[float, float, float],
#         *,
#         density: float | None = None,
#         susceptibility: float | npt.NDArray | None = None,
#         remanent_mag: npt.NDArray | None = None,
#     ):
#         # Sanity checks on semiaxes
#         for value, semiaxis in zip((a, b), ("a", "b"), strict=True):
#             self._check_positive_semiaxis(value, semiaxis)
#         self._check_semiaxes_lenghts(a, b)
#         self._a, self._b = a, b
#
#         # Angles and center
#         self.yaw, self.pitch = yaw, pitch
#         self.center = center
#
#         # Physical properties of the ellipsoid
#         self.density = density
#         self.susceptibility = susceptibility
#         self.remanent_mag = remanent_mag
#
#     @property
#     def a(self) -> float:
#         """Length of the first semiaxis."""
#         return self._a
#
#     @a.setter
#     def a(self, value: float):
#         self._check_positive_semiaxis(value, "a")
#         self._check_semiaxes_lenghts(value, self.b)
#         self._a = value
#
#     @property
#     def b(self) -> float:
#         """Length of the second semiaxis."""
#         return self._b
#
#     @b.setter
#     def b(self, value: float):
#         self._check_positive_semiaxis(value, "b")
#         self._check_semiaxes_lenghts(self.a, value)
#         self._b = value
#
#     @property
#     def c(self):
#         """Length of the third semiaxis, equal to ``b`` by definition."""
#         return self.b
#
#     @property
#     def roll(self):
#         """Roll angle, equal to zero."""
#         return 0.0
#
#     def _check_semiaxes_lenghts(self, a, b):
#         if not (a < b):
#             msg = (
#                 "Invalid ellipsoid axis lengths for oblate ellipsoid: "
#                 f"expected a < b (= c ) but got a = {a}, b = {b}"
#             )
#             raise ValueError(msg)
#
#
# class Sphere(BaseEllipsoid):
#     """
#     Homogeneous sphere.
#
#     Represent a sphere as a particular type of ellipsoid where ``a == b == c``.
#
#     Parameters
#     ----------
#     a : float
#         Radius or semiaxes lengths in meters.
#     center : tuple of float
#         Coordinates of the center of the sphere in the following order: `easting`,
#         `northing`, `upward`. All must be in meters.
#     density : float or None, optional
#         Density of the sphere in :math:`kg/m^3`.
#     susceptibility : float, (3, 3) array or None, optional
#         Magnetic susceptibility of the sphere in SI units.
#         A single float represents isotropic susceptibility in the body.
#         A (3, 3) array represents the susceptibility tensor to account for anisotropy.
#         If None, zero susceptibility will be assigned to the sphere.
#     remanent_mag : (3,) array or None, optional
#         Remanent magnetization vector of the sphere in A/m units. Its components
#         are defined in the easting-northing-upward coordinate system and should be
#         passed in that order.
#         If None, no remanent magnetization will be assigned to the sphere.
#
#     Notes
#     -----
#     All semiaxes (``a``, ``b`` and ``c``) are equal to each other.
#     All rotation angles (``yaw``, ``pitch`` and ``roll``) are equal to zero for the
#     sphere, since sphere is invariant to rotations.
#     """
#
#     def __init__(
#         self,
#         a: float,
#         center: tuple[float, float, float],
#         *,
#         density: float | None = None,
#         susceptibility: float | npt.NDArray | None = None,
#         remanent_mag: npt.NDArray | None = None,
#     ):
#         self._check_positive_semiaxis(a, "a")
#         self._a = a
#         self.center = center
#
#         # Physical properties of the sphere
#         self.density = density
#         self.susceptibility = susceptibility
#         self.remanent_mag = remanent_mag
#
#     @property
#     def a(self) -> float:
#         """Length of the first semiaxis."""
#         return self._a
#
#     @a.setter
#     def a(self, value: float) -> None:
#         self._check_positive_semiaxis(value, "a")
#         self._a = value
#
#     @property
#     def b(self) -> float:
#         """Length of the second semiaxis."""
#         return self._a
#
#     @property
#     def c(self) -> float:
#         """Length of the third semiaxis."""
#         return self._a
#
#     @property
#     def radius(self) -> float:
#         """Sphere radius."""
#         return self._a
#
#     @property
#     def yaw(self):
#         """Yaw angle, equal to zero."""
#         return 0.0
#
#     @property
#     def pitch(self):
#         """Pitch angle, equal to zero."""
#         return 0.0
#
#     @property
#     def roll(self):
#         """Roll angle, equal to zero."""
#         return 0.0
#
#     @property
#     def rotation_matrix(self) -> npt.NDArray:
#         """
#         Create a rotation matrix for the sphere.
#
#         .. important:
#
#             The rotation matrix for the sphere is always the identity matrix.
#
#         Returns
#         -------
#         rotation_matrix : (3, 3) array
#             Identity matrix.
#         """
#         return np.eye(3, dtype=np.float64)
