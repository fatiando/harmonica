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
    yaw : float, optional
        Rotation angle about the upward axis, in degrees.
    pitch : float, optional
        Rotation angle about the northing axis (after yaw rotation), in degrees.
        A positive pitch angle *lifts* the side of the ellipsoid pointing in easting
        direction.
    roll : float, optional
        Rotation angle about the easting axis (after yaw and pitch rotation), in
        degrees.
    center : tuple of float, optional
        Coordinates of the center of the ellipsoid in the following order: `easting`,
        `northing`, `upward`, in meters.
    density : float or None, optional
        Density of the ellipsoid in :math:`kg/m^3`.
    susceptibility : float, (3, 3) array or None, optional
        Magnetic susceptibility of the ellipsoid in SI units.
        A single float represents isotropic susceptibility in the body.
        A (3, 3) array represents the susceptibility tensor to account for anisotropy
        (defined in the local coordinate system of the ellipsoid).
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

    .. figure:: ../../_static/figures/ellipsoid-rotations.svg
       :name: ellipsoid rotations
       :width: 90%
       :alt: Figure showing an ellipsoid with arbitrary rotation given by the yaw, pitch and roll angles.

    """

    def __init__(
        self,
        a: float,
        b: float,
        c: float,
        yaw: float = 0.0,
        pitch: float = 0.0,
        roll: float = 0.0,
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
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

    def __str__(self) -> str:
        e, n, u = self.center
        string = (
            f"{type(self).__name__}:\n"
            f"  • a:      {float(self.a)} m\n"
            f"  • b:      {float(self.b)} m\n"
            f"  • c:      {float(self.c)} m\n"
            f"  • center: ({float(e)}, {float(n)}, {float(u)}) m\n"
            f"  • yaw:    {float(self.yaw)}\n"
            f"  • pitch:  {float(self.pitch)}\n"
            f"  • roll:   {float(self.roll)}"
        )

        if self.density is not None:
            string += f"\n  • density: {float(self.density)} kg/m³"
        if self.susceptibility is not None:
            if isinstance(self.susceptibility, Real):
                string += f"\n  • susceptibility: {float(self.susceptibility)}"
            else:
                string += "\n  • susceptibility:"
                matrix_as_str = str(self.susceptibility)
                for line in matrix_as_str.splitlines():
                    string += f"\n        {line}"
        if self.remanent_mag is not None:
            me, mn, mu = self.remanent_mag
            string += f"\n  • remanent_mag: ({float(me)}, {float(mn)}, {float(mu)}) A/m"
        return string

    def __repr__(self):
        module = next(iter(self.__class__.__module__.split(".")))
        attrs = [
            f"a={float(self.a)}",
            f"b={float(self.b)}",
            f"c={float(self.c)}",
            f"center={tuple(float(i) for i in self.center)}",
            f"yaw={float(self.yaw)}",
            f"pitch={float(self.pitch)}",
            f"roll={float(self.roll)}",
        ]

        if self.density is not None:
            attrs.append(f"density={float(self.density)}")

        if self.susceptibility is not None:
            if isinstance(self.susceptibility, Real):
                susceptibility = f"{float(self.susceptibility)}"
            else:
                susceptibility = []
                for line in str(self.susceptibility).splitlines():
                    susceptibility.append(line.strip())
                susceptibility = " ".join(susceptibility)
            attrs.append(f"susceptibility={susceptibility}")

        if self.remanent_mag is not None:
            attrs.append(f"remanent_mag={self.remanent_mag}")

        attrs = ", ".join(attrs)
        return f"{module}.{self.__class__.__name__}({attrs})"

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
