# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
import numpy as np


def magnetic_angles_to_vec(intensity, inclination, declination):
    """
    Convert magnetic field angles to magnetic field vector

    Convert intensity, inclination and declination angles of the magnetic field
    to a 3-component magnetic vector.

    .. note::

        Inclination is measured positive downward from the horizontal plane,
        and declination is measured with respect to north, where positive
        angles indicate east.

    Parameters
    ----------
    intensity: float or array
        Intensity (norm) of the magnetic vector in A/m.
    inclination : float or array
        Inclination angle of the magnetic vector in degree.
        It must be in ``degrees``.
    declination : float or array
        Declination angle of the magnetic vector.
        It must be in ``degrees``.

    Returns
    -------
    magnetic_e : float or array
        Easting component of the magnetic vector.
    magnetic_n : float or array
        Northing component of the magnetic vector.
    magnetic_u : float or array
        Upward component of the magnetic vector.

    Examples
    --------
    >>> mag_e, mag_n, mag_u = magnetic_angles_to_vec(3.0, 45.0, 45.0)
    >>> print(mag_e, mag_n, mag_u)
    1.5 1.5 -2.121
    """
    # Transform to radians
    inc_rad = np.radians(inclination)
    dec_rad = np.radians(declination)
    # Calculate the 3 components
    magnetic_e = intensity * np.cos(inc_rad) * np.sin(dec_rad)
    magnetic_n = intensity * np.cos(inc_rad) * np.cos(dec_rad)
    magnetic_u = -intensity * np.sin(inc_rad)
    return magnetic_e, magnetic_n, magnetic_u


def magnetic_vec_to_angles(magnetic_e, magnetic_n, magnetic_u, degrees=True):
    r"""
    Convert magnetic field vector to magnetic field angles

    Convert the 3-component magnetic vector to intensity, and inclination and
    declination angles.

    .. note::

        Inclination is measured positive downward from the horizontal plane and
        declination is measured with respect to North and it is positive east.

    Parameters
    ----------
    magnetic_e : float or array
        Easting component of the magnetic vector.
    magnetic_n : float or array
        Northing component of the magnetic vector.
    magnetic_u : float or array
        Upward component of the magnetic vector.
    degrees : bool (optional)
        If True, the angles are returned in degrees.
        If False, the angles are returned in radians.
        Default True.

    Returns
    -------
    intensity: float or array
        Intensity of the magnetic vector.
    inclination : float or array
        Inclination angle of the magnetic vector.
        If ``degrees`` is True, then the angle is returned in degree, else it's
        returned in radians.
    declination : float or array
        Declination angle of the magnetic vector.
        If ``degrees`` is True, then the angle is returned in degrees, else
        it's returned in radians.

    Notes
    -----
    The intensity of the magnetic vector is calculated as:

    .. math::

        T = \sqrt{B_e^2 + B_n^2 + B_u^2}

    where :math:`B_e`, :math:`B_n`, :math:`B_u` are the easting, northing and
    upward components of the magnetic vector, respectively.

    The inclination angle is defined as the angle between the magnetic field
    vector and the horizontal plane:

    .. math::

        I = \arctan \frac{- B_u}{\sqrt{B_e^2 + B_n^2}}

    And the declination angle is defined as the azimuth of the projection of
    the magnetic field vector onto the horizontal plane (starting from the
    northing direction, positive to the east and negative to the west):

    .. math::

        D = \arctan \frac{B_e}{B_n}

    Examples
    --------
    >>> intensity, inc, dec = magnetic_vec_to_angles(1.5, 1.5, -2.12132)
    >>> print(intensity, inc, dec)
    3.0 45.0 45.0
    """
    # Compute the intensity as a norm
    intensity = np.sqrt(magnetic_e**2 + magnetic_n**2 + magnetic_u**2)
    # Compute the horizontal component of the magnetic vector
    horizontal_component = np.sqrt(magnetic_e**2 + magnetic_n**2)
    # Calculate the inclination and declination
    inclination = np.arctan2(-magnetic_u, horizontal_component)
    declination = np.arctan2(magnetic_e, magnetic_n)
    # Convert to degree if needed
    if degrees:
        inclination = np.degrees(inclination)
        declination = np.degrees(declination)
    return intensity, inclination, declination


def total_field_anomaly(magnetic_field, inclination, declination):
    r"""
    The total field anomaly from the anomalous magnetic field.

    Compute the total field anomaly from the anomalous magnetic field given the
    regional field direction.

    .. note::

        Inclination is measured positive downward from the horizontal plane and
        declination is measured with respect to North and it is positive east.

    Parameters
    ----------
    magnetic_field : tuple of floats or tuple of arrays
        Three component vector of the anomalous magnetic field.
    inclination : float or array
        Inclination angle of the regional field.
        It must be in degrees.
    declination : float or array
        Declination angle of the regional field.
        It must be in degrees.

    Returns
    -------
    total_field_anomaly : float or array
        The magnetic total field anomaly in the same units as the
        ``magnetic_field``.

    Notes
    -----
    Given the magnetic field, :math:`\mathbf{B}`, the regional field,
    :math:`\mathbf{F}`, the total field anomaly can be computed as:

    .. math::

        \Delta T = \mathbf{B} \cdot \mathbf{\hat{F}}

    where :math:`\mathbf{\hat{F}}` is the unit vector in the same direction
    as the regional field.

    Examples
    --------
    >>> tfa = total_field_anomaly([0, 0, -50e3], 90.0, 0.0)
    >>> print(tfa)
    50000.0
    """
    b_e, b_n, b_u = tuple(np.array(i) for i in magnetic_field)
    f_e, f_n, f_u = magnetic_angles_to_vec(1, inclination, declination)
    tfa = b_e * f_e + b_n * f_n + b_u * f_u
    return tfa
