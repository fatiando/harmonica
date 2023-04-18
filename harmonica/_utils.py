import numpy as np

def magnetic_vec_to_ang(magnetic_e, magnetic_n, magnetic_u, degrees=True):
    r"""
    Convert the 3-component of the magnetic vector to magnetic intensity and inclination
    and declination angles.

    The intensity of the magnetic vector is calculate as:

    .. math::

        I = \sqrt{B_e^2 + B_n^2 + B_u^2}

    where :math:`B_e`, :math:`B_n`, :math:`B_u` are the easting, northing and upward
    components of the magnetic vector, respectively.

    The inclination angle is defined as the angle between the magnetic field vector and
    the horizontal plane:

    .. math::

        Inc = \arctan \frac{-B_u}{\sqrt{B_e^2 + B_n^2}}

    And the declination angle is defined as the azimuth of the projection of the
    magnetic field vector onto the horizontal plane (starting from the northing
    direction, positive to the east and negative to the west):

    .. math::

        Dec = \arcsin \frac{B_e}{\sqrt{B_e^2 + B_n^2}}

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
        If ``degrees`` is True, then the angle is returned in degrees.
        If ``degrees`` is False, then it's returned in radians.
    declination : float or array
        Declination angle of the magnetic vector.
        If ``degrees`` is True, then the angle is returned in degrees.
        If ``degrees`` is False, then it's returned in radians.

    Examples
    --------
    >>> intensity, inc, dec = magnetic_vec_to_ang(1.5, 1.5, 2.121320343559643)
    >>> print(intensity, inc, dec)
    3.000 45.000 45.000
    """
    # Compute the intensity
    vectors = np.vstack((magnetic_e, magnetic_n, magnetic_u)).T
    intensity = np.linalg.norm(vectors)

    # Compute the horizontal component of magnetic vector
    horizontal_component = np.sqrt(magnetic_e**2 + magnetic_n**2)
    if horizontal_component == 0:
        inclination = -np.sign(magnetic_u) * np.pi /2
        declination = 0
    else:
        # Compute the two angles
        inclination = np.arctan(- magnetic_u / horizontal_component)
        declination = np.arcsin(magnetic_e / horizontal_component)
    # Convert to degrees if needed
    if degrees:
        inclination = np.degrees(inclination)
        declination = np.degrees(declination)
    return intensity, inclination, declination


