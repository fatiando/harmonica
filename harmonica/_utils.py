import numpy as np


def magnetic_ang_to_vec(intensity, inclination, declination):
    """
    Convert intensity, inclination and declination angles to a 3-component magnetic
    vector.

    .. note::
        Inclination is measured positive downward from the horizontal plane and
        declination is measured with respect to North and it is positive east.

    Parameters
    ----------
    intensity: float or array
        Intensity (norm) of the magnetic vector in degree.
    inclination : float or array
        Inclination angle of the magnetic vector in degree.
        It mast be in ``degrees``.
        If ``degrees`` is False, then it's returned in radians.
    declination : float or array
        Declination angle of the magnetic vector.
        It mast be in ``degrees``.

    Returns
    -------
    magnetic_vector : array = [magnetic_e, magnetic_n, magnetic_u]
        Magnetic vector.

    Examples
    --------
    >>> magnetic_e, magnetic_n, magnetic_u = magnetic_ang_to_vec(3.0, 45.0, 45.0)
    >>> print(magnetic_e, magnetic_n, magnetic_u)
    1.5, 1.5000000000000002, -2.1213203435596424
    """
    inc_rad = np.radians(inclination)
    dec_rad = np.radians(declination)
    magnetic_e = intensity * np.cos(inc_rad) * np.sin(dec_rad)
    magnetic_n = intensity * np.cos(inc_rad) * np.cos(dec_rad)
    magnetic_u = - intensity *  np.sin(inc_rad)
    return magnetic_e, magnetic_n, magnetic_u

def magnetic_vec_to_ang(magnetic_e, magnetic_n, magnetic_u, degrees=True):
    r"""
    Convert the 3-component of the magnetic vector to magnetic intensity and inclination
    and declination angles.

    The intensity of the magnetic vector is calculate as:

    .. math::

        T = \sqrt{B_e^2 + B_n^2 + B_u^2}

    where :math:`B_e`, :math:`B_n`, :math:`B_u` are the easting, northing and upward
    components of the magnetic vector, respectively.

    The inclination angle is defined as the angle between the magnetic field vector and
    the horizontal plane:

    .. math::

        Inc = \arctan \frac{- B_u}{\sqrt{B_e^2 + B_n^2}}

    And the declination angle is defined as the azimuth of the projection of the
    magnetic field vector onto the horizontal plane (starting from the northing
    direction, positive to the east and negative to the west):

    .. math::

        Dec = \arcsin \frac{B_e}{\sqrt{B_e^2 + B_n^2}}

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
        If ``degrees`` is True, then the angle is returned in degrees, else it's
        returned in radians.

    Examples
    --------
    >>> intensity, inc, dec = magnetic_vec_to_ang(1.5, 1.5, -2.12132)
    >>> print(intensity, inc, dec)
    2.999999757066657, 44.99999536031435, 45.00000000000001
    """
    # Compute the intensity
    vectors = np.vstack((magnetic_e, magnetic_n, magnetic_u)).T
    intensity = np.linalg.norm(vectors, axis=1)
    # Compute the horizontal component of the magnetic vector
    horizontal_component = np.array([np.sqrt(magnetic_e**2 + magnetic_n**2)])
    # Mask the values equal to zero
    horizontal_component = np.ma.masked_values(horizontal_component, 0.)
    # Calculate the inclination and declination using the mask
    inclination = np.arctan(- magnetic_u / horizontal_component)
    declination = np.arcsin(magnetic_e / horizontal_component)
    # Fill the masked values
    inclination = inclination.filled(- np.sign(magnetic_u) * np.pi / 2)
    declination = declination.filled(0)
    # Convert to degree if needed
    if degrees:
        inclination = np.degrees(inclination)
        declination = np.degrees(declination)
    if len(intensity) == 1:
        intensity = intensity[0]
    return intensity, inclination[0], declination[0]
