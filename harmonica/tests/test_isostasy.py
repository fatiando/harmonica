"""
Testing isostasy calculation
"""
import numpy as np
import numpy.testing as npt

from ..isostasy import isostasy_airy


def test_zero_topography():
    "Test zero root for zero topography"
    topography = np.array([0], dtype=np.float64)
    density_upper_crust = 1
    density_lower_crust = 2
    density_mantle = 4
    density_oceanic_crust = 3
    density_water = 1
    npt.assert_equal(
        isostasy_airy(
            topography, density_upper_crust, density_lower_crust, density_mantle
        ),
        topography,
    )
    npt.assert_equal(
        isostasy_airy(
            topography,
            density_upper_crust,
            density_lower_crust,
            density_mantle,
            density_oceanic_crust=density_oceanic_crust,
            density_water=density_water,
        ),
        topography,
    )


def test_no_zero_topography():
    "Test no zero root for no zero topography"
    topography = np.array([-2, -1, 0, 1, 2, 3], dtype=np.float64)
    root = topography.copy()
    root[:2] *= 2
    root[2:] *= 0.5
    root_no_oceanic = root.copy()
    root_no_oceanic[:2] = np.nan
    density_upper_crust = 1
    density_lower_crust = 2
    density_mantle = 4
    density_oceanic_crust = 3
    density_water = 1
    npt.assert_equal(
        isostasy_airy(
            topography, density_upper_crust, density_lower_crust, density_mantle
        ),
        root_no_oceanic,
    )
    npt.assert_equal(
        isostasy_airy(
            topography,
            density_upper_crust,
            density_lower_crust,
            density_mantle,
            density_oceanic_crust=density_oceanic_crust,
            density_water=density_water,
        ),
        root,
    )
