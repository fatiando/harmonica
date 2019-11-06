"""
Test functions to create synthetic surveys
"""
import numpy.testing as npt
from ..synthetic import airborne_survey, ground_survey


def test_ground_survey():
    """
    Test if the sythetic ground survey returns the expected survey
    """
    region = (-1.2e3, 4.3e3, -6.4e3, -1.3e3)
    survey = ground_survey(region)
    assert set(survey.columns) == set(["easting", "northing", "elevation"])
    assert survey.easting.size == 963
    npt.assert_allclose(survey.easting.min(), region[0])
    npt.assert_allclose(survey.easting.max(), region[1])
    npt.assert_allclose(survey.northing.min(), region[2])
    npt.assert_allclose(survey.northing.max(), region[3])
    npt.assert_allclose(survey.elevation.min(), 0.0)
    npt.assert_allclose(survey.elevation.max(), 2052.2)


def test_airborne_survey():
    """
    Test if the synthetic airborne survey returns the expected survey
    """
    region = (-1.2e3, 4.3e3, -6.4e3, -1.3e3)
    survey = airborne_survey(region)
    assert set(survey.columns) == set(["easting", "northing", "elevation"])
    assert survey.easting.size == 5673
    npt.assert_allclose(survey.easting.min(), region[0])
    npt.assert_allclose(survey.easting.max(), region[1])
    npt.assert_allclose(survey.northing.min(), region[2])
    npt.assert_allclose(survey.northing.max(), region[3])
    npt.assert_allclose(survey.elevation.min(), 359.0)
    npt.assert_allclose(survey.elevation.max(), 1255.0)


def test_cut_region_ground_survey():
    """
    Test if a different cut_region produces a different ground survey
    """
    region = (-50e3, 50e3, -50e3, 50e3)
    cut_region = (10, 30, -30, -12)
    survey = ground_survey(region, cut_region=cut_region)
    assert survey.easting.size > 963
    npt.assert_allclose(survey.easting.min(), region[0])
    npt.assert_allclose(survey.easting.max(), region[1])
    npt.assert_allclose(survey.northing.min(), region[2])
    npt.assert_allclose(survey.northing.max(), region[3])
    assert survey.elevation.min() <= 0.0
    assert survey.elevation.max() >= 2052.2


def test_cut_region_airborne_survey():
    """
    Test if a different cut_region produces a different airborne survey
    """
    region = (-50e3, 50e3, -50e3, 50e3)
    cut_region = (-7, -2, 53, 58)
    survey = airborne_survey(region, cut_region=cut_region)
    assert survey.easting.size > 5673
    npt.assert_allclose(survey.easting.min(), region[0])
    npt.assert_allclose(survey.easting.max(), region[1])
    npt.assert_allclose(survey.northing.min(), region[2])
    npt.assert_allclose(survey.northing.max(), region[3])
    assert survey.elevation.min() <= 359.0
    assert survey.elevation.max() >= 1255.0
