"""
Test functions to create synthetic surveys
"""
import numpy.testing as npt
from ..synthetic import airborne_survey, ground_survey


def test_ground_survey():
    """
    Test if the sythetic ground survey returns the expected survey
    """
    # Ground survey without scaling
    expected_region = (
        13.60833,
        20.28333,
        -24.2,
        -17.50333,
    )  # expected region for the default subsection
    survey = ground_survey()
    assert set(survey.columns) == set(["longitude", "latitude", "height"])
    assert survey.longitude.size == 963
    npt.assert_allclose(survey.longitude.min(), expected_region[0])
    npt.assert_allclose(survey.longitude.max(), expected_region[1])
    npt.assert_allclose(survey.latitude.min(), expected_region[2])
    npt.assert_allclose(survey.latitude.max(), expected_region[3])
    npt.assert_allclose(survey.height.min(), 0.0)
    npt.assert_allclose(survey.height.max(), 2052.2)
    # Ground survey with scaling
    region = (-10.1, 9.7, -20.3, -10.5)  # a random region to scale the survey
    survey = ground_survey(region=region)
    assert set(survey.columns) == set(["longitude", "latitude", "height"])
    assert survey.longitude.size == 963
    npt.assert_allclose(survey.longitude.min(), region[0])
    npt.assert_allclose(survey.longitude.max(), region[1])
    npt.assert_allclose(survey.latitude.min(), region[2])
    npt.assert_allclose(survey.latitude.max(), region[3])
    npt.assert_allclose(survey.height.min(), 0.0)
    npt.assert_allclose(survey.height.max(), 2052.2)


def test_airborne_survey():
    """
    Test if the synthetic airborne survey returns the expected survey
    """
    expected_region = (
        -4.99975,
        -4.00003,
        56.00011,
        56.49997,
    )  # expected region for the default subsection
    survey = airborne_survey()
    assert set(survey.columns) == set(["longitude", "latitude", "height"])
    assert survey.longitude.size == 5673
    npt.assert_allclose(survey.longitude.min(), expected_region[0])
    npt.assert_allclose(survey.longitude.max(), expected_region[1])
    npt.assert_allclose(survey.latitude.min(), expected_region[2])
    npt.assert_allclose(survey.latitude.max(), expected_region[3])
    npt.assert_allclose(survey.height.min(), 359.0)
    npt.assert_allclose(survey.height.max(), 1255.0)
    # Ground survey with scaling
    region = (-10.1, 9.7, -20.3, -10.5)  # a random region to scale the survey
    survey = airborne_survey(region=region)
    assert set(survey.columns) == set(["longitude", "latitude", "height"])
    assert survey.longitude.size == 5673
    npt.assert_allclose(survey.longitude.min(), region[0])
    npt.assert_allclose(survey.longitude.max(), region[1])
    npt.assert_allclose(survey.latitude.min(), region[2])
    npt.assert_allclose(survey.latitude.max(), region[3])
    npt.assert_allclose(survey.height.min(), 359.0)
    npt.assert_allclose(survey.height.max(), 1255.0)


def test_subsection_ground_survey():
    """
    Test if ground survey is changed against a different subsection
    """
    subsection = (10, 30, -30, -12)  # a bigger subsection than the default one
    region = (-10.1, 9.7, -20.3, -10.5)  # a random region to scale the survey
    survey = ground_survey(region=region, subsection=subsection)
    assert survey.longitude.size > 963
    npt.assert_allclose(survey.longitude.min(), region[0])
    npt.assert_allclose(survey.longitude.max(), region[1])
    npt.assert_allclose(survey.latitude.min(), region[2])
    npt.assert_allclose(survey.latitude.max(), region[3])
    assert survey.height.min() <= 0.0
    assert survey.height.max() >= 2052.2


def test_subsection_airborne_survey():
    """
    Test if a different cut_region produces a different airborne survey
    """
    subsection = (-7, -2, 53, 58)  # a bigger subsection than the default one
    region = (-10.1, 9.7, -20.3, -10.5)  # a random region to scale the survey
    survey = airborne_survey(region=region, subsection=subsection)
    assert survey.longitude.size > 5673
    npt.assert_allclose(survey.longitude.min(), region[0])
    npt.assert_allclose(survey.longitude.max(), region[1])
    npt.assert_allclose(survey.latitude.min(), region[2])
    npt.assert_allclose(survey.latitude.max(), region[3])
    assert survey.height.min() <= 359.0
    assert survey.height.max() >= 1255.0
