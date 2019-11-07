"""
Create synthetic surveys for gravity and magnetic observations
"""
import pyproj
from verde import get_region, inside
from verde.coordinates import check_region

from ..datasets import fetch_britain_magnetic, fetch_south_africa_gravity


def airborne_survey(region, cut_region=(-5.0, -4.0, 56.0, 56.5)):
    """
    Create a synthetic ground survey

    The observation points are sampled from the Great Britain total-field magnetic
    anomaly dataset. Only a portion of the original survey is sampled and its region is
    rescaled to the passed ``region``.

    Parameters
    ----------
    region : tuple or list
        Boundaries of the synthetic region where the observation points will be located
        in the following order: (``east``, ``west``, ``south``, ``north``, ...). All
        subsequent boundaries will be ignored. All boundaries should be in Cartesian
        coordinates and in meters.
    cut_region : tuple (optional)
        Region to reduce the extension of the survey. Must be boundaries of the original
        survey, in degrees.

    Returns
    -------
    survey : :class:`pandas.DataFrame`
        Dataframe containing the coordinates of the observation points and their
        elevation Cartesian coordinates for the synthetic model. All coordinates and
        altitude are in meters.

    See also
    --------
    datasets.fetch_britain_magnetic:
        Fetch total-field magnetic anomaly data of Great Britain.
    """
    # Sanity checks for region and cut_region
    check_region(region[:4])
    check_region(cut_region)
    # Fetch airborne magnetic survey from Great Britain
    survey = fetch_britain_magnetic()
    # Rename the "altitude_m" column to "elevation"
    survey["elevation"] = survey["altitude_m"]
    # Cut the region into the cut_region, project it with a mercator projection to
    # convert the coordinates into Cartesian and move this Cartesian region into the
    # passed region
    survey = _adecuate_survey(survey, region, cut_region)
    return survey


def ground_survey(region, cut_region=(13.60, 20.30, -24.20, -17.5)):
    """
    Create a synthetic ground survey

    The observation points are sampled from the South Africa gravity dataset
    (see :func:`harmonica.datasets.fetch_south_africa_gravity`).
    Only a portion of the original survey is sampled and its region is rescaled to the
    passed ``region``.

    Parameters
    ----------
    region : tuple or list
        Boundaries of the synthetic region where the observation points will be located
        in the following order: (``east``, ``west``, ``south``, ``north``, ...). All
        subsequent boundaries will be ignored. All boundaries should be in Cartesian
        coordinates and in meters.
    cut_region : tuple (optional)
        Region to reduce the extension of the survey. Must be boundaries of the original
        survey, in degrees.

    Returns
    -------
    survey : :class:`pandas.DataFrame`
        Dataframe containing the coordinates of the observation points and their
        elevation Cartesian coordinates for the synthetic model. All coordinates and
        altitude are in meters.

    See also
    --------
    datasets.fetch_south_africa_gravity: Fetch gravity station data from South Africa.
    """
    # Sanity checks for region and cut_region
    check_region(region[:4])
    check_region(cut_region)
    # Fetch ground gravity survey from South Africa
    survey = fetch_south_africa_gravity()
    # Cut the region into the cut_region, project it with a mercator projection to
    # convert the coordinates into Cartesian and move this Cartesian region into the
    # passed region
    survey = _adecuate_survey(survey, region, cut_region)
    return survey


def _adecuate_survey(survey, region, cut_region):
    """
    Cut, project and move the original survey to the passed region

    Parameters
    ----------
    survey : :class:`pandas.DataFrame`
        Original survey as a :class:`pandas.DataFrame` containing the following columns:
        ``longitude``, ``latitude`` and ``elevation``. The ``longitude`` and
        ``latitude`` must be in degrees and the ``elevation`` in meters.
    region : tuple or list
        Boundaries of the synthetic region where the observation points will be located
        in the following order: (``east``, ``west``, ``south``, ``north``, ...). All
        subsequent boundaries will be ignored. All boundaries should be in Cartesian
        coordinates and in meters.
    cut_region : tuple (optional)
        Region to reduce the extension of the survey. Must be boundaries of the original
        survey, in degrees.

    Returns
    -------
    survey : :class:`pandas.DataFrame`
        Dataframe containing the coordinates of the observation points and their
        elevation Cartesian coordinates for the synthetic model. All coordinates and
        altitude are in meters.
    """
    # Cut the data into the cut_region
    inside_points = inside((survey.longitude, survey.latitude), cut_region)
    survey = survey[inside_points].copy()
    # Project coordinates
    projection = pyproj.Proj(proj="merc", lat_ts=(cut_region[2] + cut_region[3]) / 2)
    survey["easting"], survey["northing"] = projection(
        survey.longitude.values, survey.latitude.values
    )
    # Move projected coordinates to the boundaries of the region argument
    w, e, s, n = region[:4]
    easting_min, easting_max, northing_min, northing_max = get_region(
        (survey.easting, survey.northing)
    )
    survey["easting"] = (e - w) / (easting_max - easting_min) * (
        survey.easting - easting_min
    ) + w
    survey["northing"] = (n - s) / (northing_max - northing_min) * (
        survey.northing - northing_min
    ) + s
    # Keep only the easting, northing and elevation on the DataFrame
    survey = survey.filter(["easting", "northing", "elevation"])
    return survey
