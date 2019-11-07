"""
Create synthetic surveys for gravity and magnetic observations
"""
from verde import get_region, inside
from verde.coordinates import check_region

from ..datasets import fetch_britain_magnetic, fetch_south_africa_gravity


def airborne_survey(region, subsection=(-5.0, -4.0, 56.0, 56.5)):
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
    check_region(subsection)
    # Fetch airborne magnetic survey from Great Britain
    survey = fetch_britain_magnetic()
    # Rename the "altitude_m" column to "height"
    survey.rename({"altitude_m": "height"})
    # Keep only the longitude, latitude and height on the DataFrame
    survey = survey.filter(["longitude", "latitude", "height"])
    # Cut the region into the cut_region, project it with a mercator projection to
    # convert the coordinates into Cartesian and move this Cartesian region into the
    # passed region
    survey = _cut_and_scale(survey, region, subsection)
    return survey


def ground_survey(region, subsection=(13.60, 20.30, -24.20, -17.5)):
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
    check_region(subsection)
    # Fetch ground gravity survey from South Africa
    survey = fetch_south_africa_gravity()
    # Rename the "elevation" column to "height"
    survey.rename({"elevation": "height"})
    # Keep only the longitude, latitude and height on the DataFrame
    survey = survey.filter(["longitude", "latitude", "height"])
    # Cut the region into the cut_region, project it with a mercator projection to
    # convert the coordinates into Cartesian and move this Cartesian region into the
    # passed region
    survey = _cut_and_scale(survey, region, subsection)
    return survey


def _cut_and_scale(survey, region, subsection):
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
    inside_points = inside((survey.longitude, survey.latitude), subsection)
    survey = survey[inside_points].copy()
    # Scale survey coordinates to the passed region
    w, e, s, n = region[:4]
    longitude_min, longitude_max, latitude_min, latitude_max = get_region(
        (survey.longitude, survey.latitude)
    )
    survey["longitude"] = (e - w) / (longitude_max - longitude_min) * (
        survey.longitude - longitude_min
    ) + w
    survey["latitude"] = (n - s) / (latitude_max - latitude_min) * (
        survey.latitude - latitude_min
    ) + s
    return survey
