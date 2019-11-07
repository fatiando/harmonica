"""
Create synthetic surveys for gravity and magnetic observations
"""
from verde import get_region, inside
from verde.coordinates import check_region

from ..datasets import fetch_britain_magnetic, fetch_south_africa_gravity


def airborne_survey(region=None, subsection=(-5.0, -4.0, 56.0, 56.5)):
    """
    Create measurement locations for a synthetic airborne survey.

    The observation points are sampled from the Great Britain total-field magnetic
    anomaly dataset (see :func:`harmonica.datasets.fetch_britain_magnetic`).
    A portion of the original survey is cut (*data_region*) and the coordinates may be
    scaled to the given *region*.

    Parameters
    ----------
    region : tuple or list (optional)
        Survey horizontal coordinates will be scaled to span this area.
        The boundaries must be passed in the following order:
        (``east``, ``west``, ``south``, ``north``, ...), defined on a geodetic
        coordinate system and in degrees.
        Only the 4 horizontal boundaries are used. Subsequent boundaries will be ignored.
        If ``None``, the survey points won't be scaled. Default ``None``.
    subsection : tuple or list (optional)
        Subsection of the original Great Britain magnetic dataset that will be sampled.
        The boundaries must be passed in the following order:
        (``east``, ``west``, ``south``, ``north``, ...), defined on a geodetic
        coordinate system and in degrees. All subsequent boundaries will be ignored.

    Returns
    -------
    survey : :class:`pandas.DataFrame`
        Dataframe containing the coordinates of the observation points on a geodetic
        coordinate system. Longitudes and latitudes are in degrees, and heights in
        meters.

    See also
    --------
    datasets.fetch_britain_magnetic:
        Fetch total-field magnetic anomaly data of Great Britain.
    """
    # Sanity checks for region and subsection
    if region is not None:
        check_region(region[:4])
    check_region(subsection)
    # Fetch airborne magnetic survey from Great Britain
    survey = fetch_britain_magnetic()
    # Rename the "elevation" column to "height" and
    # keep only the longitude, latitude and height
    survey = survey.rename(columns={"altitude_m": "height"}).filter(
        ["longitude", "latitude", "height"]
    )
    # Cut the survey into the subsection and scale it to the passed region
    survey = _cut_and_scale(survey, region, subsection)
    return survey


def ground_survey(region=None, subsection=(13.60, 20.30, -24.20, -17.5)):
    """
    Create a synthetic ground survey

    The observation points are sampled from the South Africa gravity dataset
    (see :func:`harmonica.datasets.fetch_south_africa_gravity`).
    Only a portion of the original survey is sampled and its region may be scaled to the
    passed ``region``.

    Parameters
    ----------
    region : tuple or list (optional)
        Region at which the survey points coordinates will be scaled.
        The boundaries must be passed in the following order:
        (``east``, ``west``, ``south``, ``north``, ...), defined on a geodetic
        coordinate system and in degrees.
        All subsequent boundaries will be ignored.
        If ``None``, the survey points won't be scaled. Default ``None``.
    subsection : tuple or list (optional)
        Region where the original Great Britain magnetic dataset will be sampled.
        The boundaries must be passed in the following order:
        (``east``, ``west``, ``south``, ``north``, ...), defined on a geodetic
        coordinate system and in degrees. All subsequent boundaries will be ignored.

    Returns
    -------
    survey : :class:`pandas.DataFrame`
        Dataframe containing the coordinates of the observation points on a geodetic
        coordinate system. Longitudes and latitudes are in degrees, and heights in
        meters.

    See also
    --------
    datasets.fetch_south_africa_gravity: Fetch gravity station data from South Africa.
    """
    # Sanity checks for region and subsection
    if region is not None:
        check_region(region[:4])
    check_region(subsection)
    # Fetch ground gravity survey from South Africa
    survey = fetch_south_africa_gravity()
    # Rename the "elevation" column to "height" and
    # keep only the longitude, latitude and height
    survey = survey.rename(columns={"elevation": "height"}).filter(
        ["longitude", "latitude", "height"]
    )
    # Cut the survey into the subsection and scale it to the passed region
    survey = _cut_and_scale(survey, region, subsection)
    return survey


def _cut_and_scale(survey, region, subsection):
    """
    Cut a subsection from the original survey and scale it to the given region.

    Parameters
    ----------
    survey : :class:`pandas.DataFrame`
        Original survey as a :class:`pandas.DataFrame` containing the following columns:
        ``longitude``, ``latitude`` and ``height``.
    region : tuple or list (optional)
        Region to which the survey points coordinates will be scaled.
        The boundaries must be passed in the following order:
        (``east``, ``west``, ``south``, ``north``, ...), defined on a geodetic
        coordinate system and in degrees.
        All subsequent boundaries will be ignored.
        If ``None``, the survey points won't be scaled.
    subsection : tuple or list (optional)
        Region where the original Great Britain magnetic dataset will be sampled.
        The boundaries must be passed in the following order:
        (``east``, ``west``, ``south``, ``north``, ...), defined on a geodetic
        coordinate system and in degrees. All subsequent boundaries will be ignored.

    Returns
    -------
    survey : :class:`pandas.DataFrame`
        Dataframe containing the coordinates of the observation points on a geodetic
        coordinate system. Longitudes and latitudes are in degrees, and heights in
        meters.
    """
    # Cut the data into the subsection
    inside_points = inside((survey.longitude, survey.latitude), subsection)
    survey = survey[inside_points].copy()
    # Scale survey coordinates to the passed region
    if region is not None:
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
