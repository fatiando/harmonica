# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Define a layer of tesseroids.
"""

import warnings

import numpy as np
import verde as vd
import xarray as xr

from .tesseroid_gravity import tesseroid_gravity


def tesseroid_layer(coordinates, surface, reference, properties=None):
    """
    Create a layer of tesseroids of equal size.

    Build a regular grid of tesseroids (spherical prisms) with variable top and bottom
    boundaries and properties like density, magnetization, etc. The function returns a
    :class:`xarray.Dataset` containing ``longitude``, ``latitude``, ``top`` and
    ``bottom`` coordinates, and all physical properties as ``data_var`` s. The
    ``longitude`` and ``latitude`` coordinates correspond to the location of the center
    of each tesseroid.

    Unlike the :func:`harmonica.prism_layer`, the top and bottom boundaries of
    tesseroids, provided via `surface` and `reference`, are not heights relative to a
    reference level (e.g., sea level). Instead, they are radii from the center of the
    body (ellipsoid or sphere) to the top and bottom boundaries of the tesseroids. To
    convert top and bottom boundaries from heights to radii, you can use the
    :meth:`boule.geocentric_radius` method from :mod:`boule` to calculate the radius
    based on the latitude, and add these values to your height values.

    The ``tesseroid_layer`` dataset accessor can be used to access special methods
    and attributes for the layer of tesseroids, like the horizontal dimensions of
    the tesseroids, getting the boundaries of each tesseroids, etc.
    See :class:`DatasetAccessorTesseroidLayer` for the definition of these methods
    and attributes.

    Parameters
    ----------
    coordinates : tuple
        List containing the coordinates of the centers of the tesseroids in
        spherical coordinates in the following order ``longitude`` and
        ``latitude``. All coordinates should be in degrees.
    surface : 2d-array
        Array used to create the uppermost boundary of the tesseroids layer. Values
        should be radii (not heights) in meters. On every point where ``surface`` is
        below ``reference``, the ``surface`` value will be used to set the ``bottom``
        boundary of that tesseroid, while the ``reference`` value will be used
        to set the ``top`` boundary of the tesseroid.
    reference : 2d-array or float
        Reference surface used to create the lowermost boundary of the
        tesseroids layer. It can be either a plane or an irregular surface
        passed as 2d array. Values must be radii (not heights) in meters.
    properties : dict or None
        Dictionary containing the physical properties of the tesseroids. The
        keys must be strings that will be used to name the corresponding
        ``data_var`` inside the :class:`xarray.Dataset`, while the values must
        be 2d-arrays. All physical properties must be passed in SI units. If
        None, no ``data_var`` will be added to the :class:`xarray.Dataset`.
        Default is None.

    Returns
    -------
    dataset : :class:`xarray.Dataset`
        Dataset containing the coordinates of the center of each tesseroid, the
        height of its top and bottom boundaries ans its corresponding physical
        properties.

    See Also
    --------
    harmonica.DatasetAccessorsTesseroidLayer

    Examples
    --------
    >>> import numpy as np
    >>> import boule as bl
    >>> # Create a synthetic relief
    >>> longitude = np.linspace(0, 10, 5)
    >>> latitude = np.linspace(2, 8, 4)
    >>> mesh_longitude, mesh_latitude = np.meshgrid(longitude, latitude)
    >>> surface_heights = np.arange(20, dtype=float).reshape((4, 5))
    >>> # Convert heights to radii
    >>> reference_radii = bl.WGS84.geocentric_radius(mesh_latitude)
    >>> surface_radii = surface_heights + reference_radii
    >>> # Define constant densities
    >>> density = 2670.0 * np.ones_like(surface_radii)
    >>> # Define a layer of tesseroids
    >>> tesseroids = tesseroid_layer(
    ...     (longitude, latitude),
    ...     surface_radii,
    ...     reference=reference_radii,
    ...     properties={"density": density},
    ... )
    >>> print(tesseroids) # doctest: +SKIP
    <xarray.Dataset>
    Dimensions:   (latitude: 4, longitude: 5)
    Coordinates:
      * latitude   (latitude) float64 2.0 4.0 6.0 8.0
      * longitude  (longitude) float64 0.0 2.5 5.0 7.5 10.0
        top       (latitude, longitude) float64 6.378e+06 6.378e+06 ... 6.378e+06
        bottom    (latitude, longitude) float64 6.378e+06 6.378e+06 ... 6.378e+06
    Data variables:
        density   (latitude, longitude) float64 2.67e+03 2.67e+03 ... 2.67e+03
    Attributes:
        longitude_units:  degrees
        latitude_units:   degrees
        radius_units:     meters
        properties_units:  SI
    >>> # Get the boundaries of the layer (will exceed the region)
    >>> boundaries = tesseroids.tesseroid_layer.boundaries
    >>> list(float(b) for b in boundaries)
    [-1.25, 11.25, 1.0, 9.0]
    >>> # Get the boundaries of one of the tesseroids
    >>> tesseroid = tesseroids.tesseroid_layer.get_tesseroid((0, 2))
    >>> list(float(b) for b in tesseroid)
    [3.75, 6.25, 1.0, 3.0, 6378137.0, 6378138.0]
    """
    dims = ("latitude", "longitude")
    # Initialize data and data_names as None
    data, data_names = None, None
    # If properties were passed, then replace data_names and data for its keys
    # and values, respectively
    if properties:
        data_names = tuple(p for p in properties)
        data = tuple(np.asarray(p) for p in properties.values())
    # Create xr.Dataset for tesseroids
    tesseroids = vd.make_xarray_grid(
        coordinates, data=data, data_names=data_names, dims=dims
    )
    _check_regular_grid(tesseroids.longitude.values, tesseroids.latitude.values)
    # Check if tesseroid boundaries are overlapped
    _check_overlap(tesseroids.longitude.values)
    # Append some attributes to the xr.Dataset
    attrs = {
        "longitude_units": "degrees",
        "latitude_units": "degrees",
        "radius_units": "meters",
        "properties_units": "SI",
    }
    tesseroids.attrs = attrs
    # Create the top and bottom coordinates of the prisms
    tesseroids.tesseroid_layer.update_top_bottom(surface, reference)
    return tesseroids


def _check_regular_grid(longitude, latitude):
    """
    Check if the longitude and latitude coordinates define a regular grid.

    .. note:

        This function should live inside Verde in the future

    """
    if not np.allclose(longitude[1] - longitude[0], longitude[1:] - longitude[:-1]):
        msg = "Passed longitude coordinates are not evenly spaced."
        raise ValueError(msg)
    if not np.allclose(latitude[1] - latitude[0], latitude[1:] - latitude[:-1]):
        msg = "Passed latitude coordinates are not evenly spaced."
        raise ValueError(msg)


def _check_overlap(longitude):
    """
    Check if the prisms boundaries are overlapped.
    """
    spacing = longitude[1] - longitude[0]
    if longitude.max() - longitude.min() >= 360 - spacing:
        msg = (
            "Found invalid longitude coordinates that would create overlapping "
            "tesseroids around the globe."
        )
        raise ValueError(msg)


@xr.register_dataset_accessor("tesseroid_layer")
class DatasetAccessorTesseroidLayer:
    """
    Define dataset accessor for layer of tesseroids.

    .. warning::

        This class in not intended to be initialized.
        Use the `tesseroid_layer` accessor for accessing the methods and
        attributes of this class.

    See Also
    --------
    harmonica.tesseroid_layer
    """

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    @property
    def dims(self):
        """
        Return the dims tuple of the prism layer.

        The tuple follows the xarray order: ``"latitude"``, ``"longitude"``.
        """
        return ("latitude", "longitude")

    @property
    def spacing(self):
        """
        Spacing between center of tesseroids.

        Returns
        -------
        s_latitude : float
            Spacing between center of the tesseroids on the latitude direction.
        s_longitude : float
            Spacing between center of the tesseroids on the longitude
            direction.
        """
        latitude, longitude = self._obj.latitude.values, self._obj.longitude.values
        _check_regular_grid(longitude, latitude)
        s_latitude, s_longitude = latitude[1] - latitude[0], longitude[1] - longitude[0]
        return s_latitude, s_longitude

    @property
    def size(self):
        """
        Return the total number of tesseroids on the layer.

        Returns
        -------
        size :  int
            Total number of tesseroids in the layer.
        """
        return self._obj.latitude.size * self._obj.longitude.size

    @property
    def shape(self):
        """
        Return the number of tesseroids on each directions.

        Returns
        -------
        n_latitude : int
            Number of tesseroids on the latitude direction.
        n_longitude : int
            Number of tesseroids on the longitude direction.
        """
        return (self._obj.latitude.size, self._obj.longitude.size)

    @property
    def boundaries(self):
        """
        Boundaries of the layer.

        Returns
        -------
        boundaries : tuple
        Boundaries of the layer of tesseroids in the following order:
        ``longitude_w``, ``longitude_e``, ``latitude_s``, ``latitude_n``
        """
        s_latitude, s_longitude = self.spacing
        longitude_w = self._obj.longitude.values.min() - s_longitude / 2
        longitude_e = self._obj.longitude.values.max() + s_longitude / 2
        latitude_s = self._obj.latitude.values.min() - s_latitude / 2
        latitude_n = self._obj.latitude.values.max() + s_latitude / 2
        return longitude_w, longitude_e, latitude_s, latitude_n

    def update_top_bottom(self, surface, reference):
        """
        Update top and bottom boundaries of the layer.

        Change the values of the ``top`` and ``bottom`` coordinates based on
        the passed ``surface`` and ``reference``. The ``top`` and ``bottom``
        boundaries of every tesseroid will be equal to the corresponding
        ``surface`` and ``reference`` values, respectively, if ``surface`` is
        above the ``reference`` on that point. Otherwise the ``top`` and
        ``bottom`` boundaries of the tesseroid will be equal to its
        corresponding ``reference`` and ``surface``, respectively.

        Parameters
        ----------
        surface : 2d-array
            Array used to create the uppermost boundary of the tesseroid layer.
            All radii should be in meters. On every point where ``surface``
            is below ``reference``, the ``surface`` value will be used to set
            the ``bottom`` boundary of that tesseroid, while the ``reference``
            value will be used to set the ``top`` boundary of the tesseroid.
        reference : 2d-array or float
            Reference surface used to create the lowermost boundary of the
            tesseroid layer. It can be either a plane or an irregular surface
            passed as 2d array. Radii must be in meters.
        """
        surface, reference = np.asarray(surface), np.asarray(reference)
        if surface.shape != self.shape:
            raise ValueError(
                f"Invalid surface array with shape '{surface.shape}'. "
                + "Its shape should be compatible with the coordinates "
                + "of the layer of tesseroids."
            )
        if reference.ndim != 0:
            if reference.shape != self.shape:
                raise ValueError(
                    f"Invalid reference array with shape '{reference.shape}'. "
                    + "Its shape should be compatible with the coordinates "
                    + "of the layer of tesseroids."
                )
        else:
            reference = reference * np.ones(self.shape)
        top = surface.copy()
        bottom = reference.copy()
        reverse = surface < reference
        top[reverse] = reference[reverse]
        bottom[reverse] = surface[reverse]
        self._obj.coords["top"] = (self.dims, top)
        self._obj.coords["bottom"] = (self.dims, bottom)

    def gravity(
        self,
        coordinates,
        field,
        progressbar=False,
        density_name="density",
        thickness_threshold=None,
        **kwargs,
    ):
        """
        Compute the gravity generated by the layer of tesseroids.

        Uses :func:`harmonica.tesseroid_gravity` for computing the gravity field
        generated by the tesseroids of the layer.
        The density of the tesseroids will be assigned from the ``data_var`` chosen
        through the ``density_name`` argument.
        Ignores the tesseroids which ``top`` or ``bottom`` boundaries are
        ``np.nan``s.
        Tesseroids thinner than a given threshold can be optionally ignored through
        the ``thickness_threshold`` argument.
        All ``kwargs`` will be passed to :func:`harmonica.tesseroid_gravity`.

        Parameters
        ----------
        coordinates : list of arrays
            List of arrays containing the ``longitude``, ``latitude`` and
            ``radius`` coordinates of the computation points, defined on
            a spherical geocentric coordinate system. Both ``longitude`` and
            ``latitude`` should be in degrees and ``radius`` in meters.
        field : str
            Gravitational field that wants to be computed.
            The available fields are:
            - Gravitational potential: ``potential``
            - Downward acceleration: ``g_z``
        progressbar : bool (optional)
            If True, a progress bar of the computation will be printed to
            standard error (stderr). Requires :mod:`numba_progress` to be
            installed. Default to ``False``.
        density_name : str (optional)
            Name of the property layer (or ``data_var`` of the
            :class:`xarray.Dataset`) that will be used for the density of each
            tesseroid in the layer. Default to ``"density"``
        thickness_threshold : float or None
            Tesseroids thinner than this threshold will be ignored in the
            forward gravity calculation. If None, every tesseroid with non-zero
            volume will be considered. Default to None.

        Returns
        -------
        result : array
            Gravitational field generated by the tesseroid on the computation
            point. Gravitational potential will be returned in SI units, while
            acceleration components will be in mGal.

        See Also
        --------
        harmonica.tesseroid_gravity
        """
        # Get boundaries and density of the tesseroids
        boundaries = self._to_tesseroids()
        density = self._obj[density_name].values
        # Get the mask for selecting  only the tesseroid whose top boundary,
        # bottom boundary and density have no nans
        mask = self._get_nonans_mask(property_name=density_name)
        # Select only the boundaries and density elements for masked tesseroid
        boundaries = boundaries[mask.ravel()]
        density = density[mask]
        # Discard thin tesseroids and their densities
        if thickness_threshold is not None:
            boundaries, density = _discard_thin_tesseroids(
                boundaries,
                density,
                thickness_threshold,
            )
        # Return gravity field of tesseroids
        return tesseroid_gravity(
            coordinates,
            tesseroids=boundaries,
            density=density,
            field=field,
            progressbar=progressbar,
            **kwargs,
        )

    def _get_nonans_mask(self, property_name=None):
        """
        Build a mask for tesseroid with no nans on top, bottom or a property.

        Parameters
        ----------
        property_name : str (optional)
            Name of the property layer (or ``data_var`` of the
            :class:`xarray.Dataset`) that will be used for masking the tesseroids
            in the layer.

        Returns
        -------
        mask : 2d-array
            Array of bools that can be used as a mask for selecting tesseroids
            with no nans on top boundaries, bottom boundaries ans the passed
            property.
        """
        # Mask the tesseroid that contains no nans on top and bottom boundaries
        mask = np.logical_and(
            np.logical_not(np.isnan(self._obj.top.values)),
            np.logical_not(np.isnan(self._obj.bottom.values)),
        )
        # Mask the tesseroids that contains nans on the selected property
        if property_name is not None:
            mask_property = np.logical_not(np.isnan(self._obj[property_name].values))
            # Warn if a nan is found within the masked property
            if not mask_property[mask].all():
                warnings.warn(
                    f"Found missing values in '{property_name}' property "
                    + "of the tesseroid layer. The tesseroids with nan as "
                    + f"'{property_name}' will be ignored.",
                    stacklevel=1,
                )
            mask = np.logical_and(mask, mask_property)
        return mask

    def _to_tesseroids(self):
        """
        Return the boundaries of each tesseroid of the layer.

        Returns
        -------
        tesseroids : 2d-array
            Array containing the boundaries of each tesseroid of the layer.
            Each row contains the boundaries of each tesseroid in the following
            order: ``longitude_w``, ``longitude_e``, ``latitude_s``,
            ``latitude_n``, ``bottom``, ``top``.
        """
        longitude, latitude = np.meshgrid(
            self._obj.longitude.values, self._obj.latitude.values
        )
        (
            longitude_w,
            longitude_e,
            latitude_s,
            latitude_n,
        ) = self._get_tesseroid_horizontal_boundaries(
            longitude.ravel(), latitude.ravel()
        )
        bottom = self._obj.bottom.values.ravel()
        top = self._obj.top.values.ravel()
        tesseroids = np.vstack(
            (longitude_w, longitude_e, latitude_s, latitude_n, bottom, top)
        ).T
        return tesseroids

    def _get_tesseroid_horizontal_boundaries(self, longitude, latitude):
        """
        Compute the horizontal boundaries of the tesseroid.

        Parameters
        ----------
        latitude: float or array
            Longitude coordinates of the center of the tesseroid
        longitude : float or array
            Longitude coordinates of the center of the tesseroid
        """
        spacing = self.spacing
        longitude_w = longitude - spacing[1] / 2
        longitude_e = longitude + spacing[1] / 2
        latitude_s = latitude - spacing[0] / 2
        latitude_n = latitude + spacing[0] / 2
        return longitude_w, longitude_e, latitude_s, latitude_n

    def get_tesseroid(self, indices):
        """
        Return the boundaries of the chosen tesseroid.

        Parameters
        ----------
        indices : tuple
            Indices of the desired tesseroid of the layer in the following
            order: ``(index_latitude, index_longitude)``.

        Returns
        -------
        tesseroid : tuple
           Boundaries of the tesseroids in the following order:
           ``longitude_w``, ``longitude_e``, ``latitude_s``, ``latitude_n``,
           ``bottom``, ``top``.
        """
        # Get the center of the tesseroid
        center_longitude = self._obj.longitude.values[indices[1]]
        center_latitude = self._obj.latitude.values[indices[0]]
        # Calculate the boundaries of the tesseroid
        (
            longitude_w,
            longitude_e,
            latitude_s,
            latitude_n,
        ) = self._get_tesseroid_horizontal_boundaries(center_longitude, center_latitude)
        bottom = self._obj.bottom.values[indices]
        top = self._obj.top.values[indices]
        # return longitude_w, longitude_e, latitude_s, latitude_n, bottom, top
        return longitude_w, longitude_e, latitude_s, latitude_n, bottom, top


def _discard_thin_tesseroids(
    tesseroids,
    density,
    thickness_threshold,
):
    """
    Discard tesseroids with a thickness below a threshold.

    Parameters
    ----------
    tesseroids : 2d-array
        Array containing the boundaries of the tesseroids in the following order:
        ``longitude_w``, ``longitude_e``, ``latitude_s``, ``latitude_n``,
           ``bottom``, ``top``.
        The array must have the following shape: (``n_tesseroids``, 6), where
        ``n_tesseroids`` is the total number of tesseroids.
    density : 1d-array
        Array containing the density of each tesseroid in kg/m^3. Must have the
        same size as the number of tesseroids.
    thickness_threshold : float
        Tesseroids thinner than this threshold will be discarded.

    Returns
    -------
    tesseroids : 2d-array
        A copy of the ``tesseroids`` array that doesn't include the thin tesseroids.
    density : 1d-array
        A copy of the ``density`` array that doesn't include the density values
        for thin tesseroids.
    """
    bottom, top = tesseroids[:, -2], tesseroids[:, -1]
    # Mark tesseroids with thickness < threshold  as null tesseroids
    thickness = top - bottom
    null_tesseroids = thickness < thickness_threshold
    # Keep only thick tesseroids and their densities
    tesseroids = tesseroids[np.logical_not(null_tesseroids), :]
    density = density[np.logical_not(null_tesseroids)]
    return tesseroids, density
