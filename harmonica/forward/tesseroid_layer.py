"""
Define a layer of tesseroids
"""
import warnings

import numpy as np
import verde as vd
import xarray as xr


def tesseroid_layer(
    coordinates,
    surface,
    reference,
    properties=None,
):
    """
    Create a layer of tesseroids of equal size

    Parameters
    ----------
    coordinates : tuple
        List containing the coordinates of the centers of the tesseroids in
        spherical coordinates in the following order ``latitude`` and ``longitude``.
    surface : 2d-array
        Array used to create the uppermost boundary of the tesserois layer. All
        heights should be in meters. On every point where ``surface`` is below
        ``reference``, the ``surface`` value will be used to set the ``bottom``
        boundary of that tesseroid, while the ``reference`` value will be used
        to set the ``top`` boundary of the tesseroid.
    reference : 2d-array or float
        Reference surface used to create the lowermost boundary of the tesseroids
        layer. It can be either a plane or an irregular surface passed as 2d array.
        Height(s) must be in meters.
    properties : dict or None
        Dictionary containing the physical properties of the tesseroids. The keys must
        be strings that will be used to name the corresponding ``data_var`` inside the
        :class:`xarray.Dataset`, while the values must be 2d-arrays. All physical
        properties must be passed in SI units. If None, no ``data_var`` will be
        added to the :class:`xarray.Dataset`. Default is None.

    Returns
    -------
    dataset : :class:`xarray.Dataset`
        Dataset containing the coordinates of the center of each tesseroid, the
        height of its top and bottom boundaries ans its corresponding physical properties.

    See also
    --------
    harmonica.DatasetAccessorsTesseroidLayer

    """
    dims = ("latitude", "longitude")
    # Initialize data and data_names as None
    data, data_names = None, None
    # If properties were passed, then replace data_names and data for its keys
    # and values, respectively
    if properties:
        data_names = tuple(p for p in properties.keys())
        data = tuple(np.asarray(p) for p in properties.values())
    # Create xr.Dataset for tesseroids
    tesseroids = vd.make_xarray_grid(
        coordinates, data=data, data_names=data_names, dims=dims
    )
    _check_regular_grid(tesseroids.latitude.values, tesseroids.longitude.values)
    # Append some attributes to the xr.Dataset
    attrs = {}
    tesseroids.attrs = attrs
    # Create the top and bottom coordinates of the prisms
    tesseroids.tesseroid_layer.update_top_bottom(surface, reference)
    return tesseroids


def _check_regular_grid(latitude, longitude):
    """
    Check if the latitude and longitude coordinates define a regular grid

    .. note:

        This function should live inside Verde in the future

    """
    if not np.allclose(latitude[1] - latitude[0], latitude[1:] - latitude[:-1]):
        raise ValueError("Passed latitude coordinates are note evenly spaced.")
    if not np.allclose(longitude[1] - longitude[0], longitude[1:] - longitude[:-1]):
        raise ValueError("Passed longitude coordinates are note evenly spaced.")


@xr.register_dataset_accessor("tesseroid_layer")
class DatasetAccessorTesseroidLayer:
    """
    Define dataset accessor for layer of tesseroids

    .. warning::

        This class in not intended to be initialized.
        Use the `tesseroid_layer` accessor for accessing the methods and attributes
        of this class.

    See also
    --------
    harmonica.tesseroid_layer
    """

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    @property
    def dims(self):
        """
        Return the dims tuple of the prism layer

        The tuple follows the xarray order: ``"latitude"``, ``"longitude"``.
        """
        return ("latitude", "longitude")

    @property
    def spacing(self):
        """
        Spacing between center of tesseroids

        Returns
        -------
        s_latitude : float
            Spacing between center of the tesseroids on the latitude direction.
        s_longitude : float
            Spacing between center of the tesseroids on the longitude direction.
        """
        latitude, longitude = self._obj.latitude.values, self._obj.longitude.values
        _check_regular_grid(latitude, longitude)
        s_latitude, s_longitude = latitude[1] - latitude[0], longitude[1] - longitude[0]
        return s_latitude, s_longitude

    @property
    def size(self):
        """
        Return the total number of tesseroids on the layer

        Returns
        -------
        size :  int
            Total number of tesseroids in the layer.
        """
        return self._obj.latitude.size * self._obj.longitude.size

    @property
    def shape(self):
        """
        Return the number of tesseroids on each directions

        Returns
        -------
        n_latitude : int
            Number of tesseroids on the latitude direction.
        n_longitude : int
            Number of tesserods on the longitude direction.
        """
        return (self._obj.latitude.size, self._obj.longitude.size)

    def update_top_bottom(self, surface, reference):
        """
        Update top and bottom boundaries of the layer

        Change the values of the ``top`` and ``bottom`` coordinates based on
        the passed ``surface`` and ``reference``. The ``top`` and ``bottom``
        boundaries of every tesseroid will be equal to the corresponding ``surface``
        and ``reference`` values, respectively, if ``surface`` is above the
        ``reference`` on that point. Otherwise the ``top`` and ``bottom`` boundaries
        of the tesseroid will be equal to its corresponding ``reference`` and
        ``surface``, respectively.

        Parameters
        ----------
        surface : 2d-array
            Array used to create the uppermost boundary of the tesseroid layer.
            All heights should be in meters. On every point where ``surface``
            is below ``reference``, the ``surface`` value will be used to set
            the ``bottom`` boundary of that tesseroid, while the ``reference``
            value will be used to set the ``top`` boundary of the tesseroid.

        reference : 2d-array or float
            Reference surface used to create the lowermost boundary of the
            tesseroid layer. It can be either a plane or an irregular surface
            passed as 2d array. Height(s) must be in meters.
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
