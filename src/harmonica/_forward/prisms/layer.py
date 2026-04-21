# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Define a layer of prisms.
"""

import warnings

import numba
import numpy as np
import verde as vd
import xarray as xr

from ...visualization import prism_to_pyvista
from ..utils import initialize_progressbar
from .gravity import FIELDS


def prism_layer(
    coordinates,
    surface,
    reference,
    properties=None,
):
    """
    Create a layer of prisms of equal size.

    Build a regular grid of prisms of equal size on the horizontal directions
    with variable top and bottom boundaries and properties like density,
    magnetization, etc. The function returns a :class:`xarray.Dataset`
    containing ``easting``, ``northing``, ``top`` and ``bottom`` coordinates,
    and all physical properties as ``data_var`` s. The ``easting`` and
    ``northing`` coordinates correspond to the location of the center of each
    prism.

    The ``prism_layer`` dataset accessor can be used to access special methods
    and attributes for the layer of prisms, like the horizontal dimensions of
    the prisms, getting the boundaries of each prisms, etc.
    See :class:`DatasetAccessorPrismLayer` for the definition of these methods
    and attributes.

    Parameters
    ----------
    coordinates : tuple
        List containing the coordinates of the centers of the prisms in the
        following order: ``easting``, ``northing``. The arrays must be 1d
        arrays containing the coordinates of the centers per axis, or could be
        2d arrays as the ones returned by :func:`numpy.meshgrid`. All
        coordinates should be in meters and should define a regular grid.
    surface : 2d-array
        Array used to create the uppermost boundary of the prisms layer. All
        heights should be in meters. On every point where ``surface`` is below
        ``reference``, the ``surface`` value will be used to set the
        ``bottom`` boundary of that prism, while the ``reference`` value will
        be used to set the ``top`` boundary of the prism.
    reference : float or 2d-array
        Reference surface used to create the lowermost boundary of the prisms
        layer. It can be either a plane or an irregular surface passed as 2d
        array. Height(s) must be in meters.
    properties : dict or None
        Dictionary containing the physical properties of the prisms. The keys
        must be strings that will be used to name the corresponding ``data_var``
        inside the :class:`xarray.Dataset`, while the values must be 2d-arrays.
        All physical properties must be passed in SI units. If None, no
        ``data_var`` will be added to the :class:`xarray.Dataset`. Default is
        None.

    Returns
    -------
    dataset : :class:`xarray.Dataset`
        Dataset containing the coordinates of the center of each prism, the
        height of its top and bottom boundaries and its corresponding physical
        properties.

    See Also
    --------
    harmonica.DatasetAccessorPrismLayer

    Examples
    --------
    >>> # Create a synthetic relief
    >>> import numpy as np
    >>> easting = np.linspace(0, 10, 5)
    >>> northing = np.linspace(2, 8, 4)
    >>> surface = np.arange(20, dtype=float).reshape((4, 5))
    >>> density = 2670.0 * np.ones_like(surface)
    >>> # Define a layer of prisms
    >>> prisms = prism_layer(
    ...     (easting, northing),
    ...     surface,
    ...     reference=0,
    ...     properties={"density": density},
    ... )
    >>> print(prisms) # doctest: +SKIP
    <xarray.Dataset>
    Dimensions:   (northing: 4, easting: 5)
    Coordinates:
      * easting   (easting) float64 0.0 2.5 5.0 7.5 10.0
      * northing  (northing) float64 2.0 4.0 6.0 8.0
        top       (northing, easting) float64 0.0 1.0 2.0 3.0 ... 17.0 18.0 19.0
        bottom    (northing, easting) float64 0.0 0.0 0.0 0.0 ... 0.0 0.0 0.0 0.0
    Data variables:
        density   (northing, easting) float64 2.67e+03 2.67e+03 ... 2.67e+03
    Attributes:
        coords_units:      meters
        properties_units:  SI
    >>> # Get the boundaries of the layer (will exceed the region)
    >>> boundaries = prisms.prism_layer.boundaries
    >>> list(float(b) for b in boundaries)
    [-1.25, 11.25, 1.0, 9.0]
    >>> # Get the boundaries of one of the prisms
    >>> prism = prisms.prism_layer.get_prism((0, 2))
    >>> list(float(b) for b in prism)
    [3.75, 6.25, 1.0, 3.0, 0.0, 2.0]
    """
    dims = ("northing", "easting")
    # Initialize data and data_names as None
    data, data_names = None, None
    # If properties were passed, then replace data_names and data for its keys
    # and values, respectively
    if properties:
        data_names = tuple(p for p in properties)
        data = tuple(np.asarray(p) for p in properties.values())
    # Create xr.Dataset for prisms
    prisms = vd.make_xarray_grid(
        coordinates, data=data, data_names=data_names, dims=dims
    )
    _check_regular_grid(prisms.easting.values, prisms.northing.values)
    # Append some attributes to the xr.Dataset
    attrs = {"coords_units": "meters", "properties_units": "SI"}
    prisms.attrs = attrs
    # Create the top and bottom coordinates of the prisms
    prisms.prism_layer.update_top_bottom(surface, reference)
    return prisms


def _check_regular_grid(easting, northing):
    """
    Check if the easting and northing coordinates define a regular grid.

    .. note:

        This function should live inside Verde in the future
    """
    if not np.allclose(easting[1] - easting[0], easting[1:] - easting[:-1]):
        msg = "Passed easting coordinates are not evenly spaced."
        raise ValueError(msg)
    if not np.allclose(northing[1] - northing[0], northing[1:] - northing[:-1]):
        msg = "Passed northing coordinates are not evenly spaced."
        raise ValueError(msg)


@xr.register_dataset_accessor("prism_layer")
class DatasetAccessorPrismLayer:
    """
    Defines dataset accessor for layer of prisms.

    .. warning::

        This class is not intended to be initialized.
        Use the `prism_layer` accessor for accessing the methods and
        attributes of this class.

    See Also
    --------
    harmonica.prism_layer
    """

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    @property
    def dims(self):
        """
        Return the dims tuple of the prism layer.

        The tuple follows the xarray order: ``"northing"``, ``"easting"``.
        """
        return ("northing", "easting")

    @property
    def spacing(self):
        """
        Spacing between center of prisms.

        Returns
        -------
        s_north : float
            Spacing between center of prisms on the South-North direction.
        s_east : float
            Spacing between center of prisms on the West-East direction.
        """
        easting, northing = self._obj.easting.values, self._obj.northing.values
        _check_regular_grid(easting, northing)
        s_north, s_east = northing[1] - northing[0], easting[1] - easting[0]
        return s_north, s_east

    @property
    def boundaries(self):
        """
        Boundaries of the layer.

        Returns
        -------
        boundaries : tuple
            Boundaries of the layer of prisms in the following order: ``west``,
            ``east``, ``south``, ``north``.
        """
        s_north, s_east = self.spacing
        west = self._obj.easting.values.min() - s_east / 2
        east = self._obj.easting.values.max() + s_east / 2
        south = self._obj.northing.values.min() - s_north / 2
        north = self._obj.northing.values.max() + s_north / 2
        return west, east, south, north

    @property
    def size(self):
        """
        Return the total number of prisms on the layer.

        Returns
        -------
        size : int
            Total number of prisms in the layer.
        """
        return self._obj.northing.size * self._obj.easting.size

    @property
    def shape(self):
        """
        Return the number of prisms on each direction.

        Returns
        -------
        n_north : int
            Number of prisms on the South-North direction.
        n_east : int
            Number of prisms on the West-East direction.
        """
        return (self._obj.northing.size, self._obj.easting.size)

    def _get_prism_horizontal_boundaries(self, easting, northing):
        """
        Compute the horizontal boundaries of the prism.

        Parameters
        ----------
        easting : float or array
            Easting coordinate of the center of the prism
        northing : float or array
            Northing coordinate of the center of the prism
        """
        spacing = self.spacing
        west = easting - spacing[1] / 2
        east = easting + spacing[1] / 2
        south = northing - spacing[0] / 2
        north = northing + spacing[0] / 2
        return west, east, south, north

    def update_top_bottom(self, surface, reference):
        """
        Update top and bottom boundaries of the layer.

        Change the values of the ``top`` and ``bottom`` coordinates based on
        the passed ``surface`` and ``reference``. The ``top`` and ``bottom``
        boundaries of every
        prism will be equal to the corresponding ``surface`` and ``reference``
        values, respectively, if ``surface`` is above the ``reference`` on that
        point. Otherwise the ``top`` and ``bottom`` boundaries of the prism
        will be equal to its corresponding ``reference`` and ``surface``,
        respectively.

        Parameters
        ----------
        surface : 2d-array
            Array used to create the uppermost boundary of the prisms layer.
            All heights should be in meters. On every point where ``surface``
            is below ``reference``, the ``surface`` value will be used to set
            the ``bottom`` boundary of that prism, while the ``reference``
            value will be used to set the ``top`` boundary of the prism.
        reference : float or 2d-array
            Reference surface used to create the lowermost boundary of the
            prisms layer. It can be either a plane or an irregular surface
            passed as 2d array. Height(s) must be in meters.
        """
        surface, reference = np.asarray(surface), np.asarray(reference)
        if surface.shape != self.shape:
            raise ValueError(
                f"Invalid surface array with shape '{surface.shape}'. "
                + "Its shape should be compatible with the coordinates "
                + "of the layer of prisms."
            )
        if reference.ndim != 0:
            if reference.shape != self.shape:
                raise ValueError(
                    f"Invalid reference array with shape '{reference.shape}'. "
                    + "Its shape should be compatible with the coordinates "
                    + "of the layer of prisms."
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
        *,
        density_name="density",
        thickness_threshold=None,
        parallel=True,
        progressbar=False,
    ):
        r"""
        Compute the gravity generated by the layer of prisms.

        Compute the gravity field generated by the layer of prisms on a set of
        observation points.
        The density of the prisms will be assigned from the ``data_var`` chosen
        through the ``density_name`` argument.
        Ignores the prisms which ``top`` or ``bottom`` boundaries are
        ``np.nan``'s.
        Prisms thinner than a given threshold can be optionally ignored through
        the ``thickness_threshold`` argument.

        Parameters
        ----------
        coordinates : list of arrays
            List of arrays containing the ``easting``, ``northing`` and
            ``upward`` coordinates of the computation points defined on
            a Cartesian coordinate system. All coordinates should be in meters.
        field : str
            Gravitational field that wants to be computed.
            The available fields are:
            - Gravitational potential: ``potential``
            - Eastward acceleration: ``g_e``
            - Northward acceleration: ``g_n``
            - Downward acceleration: ``g_z``
            - Diagonal tensor components: ``g_ee``, ``g_nn``, ``g_zz``
            - Non-diagonal tensor components: ``g_en``, ``g_ez``, ``g_nz``
        density_name : str, optional
            Name of the property layer (or ``data_var`` of the
            :class:`xarray.Dataset`) that will be used for the density of each
            prism in the layer. Default to ``"density"``
        thickness_threshold : float or None, optional
            Prisms thinner than this threshold will be ignored in the
            forward gravity calculation. If None, every prism with non-zero
            volume will be considered. Default to None.
        parallel : bool, optional
            Whether to run the computation of gravity fields in parallel or in serial.
        progressbar : bool, optional
            If True, a progress bar of the computation will be printed to
            standard error (stderr). Requires :mod:`numba_progress` to be
            installed. Default to ``False``.

        Returns
        -------
        result : array
            Gravitational potential is returned in :math:`\text{J}/\text{kg}`,
            acceleration components in mGal, and tensor components in Eotvos.

        See Also
        --------
        harmonica.prism_gravity
        """
        # Sanity check the field
        if field not in FIELDS:
            msg = f"Gravitational field {field} not recognized"
            raise ValueError(msg)

        # Check if prism layer defines a regular grid in horizontal coords
        _check_regular_grid(self._obj.easting.values, self._obj.northing.values)

        # Ravel coordinates to 1D
        cast = np.broadcast(*coordinates[:3])
        coordinates = tuple(np.atleast_1d(c).ravel() for c in coordinates[:3])

        # Determine parallel or serial forward modelling function
        numba_function = (
            _forward_gravity_prism_layer_parallel
            if parallel
            else _forward_gravity_prism_layer_serial
        )

        # Forward model the prism layer
        thickness_threshold = (
            0.0 if thickness_threshold is None else thickness_threshold
        )
        density = self._obj[density_name].values
        if np.isnan(density).any():
            msg = (
                "Found NaN values in 'density' property of the prisms layer. "
                "Their respective prisms will be ignored."
            )
            warnings.warn(msg, stacklevel=2)
        with initialize_progressbar(coordinates[0].size, progressbar) as progress_proxy:
            result = numba_function(
                coordinates,
                self._obj.easting.values,
                self._obj.northing.values,
                self._obj.bottom.values,
                self._obj.top.values,
                density,
                FIELDS[field],
                thickness_threshold,
                progress_proxy,
            )

        # Invert sign of gravity_u, gravity_eu, gravity_nu
        if field in ("g_z", "g_ez", "g_nz"):
            result *= -1
        # Convert to more convenient units
        if field in ("g_e", "g_n", "g_z"):
            result *= 1e5  # SI to mGal
        # Convert to more convenient units
        if field in ("g_ee", "g_nn", "g_zz", "g_en", "g_ez", "g_nz"):
            result *= 1e9  # SI to Eotvos
        return result.reshape(cast.shape)

    def _to_prisms(self):
        """
        Return the boundaries of each prism of the layer.

        Returns
        -------
        prisms : 2d-array
            Array containing the boundaries of each prism of the layer.
            Each row contains the boundaries of each prism in the following
            order: ``west``, ``east``, ``south``, ``north``, ``bottom``,
            ``top``.
        """
        easting, northing = np.meshgrid(
            self._obj.easting.values, self._obj.northing.values
        )
        west, east, south, north = self._get_prism_horizontal_boundaries(
            easting.ravel(), northing.ravel()
        )
        bottom = self._obj.bottom.values.ravel()
        top = self._obj.top.values.ravel()
        prisms = np.vstack((west, east, south, north, bottom, top)).T
        return prisms

    def get_prism(self, indices):
        """
        Return the boundaries of the chosen prism.

        Parameters
        ----------
        indices : tuple
            Indices of the desired prism of the layer in  the following order:
            ``(index_northing, index_easting)``.

        Returns
        -------
        prism : tuple
           Boundaries of the prisms in the following order:
           ``west``, ``east``, ``south``, ``north``, ``bottom``, ``top``.
        """
        # Get the center of the prism
        center_easting = self._obj.easting.values[indices[1]]
        center_northing = self._obj.northing.values[indices[0]]
        # Calculate the boundaries of the prism
        west, east, south, north = self._get_prism_horizontal_boundaries(
            center_easting, center_northing
        )
        bottom = self._obj.bottom.values[indices]
        top = self._obj.top.values[indices]
        return west, east, south, north, bottom, top

    def to_pyvista(self, drop_null_prisms=True):
        """
        Return a pyvista UnstructuredGrid to plot the PrismLayer.

        Parameters
        ----------
        drop_null_prisms : bool (optional)
            If True, prisms with zero volume or with any :class:`numpy.nan` as
            their top or bottom boundaries won't be included in the
            :class:`pyvista.UnstructuredGrid`.
            If False, every prism in the layer will be included.
            Default True.

        Returns
        -------
        pv_grid : :class:`pyvista.UnstructuredGrid`
            :class:`pyvista.UnstructuredGrid` containing each prism of the
            layer as a hexahedron along with their properties.
        """
        prisms = self._to_prisms()
        null_prisms = np.zeros_like(prisms[:, 0], dtype=bool)
        if drop_null_prisms:
            bottom, top = prisms[:, -2], prisms[:, -1]
            null_prisms = (top == bottom) | (np.isnan(top)) | (np.isnan(bottom))
            prisms = prisms[np.logical_not(null_prisms)]
        # Define properties
        properties = None
        if self._obj.data_vars:
            properties = {
                data_var: np.asarray(self._obj[data_var]).ravel()[
                    np.logical_not(null_prisms)
                ]
                for data_var in self._obj.data_vars
            }
        return prism_to_pyvista(prisms, properties=properties)


def _forward_gravity_prism_layer(
    coordinates,
    prisms_easting,
    prisms_northing,
    prisms_bottom,
    prisms_top,
    densities,
    forward_func,
    thickness_threshold=0.0,
    progress_proxy=None,
):
    """
    Forward model the gravity fields of prisms in a prism layer.

    Builds the boundaries of each one the prisms in the layer on the fly, iterating over
    the easting and northing dimensional coordinates. This function is intended to avoid
    allocating all prism boundaries in memory before computation.

    The function ignores prisms with zero density, prisms thinner than the
    ``thickness_threshold``, and prisms with NaN as top or bottom boundaries.

    .. important::

        This function is intended to be decorated with Numba before use.

    Parameters
    ----------
    coordinates : tuple of (n,) array
        Tuple with the easting, northing, and upward coordinates of the observation
        points.
    prisms_easting : (n_e,) array
        1D array with the easting coordinates of the center of the prisms.
        Use the easting dimensional coordinate of the prism layer.
    prisms_northing : (n_n,) array
        1D array with the northing coordinates of the center of the prisms.
        Use the northing dimensional coordinate of the prism layer.
    prisms_bottom, prisms_top : (n_n, n_e) array
        2D arrays with the bottom and top boundaries of the prisms.
    densities : (n_n, n_e) array
        2D array with the densities of the prisms in kg/m3.
    forward_func : callable
        Choclo function to forward model a gravity field of prisms.
    thickness_threshold : float
        Prisms thinner than this threshold will be ignored in the
        forward gravity calculation. If None, every prism with non-zero
        volume will be considered.
    progress_proxy : :class:`numba_progress.ProgressBar` or None
        Instance of :class:`numba_progress.ProgressBar` that gets updated after
        each iteration on the observation points. Use None if no progress bar
        is should be used.

    Returns
    -------
    result : array
        Gravity field in SI units.
    """
    easting, northing, upward = coordinates
    n_coords = easting.size
    half_spacing_east = (prisms_easting[1] - prisms_easting[0]) / 2
    half_spacing_north = (prisms_northing[1] - prisms_northing[0]) / 2

    # Check if we need to update the progressbar on each iteration
    update_progressbar = progress_proxy is not None

    result = np.zeros(n_coords, dtype=np.float64)
    for i in numba.prange(n_coords):
        for j, easting_center in enumerate(prisms_easting):
            west = easting_center - half_spacing_east
            east = easting_center + half_spacing_east
            for k, northing_center in enumerate(prisms_northing):
                # Ignore prisms with zero or NaN density
                density = densities[k, j]
                if density == 0.0 or np.isnan(density):
                    continue

                # Ignore thin prisms
                bottom, top = prisms_bottom[k, j], prisms_top[k, j]
                if top - bottom < thickness_threshold:
                    continue

                # Ignore prisms with nan top or bottom
                if np.isnan(top) or np.isnan(bottom):
                    continue

                south = northing_center - half_spacing_north
                north = northing_center + half_spacing_north
                result[i] += forward_func(
                    easting[i],
                    northing[i],
                    upward[i],
                    west,
                    east,
                    south,
                    north,
                    bottom,
                    top,
                    density,
                )
        # Update progress bar if called
        if update_progressbar:
            progress_proxy.update(1)
    return result


_forward_gravity_prism_layer_parallel = numba.jit(nopython=True, parallel=True)(
    _forward_gravity_prism_layer
)
_forward_gravity_prism_layer_serial = numba.jit(nopython=True, parallel=False)(
    _forward_gravity_prism_layer
)
