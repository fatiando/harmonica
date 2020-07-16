"""
Base class for equivalent layer gridders
"""
from warnings import warn
import verde.base as vdb

# BaseEQL is a subclass of Verde's BaseGridder. We will disable pylint
# abstract-method error because BaseEQL will not override fit and predict
# methods. Subclasses of BaseEQL should do override them.
# BaseEQL override grid, scatter and profile methods. On EQLs these methods
# should take the upward coordinate as a positional argument. We will disable
# pylint arguments-differ error because we intend to make these methods
# different from the ones that are being inherited.

# pylint: disable=abstract-method,arguments-differ


class BaseEQL(vdb.BaseGridder):
    """
    Base class for equivalent layer gridders

    Most methods of this class requires the implementation of
    a :meth:`~harmonica.equivalent_layer.base.BaseEQL.predict` method. The data
    returned by it should be a 1d or 2d numpy array for scalar data or a tuple
    with 1d or 2d numpy arrays for each component of vector data.

    This is a subclass of :class:`verde.base.BaseGridder`. The main difference
    is that EQL gridders also need the `upward` coordinate of data points and
    of points where the predictions will be carried out. So, this class
    redefines some :class:`verde.base.BaseGridder` methods to improve their
    usability and development.

    Overrides class arguments ``dims`` and ``extra_coords_name`` that set
    default dimensions and upward coordinate name for generated output objects
    by the ``grid``, ``profile`` and ``scatter`` methods.

    Because :class:`verde.base.BaseGridder` is a subclass of
    :class:`sklearn.base.BaseEstimator`, the
    :class:`~harmonica.equivalent_layer.base.BaseEQL` must also abide by the
    same rules of the scikit-learn classes. Mainly:

    * ``__init__`` must **only** assign values to attributes based on the
      parameters it receives. All parameters must have default values.
      Parameter checking should be done in ``fit``.
    * Estimated parameters should be stored as attributes with names ending in
      ``_``.

    """

    # The default dimension names for generated outputs
    # (pd.DataFrame, xr.Dataset, etc)
    dims = ("northing", "easting")

    # Default name for the upward coordinate used on generated outputs
    # (pd.DataFrame, xr.Dataset, etc)
    extra_coords_name = "upward"

    def grid(
        self,
        upward,
        region=None,
        shape=None,
        spacing=None,
        dims=None,
        data_names=None,
        projection=None,
        **kwargs
    ):
        """
        Interpolate the data onto a regular grid.
        The grid can be specified by either the number of points in each
        dimension (the *shape*) or by the grid node spacing. See
        :func:`verde.grid_coordinates` for details. All grid points will be
        located at the same `upward` coordinate. Other arguments for
        :func:`verde.grid_coordinates` can be passed as extra keyword arguments
        (``kwargs``) to this method.

        If the interpolator collected the input data region, then it will be
        used if ``region=None``. Otherwise, you must specify the grid region.
        Use the *dims* and *data_names* arguments to set custom names for the
        dimensions and the data field(s) in the output :class:`xarray.Dataset`.
        Default names will be provided if none are given.

        Parameters
        ----------
        upward : float
            Upward coordinate of the grid points.
        region : list = [W, E, S, N]
            The west, east, south, and north boundaries of a given region.
        shape : tuple = (n_north, n_east) or None
            The number of points in the South-North and West-East directions,
            respectively.
        spacing : tuple = (s_north, s_east) or None
            The grid spacing in the South-North and West-East directions,
            respectively.
        dims : list or None
            The names of the northing and easting data dimensions,
            respectively, in the output grid. Default is determined from the
            ``dims`` attribute of the class. Must be defined in the following
            order: northing dimension, easting dimension.
            **NOTE: This is an exception to the "easting" then
            "northing" pattern but is required for compatibility with xarray.**
        data_names : list of None
            The name(s) of the data variables in the output grid. Defaults to
            ``['scalars']`` for scalar data,
            ``['east_component', 'north_component']`` for 2D vector data, and
            ``['east_component', 'north_component', 'vertical_component']`` for
            3D vector data.
        projection : callable or None
            If not None, then should be a callable object
            ``projection(easting, northing) -> (proj_easting, proj_northing)``
            that takes in easting and northing coordinate arrays and returns
            projected northing and easting coordinate arrays. This function
            will be used to project the generated grid coordinates before
            passing them into ``predict``. For example, you can use this to
            generate a geographic grid from a Cartesian gridder.

        Returns
        -------
        grid : xarray.Dataset
            The interpolated grid. Metadata about the interpolator is written
            to the ``attrs`` attribute.

        """
        # Ignore extra_coords if passed
        _pop_extra_coords(kwargs)
        # Grid data
        grid = super().grid(
            region=region,
            shape=shape,
            spacing=spacing,
            dims=dims,
            data_names=data_names,
            projection=projection,
            extra_coords=upward,
            **kwargs,
        )
        return grid

    def scatter(
        self,
        upward,
        region=None,
        size=300,
        random_state=0,
        dims=None,
        data_names=None,
        projection=None,
        **kwargs
    ):
        """
        Interpolate values onto a random scatter of points.

        Point coordinates are generated by :func:`verde.scatter_points`. Other
        arguments for this function can be passed as extra keyword arguments
        (``kwargs``) to this method.

        If the interpolator collected the input data region, then it will be
        used if ``region=None``. Otherwise, you must specify the grid region.

        Use the *dims* and *data_names* arguments to set custom names for the
        dimensions and the data field(s) in the output
        :class:`pandas.DataFrame`. Default names are provided.

        Parameters
        ----------
        upward: float
            Upward coordinate of the grid points.
        region : list = [W, E, S, N]
            The west, east, south, and north boundaries of a given region.
        size : int
            The number of points to generate.
        random_state : numpy.random.RandomState or an int seed
            A random number generator used to define the state of the random
            permutations. Use a fixed seed to make sure computations are
            reproducible. Use ``None`` to choose a seed automatically
            (resulting in different numbers with each run).
        dims : list or None
            The names of the northing and easting data dimensions,
            respectively, in the output dataframe. Default is determined from
            the ``dims`` attribute of the class. Must be defined in the
            following order: northing dimension, easting dimension.
            **NOTE: This is an exception to the "easting" then
            "northing" pattern but is required for compatibility with xarray.**
        data_names : list of None
            The name(s) of the data variables in the output dataframe. Defaults
            to ``['scalars']`` for scalar data,
            ``['east_component', 'north_component']`` for 2D vector data, and
            ``['east_component', 'north_component', 'vertical_component']`` for
            3D vector data.
        projection : callable or None
            If not None, then should be a callable object
            ``projection(easting, northing) -> (proj_easting, proj_northing)``
            that takes in easting and northing coordinate arrays and returns
            projected northing and easting coordinate arrays. This function
            will be used to project the generated scatter coordinates before
            passing them into ``predict``. For example, you can use this to
            generate a geographic scatter from a Cartesian gridder.

        Returns
        -------
        table : pandas.DataFrame
            The interpolated values on a random set of points.

        """
        # Ignore extra_coords if passed
        _pop_extra_coords(kwargs)
        # Create scatter points and predict
        table = super().scatter(
            region=region,
            size=size,
            random_state=random_state,
            dims=dims,
            data_names=data_names,
            projection=projection,
            extra_coords=upward,
            **kwargs,
        )
        return table

    def profile(
        self,
        point1,
        point2,
        upward,
        size,
        dims=None,
        data_names=None,
        projection=None,
        **kwargs
    ):
        """
        Interpolate data along a profile between two points.

        Generates the profile along a straight line assuming Cartesian
        distances and the same upward coordinate for all points. Point
        coordinates are generated by :func:`verde.profile_coordinates`. Other
        arguments for this function can be passed as extra keyword arguments
        (``kwargs``) to this method.

        Use the *dims* and *data_names* arguments to set custom names for the
        dimensions and the data field(s) in the output
        :class:`pandas.DataFrame`. Default names are provided.

        Includes the calculated Cartesian distance from *point1* for each data
        point in the profile.

        To specify *point1* and *point2* in a coordinate system that would
        require projection to Cartesian (geographic longitude and latitude, for
        example), use the ``projection`` argument. With this option, the input
        points will be projected using the given projection function prior to
        computations. The generated Cartesian profile coordinates will be
        projected back to the original coordinate system. **Note that the
        profile points are evenly spaced in projected coordinates, not the
        original system (e.g., geographic)**.

        Parameters
        ----------
        point1 : tuple
            The easting and northing coordinates, respectively, of the first
            point.
        point2 : tuple
            The easting and northing coordinates, respectively, of the second
            point.
        upward : float
            Upward coordinate of the profile points.
        size : int
            The number of points to generate.
        dims : list or None
            The names of the northing and easting data dimensions,
            respectively, in the output dataframe. Default is determined from
            the ``dims`` attribute of the class. Must be defined in the
            following order: northing dimension, easting dimension.
            **NOTE: This is an exception to the "easting" then
            "northing" pattern but is required for compatibility with xarray.**
        data_names : list of None
            The name(s) of the data variables in the output dataframe. Defaults
            to ``['scalars']`` for scalar data,
            ``['east_component', 'north_component']`` for 2D vector data, and
            ``['east_component', 'north_component', 'vertical_component']`` for
            3D vector data.
        projection : callable or None
            If not None, then should be a callable object ``projection(easting,
            northing, inverse=False) -> (proj_easting, proj_northing)`` that
            takes in easting and northing coordinate arrays and returns
            projected northing and easting coordinate arrays. Should also take
            an optional keyword argument ``inverse`` (default to False) that if
            True will calculate the inverse transform instead. This function
            will be used to project the profile end points before generating
            coordinates and passing them into ``predict``. It will also be used
            to undo the projection of the coordinates before returning the
            results.

        Returns
        -------
        table : pandas.DataFrame
            The interpolated values along the profile.

        """
        # Ignore extra_coords if passed
        _pop_extra_coords(kwargs)
        # Create profile points and predict
        table = super().profile(
            point1,
            point2,
            size,
            dims=None,
            data_names=None,
            projection=None,
            extra_coords=upward,
            **kwargs,
        )
        return table


def _pop_extra_coords(kwargs):
    """
    Remove extra_coords from kwargs
    """
    if "extra_coords" in kwargs:
        warn("EQL gridder will ignore extra_coords: {}.".format(kwargs["extra_coords"]))
        kwargs.pop("extra_coords")
