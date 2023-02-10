.. _api:

API Reference
=============

.. automodule:: harmonica

.. currentmodule:: harmonica

Gravity Corrections
-------------------

.. autosummary::
    :toctree: generated/

    bouguer_correction

For the Normal Earth correction, see package :mod:`boule`.

Grid Transformations
--------------------

Apply well known transformations regular gridded potential fields data.

.. autosummary::
    :toctree: generated/

    derivative_easting
    derivative_northing
    derivative_upward
    upward_continuation
    gaussian_lowpass
    gaussian_highpass
    reduction_to_pole

Frequency domain filters
------------------------

Define filters in the frequency domain.

.. autosummary::
    :toctree: generated/

    filters.derivative_easting_kernel
    filters.derivative_northing_kernel
    filters.derivative_upward_kernel
    filters.upward_continuation_kernel
    filters.gaussian_lowpass_kernel
    filters.gaussian_highpass_kernel
    filters.reduction_to_pole_kernel
    filters.pseudo_gravity_kernel

Use :func:`xrft.xrft.fft` and :func:`xrft.xrft.ifft` to apply Fast-Fourier
Transforms and its inverse on :class:`xarray.DataArray`.

Equivalent Sources
------------------

.. autosummary::
    :toctree: generated/

    EquivalentSources
    EquivalentSourcesGB
    EquivalentSourcesSph

Forward modelling
-----------------

.. autosummary::
    :toctree: generated/

    point_gravity
    prism_gravity
    tesseroid_gravity
    prism_layer
    tesseroid_layer
    DatasetAccessorPrismLayer
    DatasetAccessorTesseroidLayer

Isostatic Moho
--------------

.. autosummary::
    :toctree: generated/

    isostatic_moho_airy
    isostasy_airy (**DEPRECATED**)

.. warning::

    The :func:`harmonica.isostasy_airy` function will be deprecated in
    Harmonica v0.6. Please use :func:`harmonica.isostatic_moho_airy`
    instead.

Input and Output
----------------

.. autosummary::
   :toctree: generated/

    load_icgem_gdf
    load_oasis_montaj_grid

Visualization
-------------

.. autosummary::
   :toctree: generated/

    visualization.prism_to_pyvista

Synthetic models and surveys
----------------------------

.. warning::

    The :mod:`harmonica.synthetic` module will be deprecated in Harmonica
    v0.6.0

.. autosummary::
   :toctree: generated/

    synthetic.airborne_survey
    synthetic.ground_survey


.. automodule:: harmonica.datasets

.. currentmodule:: harmonica

Datasets
--------

.. warning::

    The :mod:`harmonica.datasets` module and every sample dataset a will be
    deprecated in Harmonica v0.6.0. The examples and the user guide will
    transition to using Ensaio (https://www.fatiando.org/ensaio/) instead.

.. warning::

    The :mod:`harmonica.datasets` module will be deprecated in Harmonica
    v0.6.0


.. autosummary::
   :toctree: generated/

    datasets.locate
    datasets.fetch_gravity_earth
    datasets.fetch_geoid_earth
    datasets.fetch_topography_earth
    datasets.fetch_britain_magnetic
    datasets.fetch_south_africa_gravity

Utilities
---------

.. autosummary::
   :toctree: generated/

    test
