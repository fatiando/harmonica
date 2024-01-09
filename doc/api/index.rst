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

Gravity fields:

.. autosummary::
    :toctree: generated/

    point_gravity
    prism_gravity
    tesseroid_gravity

Magnetic fields:

.. autosummary::
    :toctree: generated/

    dipole_magnetic
    dipole_magnetic_component
    prism_magnetic

Layers and meshes:

.. autosummary::
    :toctree: generated/

    prism_layer
    tesseroid_layer
    DatasetAccessorPrismLayer
    DatasetAccessorTesseroidLayer

Isostatic Moho
--------------

.. autosummary::
    :toctree: generated/

    isostatic_moho_airy

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

Utilities
---------

.. autosummary::
   :toctree: generated/

    magnetic_vec_to_angles
    magnetic_angles_to_vec
