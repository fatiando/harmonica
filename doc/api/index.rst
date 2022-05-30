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

Apply well known transformations to regular grid of potential fields.

.. autosummary::
    :toctree: generated/

    derivative_upward

Frequency domain filters
------------------------

Define filters in the frequency domain.

.. autosummary::
    :toctree: generated/

    filters.derivative_upward_kernel

Use :mod:`xrft` to apply Fast-Fourier Transforms on :mod:`xarray.DataArray`s.

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
    DatasetAccessorPrismLayer

Isostasy
--------

.. autosummary::
    :toctree: generated/

    isostasy_airy

Input and Output
----------------

.. autosummary::
   :toctree: generated/

    load_icgem_gdf

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
