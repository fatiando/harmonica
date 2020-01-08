.. _api:

API Reference
=============

.. automodule:: harmonica

.. currentmodule:: harmonica

Gravity Corrections
-------------------

.. autosummary::
    :toctree: generated/

    normal_gravity
    bouguer_correction

Equivalent Layers
--------------------------

.. autosummary::
    :toctree: generated/

    EQLHarmonic
    EQLHarmonicSpherical

Forward modelling
-----------------

.. autosummary::
    :toctree: generated/

    point_mass_gravity
    prism_gravity
    tesseroid_gravity

Isostasy
--------

.. autosummary::
    :toctree: generated/

    isostasy_airy

Reference Ellipsoids
--------------------

.. autosummary::
   :toctree: generated/

    ReferenceEllipsoid
    set_ellipsoid
    get_ellipsoid
    print_ellipsoids

Coordinates Conversions
-----------------------

.. autosummary::
   :toctree: generated/

    geodetic_to_spherical
    spherical_to_geodetic

Input and Output
----------------

.. autosummary::
   :toctree: generated/

    load_icgem_gdf

Synthetic models and surveys
----------------------------
.. autosummary::
   :toctree: generated/

    synthetic.airborne_survey
    synthetic.ground_survey


.. automodule:: harmonica.datasets

.. currentmodule:: harmonica

Datasets
--------

.. autosummary::
   :toctree: generated/

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
