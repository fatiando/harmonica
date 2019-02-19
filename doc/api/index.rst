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


.. automodule:: harmonica.datasets

.. currentmodule:: harmonica

Datasets
--------

.. autosummary::
   :toctree: generated/

    datasets.fetch_gravity_earth
    datasets.fetch_geoid_earth
    datasets.fetch_topography_earth
    datasets.fetch_rio_magnetic

Utilities
---------

.. autosummary::
   :toctree: generated/

    test
