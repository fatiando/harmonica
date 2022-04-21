.. _sample_data:

Sample Data
===========

Harmonica provides some sample data for testing through the
:mod:`harmonica.datasets` module.

.. warning::

    The :mod:`harmonica.datasets` module will be deprecated in Harmonica
    v0.6.0


Where is my data?
-----------------

The sample data files are downloaded automatically by :mod:`pooch` the first
time you load them. The files are saved to the default cache location on your
operating system. The location varies depending on your system and
configuration. We provide the :func:`harmonica.datasets.locate` function if you
need to find the data storage location on your system.

You can change the base data directory by setting the ``HARMONICA_DATA_DIR``
environment variable to a different path.


Available datasets
------------------

These are the datasets currently available:
