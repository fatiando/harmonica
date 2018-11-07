.. _sample_data:

Sample Data
===========

Harmonica provides some sample data for testing through the :mod:`harmonica.datasets`
module. The sample data are automatically downloaded from the `Github repository
<https://github.com/fatiando/harmonica>`__ to a folder on your computer the first time
you use them. After that, the data are loaded from this folder. The download is managed
by the :mod:`pooch` package.


Where is my data?
-----------------

The data files are downloaded to a folder ``~/.harmonica/data/`` by default. This is the
*base data directory*. :mod:`pooch` will create a separate folder in the base directory
for each version of Harmonica. For example, the base data dir for v0.1.0 is
``~/.harmonica/data/v0.1.0``. If you're using the latest development version from
Github, the version is ``master``.

You can change the base data directory by setting the ``HARMONICA_DATA_DIR`` environment
variable to a different path.


Available datasets
------------------

These are the datasets currently available:
