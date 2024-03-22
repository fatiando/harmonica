.. _install:

Installing
==========

There are different ways to install Harmonica:

.. tab-set::

    .. tab-item:: pip

        Using the `pip package manager <https://pypi.org/project/pip/>`__:

        .. code:: bash

            pip install harmonica

    .. tab-item:: conda/mamba

        Using the `conda package manager <https://conda.io/>`__ (or ``mamba``)
        that comes with the Anaconda/Miniconda distribution:

        .. code:: bash

            conda install harmonica --channel conda-forge

    .. tab-item:: Development version

        You can use ``pip`` to install the latest **unreleased** version from
        GitHub (**not recommended** in most situations):

        .. code:: bash

            python -m pip install --upgrade git+https://github.com/fatiando/harmonica

.. note::

   The commands above should be executed in a terminal. On Windows, use the
   ``cmd.exe`` or the "Anaconda Prompt" app if you’re using Anaconda.


Which Python?
-------------

You'll need **Python 3.8 or greater**.
See :ref:`python-versions` if you require support for older versions.

Dependencies
------------

The required dependencies should be installed automatically when you install
Harmonica using ``conda`` or ``pip``. Optional dependencies have to be
installed manually.

.. note::

    See :ref:`dependency-versions` for the our policy of oldest supported
    versions of each dependency.

Required:

* `numpy <http://www.numpy.org/>`__
* `pandas <http://pandas.pydata.org/>`__
* `numba <https://numba.pydata.org/>`__
* `scipy <https://www.scipy.org/>`__
* `xarray <https://xarray.pydata.org/>`__
* `scikit-learn <https://scikit-learn.org>`__
* `pooch <http://www.fatiando.org/pooch/>`__
* `verde <http://www.fatiando.org/verde/>`__
* `xrft <https://xrft.readthedocs.io/>`__

Optional:

* `pyvista <https://www.pyvista.org/>`__ and
  `vtk <https://vtk.org/>`__ (>= 9): for 3D visualizations.
  See :func:`harmonica.prism_to_pyvista`.
* `numba_progress <https://pypi.org/project/numba-progress/>`__ for
  printing a progress bar on some forward modelling computations.
  See :func:`harmonica.prism_gravity`.

The examples in the :ref:`gallery` also use:

* `boule <http://www.fatiando.org/boule/>`__
* `ensaio <http://www.fatiando.org/ensaio/>`__ for downloading sample datasets
* `pygmt <https://www.pygmt.org/>`__ for plotting maps
* `pyproj <https://jswhit.github.io/pyproj/>`__ for cartographic projections


Testing your install
--------------------

We ship a full test suite with the package.
To run the tests, you'll need to install some extra dependencies first:

* `pytest <https://docs.pytest.org/>`__
* `boule <http://www.fatiando.org/boule/>`__

After that, you can test your installation by running the following inside
a Python interpreter::

    import harmonica
    harmonica.test()
