.. _install:

Installing
==========

There are different ways to install Harmonica:

.. tab-set::

    .. tab-item:: conda/mamba

        Using the `conda <https://conda.io/>`__ package manager (or ``mamba``)
        that comes with the Anaconda, Miniconda, or Miniforge distributions:

        .. code:: bash

            conda install harmonica --channel conda-forge

    .. tab-item:: pip

        Using the `pip <https://pypi.org/project/pip/>`__ package manager:

        .. code:: bash

            python -m pip install harmonica

    .. tab-item:: Development version

        You can use ``pip`` to install the latest **unreleased** version from
        GitHub (**not recommended** in most situations):

        .. code:: bash

            python -m pip install --upgrade git+https://github.com/fatiando/harmonica

.. tip::

    The commands above should be executed in a terminal. On Windows, use the
    ``cmd.exe`` or the "Anaconda Prompt" / "Miniforge Prompt" app if you're using
    Anaconda / Miniforge.

.. admonition:: Which Python?
    :class: tip

    See :ref:`python-versions` for a list of supported Python versions.

.. note::

   We recommend using the
   `Miniforge distribution <https://conda-forge.org/download/>`__
   to ensure that you have the ``conda`` package manager available.
   Installing Miniforge does not require administrative rights to your computer
   and doesn't interfere with any other Python installations in your system.
   It's also much smaller than the Anaconda distribution and is less likely to
   break when installing new software.


.. _dependencies:

Dependencies
------------

The required dependencies should be installed automatically when you install
Harmonica using ``conda`` or ``pip``. Optional dependencies have to be
installed manually.

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

.. note::

    See :ref:`dependency-versions` for our policy of oldest supported
    versions of each dependency.

The examples in the :ref:`gallery` also use:

* `boule <http://www.fatiando.org/boule/>`__
* `ensaio <http://www.fatiando.org/ensaio/>`__ for downloading sample datasets
* `pygmt <https://www.pygmt.org/>`__ for plotting maps
* `pyproj <https://jswhit.github.io/pyproj/>`__ for cartographic projections
