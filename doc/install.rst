.. _install:

Installing
==========

Which Python?
-------------

You'll need **Python 3.8 or greater**.
See :ref:`python-versions` if you require support for older versions.

We recommend using the
`Anaconda Python distribution <https://www.anaconda.com/download>`__
to ensure you have all dependencies installed and the ``conda`` package manager
available.
Installing Anaconda does not require administrative rights to your computer and
doesn't interfere with any other Python installations in your system.


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



Installing with conda
---------------------

You can install Harmonica using the `conda package manager
<https://conda.io/>`__ that comes with the Anaconda distribution::

    conda install harmonica --channel conda-forge


Installing with pip
-------------------

Alternatively, you can also use the `pip package manager
<https://pypi.org/project/pip/>`__::

    pip install harmonica


Installing the latest development version
-----------------------------------------

You can use ``pip`` to install the latest source from Github::

    pip install git+https://github.com/fatiando/harmonica

Alternatively, you can clone the git repository locally and install from
there::

    git clone https://github.com/fatiando/harmonica.git
    cd harmonica
    pip install .


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
