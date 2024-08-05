Overview
========

Harmonica provides functions and classes for processing, modelling and
interpolating gravity and magnetic data.

Its main goals are:

- Provide efficient, well designed, and fully tested code that would compress
  the building blocks for more complex workflows.
- Cover the entire data life-cycle: from raw data to 3D Earth model.
- Focus on best-practices to discourage misuse of methods.
- Easily extendable code to enable research on the developments of new methods.

Harmonica *will not* provide:

- Multi-physics partial differential equation solvers. Use
  `SimPEG <http://www.simpeg.xyz/>`__ or `PyGIMLi <https://www.pygimli.org/>`__
  instead.
- Generic processing methods like grid transformations (use `Verde
  <https://www.fatiando.org/verde>`__ or `Xarray <https://docs.xarray.dev>`__
  instead) or multidimensional FFT calculations (use `xrft
  <https://xrft.readthedocs.io>`__ instead).
- Reference ellipsoid representations and computations like normal gravity. Use
  `Boule <https://www.fatiando.org/boule>`__ instead.
- Data visualization functions. Use `matplotlib <https://matplotlib.org/>`__
  for generic plots, `Xarray <https://docs.xarray.dev>`__ for plotting grids,
  `PyGMT <https://www.pygmt.org>`__ for maps, and `PyVista
  <https://www.pyvista.org/>`__ for 3D visualizations.
- GUI applications.


Conventions
-----------

Before we get started, here are a few conventions we keep across Harmonica:

- Every physical quantity will be assumed to be given in a unit belonging to the `International System of Units (SI) <https://en.wikipedia.org/wiki/International_System_of_Units>`__. The only exceptions are:

  - **gravity accelerations** are expected in `miligal (mGal) <https://en.wikipedia.org/wiki/Gal_(unit)>`__ (:math:`1~\text{mGal} = 10^{-5}~\text{m}/\text{s}^2`).

  - **gravity tensor components** are assumed to be in `Eotvos <https://en.wikipedia.org/wiki/Eotvos_(unit)>`__ (:math:`1~\text{Eotvos} = 10^{-9}~\text{s}^{-2}`).

  - **magnetic fields** are given in nano Tesla (nT).

- Harmonica uses the same conventions as :mod:`verde`, meaning:

  - Functions expect coordinates in the order: West-East, South-North and (in occasions) Bottom-Top. Exceptions to this rule are the ``dims`` and ``shape`` arguments.

  - We avoid using names like "x", "y" and "z" to avoid ambiguity. We use
    "easting", "northing" and "upward" or "longitude", "latitude" and "height"
    instead.

- Some functions or classes expect its arguments to be defined in a specific
  coordinate system. They can either be in:

  - **Cartesian coordinates:** usually given as *easting*, *northing* and *upward* coordinates (in meters), where the vertical axis points upwards.

  - **Geodetic or ellipsoidal coordinates:** given as *longitude*, *latitude* (both in decimal degrees) and *geodetic height* (in meters).

  - **Spherical geocentric coordinates:** given as *longitude*, *spherical latitude* (both in decimal degrees) and *radius* (in meters).

.. seealso::

    Checkout the :ref:`coordinate_systems` section for more details on these
    coordinates systems.


The Library
-----------

Most classes and functions are available through the :mod:`harmonica` top level
package. Througout the documentation, we'll use ``hm`` as an alias for
:mod:`harmonica`.

.. code::

    import harmonica as hm

.. seealso::

    Checkout the :ref:`api` for a comprehensive list of the available function
    and classes in Harmonica.
