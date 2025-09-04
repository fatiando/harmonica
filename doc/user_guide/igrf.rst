IGRF calculation
================

The International Geomagnetic Reference Field (IGRF) is time-variable spherical
harmonic model of the Earth's internal magnetic field [Alken2021]_ [IAGA2024]_.
The model is released every 5 years and allows us to calculate the internal
magnetic field from 1900 until 5 years after the latest release (based on
predictions of the secular variation). Harmonica allows calculating the 14th
generation IGRF field with :class:`harmonica.IGRF14`. Here's how it works.

.. jupyter-execute::

   import datetime
   import pygmt
   import harmonica as hm

All of the functionality is wrapped in the :class:`~harmonica.IGRF14` class.
When creating an instance of it, we need to provide the date on which we want
to calculate the field:

.. jupyter-execute::

   igrf = hm.IGRF14("1954-07-29")

The date can be provided as an `ISO 8601 formatted date
<https://en.wikipedia.org/wiki/ISO_8601>`__ string like above or as a Python
:class:`datetime.datetime`:

.. jupyter-execute::

   igrf = hm.IGRF14(datetime.datetime(1954, 7, 29, hour=1, minute=20))

.. tip::

   If the time is omited, the default is midnight. If a timezone is omited, the
   default is UTC.

Calculating at given points
---------------------------

To calculate the IGRF field at a particular point or set of points, we can use
the :meth:`harmonica.IGRF14.predict` method. For example, let's calculate the
field on the date above at the `Universidade de São Paulo
<https://www5.usp.br/>`__ campus. To do so, we need to provide the longitude,
latitude, and geometric height (in meters) of the calculation point:

.. jupyter-execute::

   field = igrf.predict((-46.73441817537987, -23.559276852800025, 700))
   print(" | ".join([f"B{c}={v:.1f} nT" for c, v in zip("enu", field)]))
