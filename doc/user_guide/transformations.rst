.. _transformations:

Grid transformations
====================

Harmonica offers some functions to apply FFT-based (Fast Fourier Transform) and
finite-differences transformations to regular grids of gravity and magnetic
fields located at a constant height.

In order to apply these grid transformations, we first need a **regular grid in
Cartesians coordinates**.
Let's download a magnetic anomaly grid over the Lightning Creek Sill Complex,
Australia, readily available in :mod:`ensaio`.
We can load the data file using :mod:`xarray`:

.. jupyter-execute::

    import ensaio
    import xarray as xr

    fname = ensaio.fetch_lightning_creek_magnetic(version=1)
    magnetic_grid = xr.load_dataarray(fname)
    magnetic_grid

And plot it:

.. jupyter-execute::
   :hide-code:

    import pygmt

    # Needed so that displaying works on jupyter-sphinx and sphinx-gallery at
    # the same time. Using PYGMT_USE_EXTERNAL_DISPLAY="false" in the Makefile
    # for sphinx-gallery to work means that fig.show won't display anything here
    # either.
    pygmt.set_display(method="notebook")


.. jupyter-execute::

    import pygmt
    import verde as vd

    fig = pygmt.Figure()

    maxabs = vd.maxabs(magnetic_grid) * .6
    pygmt.makecpt(cmap="balance+h0", series=[-maxabs, maxabs], background=True)

    fig.grdimage(
        magnetic_grid,
        projection="X15c",
        cmap=True,
        frame=["af", "WeSn+tMagnetic anomaly"]
    )
    fig.colorbar(
        position="JCB+e",
        frame=["af", 'x+lnT'],
    )

    fig.show()

.. seealso::

   In case we have a regular grid defined in geographic coordinates (longitude,
   latitude) we can project them to Cartesian coordinates using the
   :func:`verde.project_grid` function and a map projection like the ones
   available in :mod:`pyproj`.

Since all the grid transformations we are going to apply are based on FFT
methods, we usually want to pad them in order their increase the accuracy.
We can easily do it through the :func:`xrft.pad` function.
First we need to define how much padding we want to add along each direction.
We will add one third of the width and height of the grid to each side:

.. jupyter-execute::

    pad_width = {
        "easting": magnetic_grid.easting.size // 3,
        "northing": magnetic_grid.northing.size // 3,
    }

And then we can pad it, but dropping the ``height`` coordinate first (this is
needed by the :func:`xrft.pad` function):

.. jupyter-execute::

    import xrft

    magnetic_grid_no_height = magnetic_grid.drop_vars("height")
    magnetic_grid_padded = xrft.pad(magnetic_grid_no_height, pad_width)
    magnetic_grid_padded

.. jupyter-execute::

    fig = pygmt.Figure()

    maxabs = vd.maxabs(magnetic_grid_padded) * .6
    pygmt.makecpt(cmap="balance+h0", series=[-maxabs, maxabs], background=True)

    fig.grdimage(
        magnetic_grid_padded,
        projection="X15c",
        cmap=True,
        frame=["af", "WeSn+tPadded magnetic anomaly"]
    )
    fig.colorbar(
        position="JCB+e",
        frame=["af", 'x+lnT'],
    )

    fig.show()

Now that we have the padded grid, we can apply any grid transformation.


Upward derivative
-----------------

Let's calculate the upward derivative (a.k.a. vertical derivative) of the
magnetic anomaly grid using the :func:`harmonica.derivative_upward` function:

.. jupyter-execute::

    import harmonica as hm

    deriv_upward = hm.derivative_upward(magnetic_grid_padded)
    deriv_upward

This grid includes all the padding we added to the original magnetic grid, so
we better unpad it using :func:`xrft.unpad`:

.. jupyter-execute::

    deriv_upward = xrft.unpad(deriv_upward, pad_width)
    deriv_upward

And plot it:

.. jupyter-execute::

    fig = pygmt.Figure()

    maxabs = vd.maxabs(deriv_upward) * .5
    pygmt.makecpt(cmap="balance+h0", series=[-maxabs, maxabs], background=True)

    fig.grdimage(
        deriv_upward,
        projection="X15c",
        cmap=True,
        frame=["af", "WeSn+tUpward derivative of the magnetic anomaly"]
    )
    fig.colorbar(
        position="JCB+e",
        frame=["af", 'x+lnT/m'],
    )

    fig.show()


Horizontal derivatives
----------------------

We can also compute horizontal derivatives over a regular grid using the
:func:`harmonica.derivative_easting` and :func:`harmonica.derivative_northing`
functions.

.. jupyter-execute::

    deriv_easting = hm.derivative_easting(magnetic_grid)
    deriv_easting

.. jupyter-execute::

    deriv_northing = hm.derivative_northing(magnetic_grid)
    deriv_northing

And plot them:

.. jupyter-execute::

    fig = pygmt.Figure()

    maxabs = vd.maxabs(deriv_easting, deriv_northing) * .4
    pygmt.makecpt(cmap="balance+h0", series=[-maxabs, maxabs], background=True)

    fig.grdimage(
        deriv_easting,
        projection="X15c",
        cmap=True,
        frame=["af", "WeSn+tEasting derivative of the magnetic anomaly"]
    )

    fig.shift_origin(xshift="16c")

    pygmt.makecpt(cmap="balance+h0", series=[-maxabs, maxabs], background=True)

    fig.grdimage(
        deriv_northing,
        projection="X15c",
        cmap=True,
        frame=["af", "wESn+tNorthing derivative of the magnetic anomaly"]
    )

    fig.colorbar(
        position="JBC+h+e+o-8c/1c+w15c/.8c",
        frame=["af", 'x+lnT/m'],
    )

    fig.show()

By default, these two functions compute the horizontal derivatives using
central finite differences methods. We can choose to use either the finite
difference or the FFT-based method through the ``method`` argument.

For example, we can pass ``method="fft"`` to compute the derivatives in the
frequency domain:

.. jupyter-execute::

    deriv_easting = hm.derivative_easting(magnetic_grid_padded, method="fft")
    deriv_easting = xrft.unpad(deriv_easting, pad_width)
    deriv_easting

.. jupyter-execute::

    deriv_northing = hm.derivative_northing(magnetic_grid_padded, method="fft")
    deriv_northing = xrft.unpad(deriv_northing, pad_width)
    deriv_northing

.. jupyter-execute::

    fig = pygmt.Figure()

    maxabs = vd.maxabs(deriv_easting, deriv_northing) * .4
    pygmt.makecpt(cmap="balance+h0", series=[-maxabs, maxabs], background=True)

    fig.grdimage(
        deriv_easting,
        projection="X15c",
        cmap=True,
        frame=["af", "WeSn+tEasting derivative of the magnetic anomaly"]
    )

    fig.shift_origin(xshift="16c")

    pygmt.makecpt(cmap="balance+h0", series=[-maxabs, maxabs], background=True)

    fig.grdimage(
        deriv_northing,
        projection="X15c",
        cmap=True,
        frame=["af", "wESn+tNorthing derivative of the magnetic anomaly"]
    )

    fig.colorbar(
        position="JBC+h+e+o-8c/1c+w15c/.8c",
        frame=["af", 'x+lnT/m'],
    )

    fig.show()

.. important::

    Horizontal derivatives through finite differences are usually more accurate
    and have less artifacts than their FFT-based counterpart.



Upward continuation
-------------------

We can also upward continue the original magnetic grid.
This is, estimating the magnetic field generated by the same sources at
a higher altitude.
The original magnetic anomaly grid is located at 500 m above the ellipsoid, as
we can see in its `height` coordinate.
If we want to get the magnetic anomaly at 1000m above the ellipsoid, we need
to upward continue it a height displacement of 500m:

.. jupyter-execute::

    upward_continued = hm.upward_continuation(
        magnetic_grid_padded, height_displacement=500
    )

This grid includes all the padding we added to the original magnetic grid, so
we better unpad it using :func:`xrft.unpad`:

.. jupyter-execute::

    upward_continued = xrft.unpad(upward_continued, pad_width)
    upward_continued

And plot it:

.. jupyter-execute::

    fig = pygmt.Figure()

    maxabs = vd.maxabs(upward_continued)
    pygmt.makecpt(cmap="balance+h0", series=[-maxabs, maxabs], background=True)

    fig.grdimage(
        upward_continued,
        projection="X15c",
        cmap=True,
        frame=["af", "WeSn+tUpward continued magnetic anomaly at 1000m"]
    )
    fig.colorbar(
        position="JCB",
        frame=["af", 'x+lnT'],
    )

    fig.show()


Reduction to the pole
---------------------

We can also apply a reduction to the pole to any magnetic anomaly grid.
This transformation consists in obtaining the magnetic anomaly of the same
sources as if they were located on the North magnetic pole.
We can apply it through the :func:`harmonica.reduction_to_pole` function.

.. important::

   Applying reduction to the pole to low latitude regions can amplify high
   frequency noise.

The reduction to the pole needs information about the orientation of the
geomagnetic field at the location of the survey and also the orientation of the
magnetization vector of the sources.

The International Global Reference Field (IGRF) can provide us information
about the inclination and declination of the geomagnetic field at the time of
the survey (1990 in this case):

.. jupyter-execute::

    inclination, declination = -52.98, 6.51

If we consider that the sources are magnetized in the same direction as the
geomagnetic survey (hypothesis that is true in case the sources don't have any
remanence), then we can apply the reduction to the pole passing only the
``inclination`` and ``declination`` of the geomagnetic field:

.. jupyter-execute::

    rtp_grid = hm.reduction_to_pole(
        magnetic_grid_padded, inclination=inclination, declination=declination
    )

    # Unpad the reduced to the pole grid
    rtp_grid = xrft.unpad(rtp_grid, pad_width)
    rtp_grid

And plot it:

.. jupyter-execute::

    fig = pygmt.Figure()

    maxabs = vd.maxabs(rtp_grid) * .8
    pygmt.makecpt(cmap="balance+h0", series=[-maxabs, maxabs], background=True)

    fig.grdimage(
        rtp_grid,
        projection="X15c",
        cmap=True,
        frame=["af", "WeSn+tReduced to the pole magnetic anomaly"]
    )
    fig.colorbar(
        position="JCB+e",
        frame=["af", 'x+lnT'],
    )

    fig.show()

If on the other hand we have any knowledge about the orientation of the
magnetization vector of the sources, we can specify the
``magnetization_inclination`` and ``magnetization_declination``:

.. jupyter-execute::

    mag_inclination, mag_declination = -25, 21

    tmp = rtp_grid = hm.reduction_to_pole(
        magnetic_grid_padded,
        inclination=inclination,
        declination=declination,
        magnetization_inclination=mag_inclination,
        magnetization_declination=mag_declination,
    )

    # Unpad the reduced to the pole grid
    rtp_grid = xrft.unpad(rtp_grid, pad_width)
    rtp_grid

.. jupyter-execute::

    fig = pygmt.Figure()

    maxabs = vd.maxabs(rtp_grid) * .8
    pygmt.makecpt(cmap="balance+h0", series=[-maxabs, maxabs], background=True)

    fig.grdimage(
        rtp_grid,
        projection="X15c",
        cmap=True,
        frame=["af", "WeSn+tReduced to the pole with remanence"]
    )
    fig.colorbar(
        position="JCB+e",
        frame=["af", 'x+lnT'],
    )

    fig.show()


Gaussian filters
-----------------

We can also apply Gaussian low-pass and high-pass filters to any regular grid.
These two need us to select a cutoff wavelength.
The low-pass filter will remove any signal with a high spatial frequency,
keeping only the signal components that have a wavelength higher than the
selected cutoff wavelength.
The high-pass filter, on the other hand, removes any signal with a low spatial
frequency, keeping only the components with a wavelength lower than the cutoff
wavelength.
These two filters can be applied to our regular grid with the
:func:`harmonica.gaussian_lowpass` and :func:`harmonica.gaussian_highpass`.

Let's define a cutoff wavelength of 5 kilometers:

.. jupyter-execute::

    cutoff_wavelength = 5e3

Then apply the two filters to our padded magnetic grid:

.. jupyter-execute::

    magnetic_low_freqs = hm.gaussian_lowpass(
        magnetic_grid_padded, wavelength=cutoff_wavelength
    )
    magnetic_high_freqs = hm.gaussian_highpass(
        magnetic_grid_padded, wavelength=cutoff_wavelength
    )

And unpad them:

.. jupyter-execute::

    magnetic_low_freqs = xrft.unpad(magnetic_low_freqs, pad_width)
    magnetic_high_freqs = xrft.unpad(magnetic_high_freqs, pad_width)

.. jupyter-execute::

    magnetic_low_freqs

.. jupyter-execute::

    magnetic_high_freqs

Let's plot the results side by side:

.. jupyter-execute::

    fig = pygmt.Figure()

    maxabs = vd.maxabs(magnetic_low_freqs, magnetic_high_freqs) * .6
    pygmt.makecpt(cmap="balance+h0", series=[-maxabs, maxabs], background=True)

    fig.grdimage(
        magnetic_low_freqs,
        projection="X15c",
        cmap=True,
        frame=["af", "WeSn+tLow-pass filtered magnetic anomaly"]
    )

    fig.shift_origin(xshift="16c")

    pygmt.makecpt(cmap="balance+h0", series=[-maxabs, maxabs], background=True)

    fig.grdimage(
        magnetic_high_freqs,
        projection="X15c",
        cmap=True,
        frame=["af", "wESn+tHigh-pass filtered magnetic anomaly"]
    )

    fig.colorbar(
        position="JBC+h+e+o-8c/1c+w15c/.8c",
        frame=["af", 'x+lnT'],
    )

    fig.show()


Total gradient amplitude
------------------------

.. hint::

    Total gradient amplitude is also known as *analytic signal*.

We can also calculate the total gradient amplitude of any magnetic anomaly grid.
This transformation consists in obtaining the amplitude of the gradient of the
magnetic field in all the three spatial directions by applying

.. math::

   A(x, y) = \sqrt{
     \left( \frac{\partial M}{\partial x} \right)^2
     + \left( \frac{\partial M}{\partial y} \right)^2
     + \left( \frac{\partial M}{\partial z} \right)^2
   }.


We can apply it through the :func:`harmonica.total_gradient_amplitude` function.

.. jupyter-execute::

    tga_grid = hm.total_gradient_amplitude(
        magnetic_grid_padded
    )

    # Unpad the total gradient amplitude grid
    tga_grid = xrft.unpad(tga_grid, pad_width)
    tga_grid

And plot it:

.. jupyter-execute::

    fig = pygmt.Figure()

    maxabs = vd.maxabs(tga_grid)
    pygmt.makecpt(cmap="viridis", series=[0, maxabs], background=True)

    fig.grdimage(
        tga_grid,
        projection="X15c",
        cmap=True,
        frame=["af", "WeSn+tTotal gradient amplitude of the magnetic anomaly"]
    )
    fig.colorbar(
        position="JCB",
        frame=["af", 'x+lnT/m'],
    )

    fig.show()

----

.. grid:: 2

    .. grid-item-card:: :jupyter-download-script:`Download Python script <transformations>`
        :text-align: center

    .. grid-item-card:: :jupyter-download-nb:`Download Jupyter notebook <transformations>`
        :text-align: center
