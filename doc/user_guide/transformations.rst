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

    import matplotlib.pyplot as plt

    tmp = magnetic_grid.plot(cmap="seismic", center=0, add_colorbar=False)
    plt.gca().set_aspect("equal")
    plt.title("Magnetic anomaly grid")
    plt.gca().ticklabel_format(style="sci", scilimits=(0, 0))
    plt.colorbar(tmp, label="nT")
    plt.show()

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

    tmp = magnetic_grid_padded.plot(cmap="seismic", center=0, add_colorbar=False)
    plt.gca().set_aspect("equal")
    plt.title("Padded magnetic anomaly grid")
    plt.gca().ticklabel_format(style="sci", scilimits=(0, 0))
    plt.colorbar(tmp, label="nT")
    plt.show()

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

    tmp = deriv_upward.plot(cmap="seismic", center=0, add_colorbar=False)
    plt.gca().set_aspect("equal")
    plt.title("Upward derivative of the magnetic anomaly")
    plt.gca().ticklabel_format(style="sci", scilimits=(0, 0))
    plt.colorbar(tmp, label="nT/m")
    plt.show()


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

    fig, (ax1, ax2) = plt.subplots(
        nrows=1, ncols=2, sharey=True, figsize=(12, 8)
    )

    cbar_kwargs=dict(
        label="nT/m", orientation="horizontal", shrink=0.8, pad=0.08, aspect=42
    )
    kwargs = dict(center=0, cmap="seismic", cbar_kwargs=cbar_kwargs)

    tmp = deriv_easting.plot(ax=ax1, **kwargs)
    tmp = deriv_northing.plot(ax=ax2, **kwargs)

    ax1.set_title("Easting derivative of the magnetic anomaly")
    ax2.set_title("Northing derivative of the magnetic anomaly")
    for ax in (ax1, ax2):
        ax.set_aspect("equal")
        ax.ticklabel_format(style="sci", scilimits=(0, 0))
    plt.show()

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

    fig, (ax1, ax2) = plt.subplots(
        nrows=1, ncols=2, sharey=True, figsize=(12, 8)
    )

    cbar_kwargs=dict(
        label="nT/m", orientation="horizontal", shrink=0.8, pad=0.08, aspect=42
    )
    kwargs = dict(center=0, cmap="seismic", cbar_kwargs=cbar_kwargs)

    tmp = deriv_easting.plot(ax=ax1, **kwargs)
    tmp = deriv_northing.plot(ax=ax2, **kwargs)

    ax1.set_title("Easting derivative of the magnetic anomaly")
    ax2.set_title("Northing derivative of the magnetic anomaly")
    for ax in (ax1, ax2):
        ax.set_aspect("equal")
        ax.ticklabel_format(style="sci", scilimits=(0, 0))
    plt.show()


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

    tmp = upward_continued.plot(cmap="seismic", center=0, add_colorbar=False)
    plt.gca().set_aspect("equal")
    plt.title("Upward continued magnetic anomaly to 1000m")
    plt.gca().ticklabel_format(style="sci", scilimits=(0, 0))
    plt.colorbar(tmp, label="nT")
    plt.show()


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

    tmp = rtp_grid.plot(cmap="seismic", center=0, add_colorbar=False)
    plt.gca().set_aspect("equal")
    plt.title("Magnetic anomaly reduced to the pole")
    plt.gca().ticklabel_format(style="sci", scilimits=(0, 0))
    plt.colorbar(tmp, label="nT")
    plt.show()

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

    tmp = rtp_grid.plot(cmap="seismic", center=0, add_colorbar=False)
    plt.gca().set_aspect("equal")
    plt.title("Reduced to the pole with remanence")
    plt.gca().ticklabel_format(style="sci", scilimits=(0, 0))
    plt.colorbar(tmp, label="nT")
    plt.show()


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

    import verde as vd

    fig, (ax1, ax2) = plt.subplots(
        nrows=1, ncols=2, sharey=True, figsize=(12, 8)
    )

    maxabs = vd.maxabs(magnetic_low_freqs, magnetic_high_freqs)
    kwargs = dict(cmap="seismic", vmin=-maxabs, vmax=maxabs, add_colorbar=False)

    tmp = magnetic_low_freqs.plot(ax=ax1, **kwargs)
    tmp = magnetic_high_freqs.plot(ax=ax2, **kwargs)

    ax1.set_title("Magnetic anomaly after low-pass filter")
    ax2.set_title("Magnetic anomaly after high-pass filter")
    for ax in (ax1, ax2):
        ax.set_aspect("equal")
        ax.ticklabel_format(style="sci", scilimits=(0, 0))

    plt.colorbar(
        tmp,
        ax=[ax1, ax2],
        label="nT",
        orientation="horizontal",
        aspect=42,
        shrink=0.8,
        pad=0.08,
    )
    plt.show()

----

.. grid:: 2

    .. grid-item-card:: :jupyter-download-script:`Download Python script <transformations>`
        :text-align: center

    .. grid-item-card:: :jupyter-download-nb:`Download Jupyter notebook <transformations>`
        :text-align: center
