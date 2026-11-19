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

    import xarray as xr
    import matplotlib.pyplot as plt
    import pyproj
    import verde as vd
    import harmonica as hm
    import ensaio

    fname = ensaio.fetch_lightning_creek_magnetic(version=1)
    magnetic_grid = xr.load_dataarray(fname)
    magnetic_grid

And plot it:

.. jupyter-execute::

    maxabs = vd.maxabs(magnetic_grid, percentile=99.9)
    tmp = magnetic_grid.plot(
        cmap="RdBu_r",
        vmin=-maxabs,
        vmax=maxabs,
        cbar_kwargs={"label":"nT"},
    )
    plt.gca().set_aspect("equal")
    plt.title("Magnetic anomaly grid")
    plt.gca().ticklabel_format(style="sci", scilimits=(0, 0))
    plt.show()

.. seealso::

   In case we have a regular grid defined in geographic coordinates (longitude,
   latitude) we can project them to Cartesian coordinates using the
   :func:`verde.project_grid` function and a map projection like the ones
   available in :mod:`pyproj`.


Upward derivative
-----------------

Let's calculate the upward derivative (a.k.a. vertical derivative) of the
magnetic anomaly grid using the :func:`harmonica.derivative_upward` function:

.. jupyter-execute::

    deriv_upward = hm.derivative_upward(magnetic_grid)
    deriv_upward

And plot it:

.. jupyter-execute::

    maxabs = vd.maxabs(deriv_upward, percentile=99.9)
    tmp = deriv_upward.plot(
        cmap="RdBu_r",
        vmin=-maxabs,
        vmax=maxabs,
        cbar_kwargs={"label":"nT/m"},
    )
    plt.gca().set_aspect("equal")
    plt.title("Magnetic anomaly grid")
    plt.gca().ticklabel_format(style="sci", scilimits=(0, 0))
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
    maxabs = vd.maxabs(deriv_easting, deriv_northing, percentile=99)


    tmp = deriv_easting.plot(
        ax=ax1,
        add_colorbar=False,
        vmin=-maxabs,
        vmax=maxabs,
        cmap='RdBu_r',
    )

    tmp = deriv_northing.plot(
        ax=ax2,
        add_colorbar=False,
        vmin=-maxabs,
        vmax=maxabs,
        cmap='RdBu_r',
    )

    ax1.set_aspect("equal")
    ax2.set_aspect("equal")
    ax1.set_title("Easting derivative of the magnetic anomaly")
    ax2.set_title("Northing derivative of the magnetic anomaly")
    ax1.ticklabel_format(style="sci", scilimits=(0, 0))
    ax2.ticklabel_format(style="sci", scilimits=(0, 0))

    fig.colorbar(tmp, ax=(ax1, ax2), orientation="horizontal", label="nT/m", fraction=.03, pad=0.08)
    plt.show()

By default, these two functions compute the horizontal derivatives using
central finite differences methods. We can choose to use either the finite
difference or the FFT-based method through the ``method`` argument.

For example, we can pass ``method="fft"`` to compute the derivatives in the
frequency domain:

.. jupyter-execute::

    deriv_easting = hm.derivative_easting(magnetic_grid, method="fft")
    deriv_easting

.. jupyter-execute::

    deriv_northing = hm.derivative_northing(magnetic_grid, method="fft")
    deriv_northing

.. jupyter-execute::

    fig, (ax1, ax2) = plt.subplots(
        nrows=1, ncols=2, sharey=True, figsize=(12, 8)
    )
    maxabs = vd.maxabs(deriv_easting, deriv_northing, percentile=99)


    tmp = deriv_easting.plot(
        ax=ax1,
        add_colorbar=False,
        vmin=-maxabs,
        vmax=maxabs,
        cmap='RdBu_r',
    )

    tmp = deriv_northing.plot(
        ax=ax2,
        add_colorbar=False,
        vmin=-maxabs,
        vmax=maxabs,
        cmap='RdBu_r',
    )

    ax1.set_aspect("equal")
    ax2.set_aspect("equal")
    ax1.set_title("Easting derivative of the magnetic anomaly")
    ax2.set_title("Northing derivative of the magnetic anomaly")
    ax1.ticklabel_format(style="sci", scilimits=(0, 0))
    ax2.ticklabel_format(style="sci", scilimits=(0, 0))

    fig.colorbar(tmp, ax=(ax1, ax2), orientation="horizontal", label="nT/m", fraction=.03, pad=0.08)
    plt.show()

.. important::

    Horizontal derivatives through finite differences are usually more accurate
    and have less artifacts than their FFT-based counterpart.


Upward continuation
-------------------

We can also upward continue the original magnetic grid using :func:`harmonica.upward_continuation`.
This is, estimating the magnetic field generated by the same sources at
a higher altitude.
The original magnetic anomaly grid is located at 500 m above the ellipsoid, as
we can see in its `height` coordinate.
If we want to get the magnetic anomaly at 1000m above the ellipsoid, we need
to upward continue it a height displacement of 500m:

.. jupyter-execute::

    change_in_height = 500  # meters
    upward_continued = hm.upward_continuation(
        magnetic_grid, height_displacement=change_in_height
    )
    upward_continued

Did you notice that the ``height`` coordinate is gone from the
upward-continued grid? We drop any non-dimensional coordinates when
doing upward continuation because we don't know the name of the
vertical coordinate and any values there would be wrong after continuation.
If we want it back, we need to assign an updated version of it:

.. jupyter-execute::

    upward_continued = upward_continued.assign_coords(
        {"height": magnetic_grid.height + change_in_height}
    )
    upward_continued

Now we can plot it:

.. jupyter-execute::

    tmp = upward_continued.plot(cmap="RdBu_r", center=0, cbar_kwargs={'label':'nT'})
    plt.gca().set_aspect("equal")
    plt.title("Upward continued magnetic anomaly to 1000m")
    plt.gca().ticklabel_format(style="sci", scilimits=(0, 0))
    plt.show()


Reduction to the pole
---------------------

We can also apply a reduction to the pole to any magnetic anomaly grid.
This transformation consists in obtaining the magnetic anomaly of the same
sources as if they were located on the North magnetic pole.
We can apply it through the :func:`harmonica.reduction_to_pole` function.

.. important::

    Reduction to the pole is unstable at low latitude regions and will result
    in artifacts.

The reduction to the pole needs information about the orientation of the
geomagnetic field at the location of the survey and also the orientation of the
magnetization vector of the sources.
The International Global Reference Field (IGRF) can provide us information
about the inclination and declination of the geomagnetic field at the time of
the survey. We don't have exact dates for the survey, but we know it was
some time between 1985 and 1999. We'll use July 1992 as a midpoint value.
It shouldn't matter too much since the secular variation is small.
We can use the :class:`harmonica.IGRF14` class to calculate the field values
at the center of the survey:

.. jupyter-execute::

    igrf = hm.IGRF14("1992-07-01")
    projection = pyproj.Proj(magnetic_grid.attrs["crs"])
    longitude, latitude = projection(
        magnetic_grid.easting.mean(),
        magnetic_grid.northing.mean(),
        inverse=True,
    )
    igrf_field = igrf.predict((longitude, latitude, magnetic_grid.height.mean()))
    intensity, inclination, declination = hm.magnetic_vec_to_angles(
        *igrf_field
    )
    print(inclination, declination)

If we consider that the sources are magnetized in the same direction as the
geomagnetic survey (hypothesis that is true in case the sources don't have any
remanence), then we can apply the reduction to the pole passing the same
``inclination`` and ``declination`` for both the geomagnetic field and the
magnetization:

.. jupyter-execute::

    rtp_grid = hm.reduction_to_pole(
        magnetic_grid,
        inclination=inclination,
        declination=declination,
        magnetization_inclination=inclination,
        magnetization_declination=declination,
    )
    rtp_grid

And plot it:

.. jupyter-execute::

    fig, (ax1, ax2) = plt.subplots(
        nrows=1, ncols=2, sharey=True, figsize=(12, 8)
    )
    maxabs = vd.maxabs(magnetic_grid, rtp_grid, percentile=99)


    tmp = magnetic_grid.plot(
        ax=ax1,
        add_colorbar=False,
        vmin=-maxabs,
        vmax=maxabs,
        cmap='RdBu_r',
    )

    tmp = rtp_grid.plot(
        ax=ax2,
        add_colorbar=False,
        vmin=-maxabs,
        vmax=maxabs,
        cmap='RdBu_r',
    )

    ax1.set_aspect("equal")
    ax2.set_aspect("equal")
    ax1.set_title("Magnetic anomaly")
    ax2.set_title("Reduced to the pole")
    ax1.ticklabel_format(style="sci", scilimits=(0, 0))
    ax2.ticklabel_format(style="sci", scilimits=(0, 0))

    fig.colorbar(tmp, ax=(ax1, ax2), orientation="horizontal", label="nT", fraction=.03, pad=0.08)
    plt.show()

The reduction concentrated the Lightning Creek anomaly and the negative
values are spread out and mixed with the regional field. So we could
consider this to be a valid reduction, indicating that the magnetization
direction used is plausible.

If on the other hand we have any knowledge about the orientation of the
magnetization vector of the sources, we can specify the
``magnetization_inclination`` and ``magnetization_declination``.
In this case, we don't have this information, but we'll show an
example of what happens to the reduction using arbitrary values:

.. jupyter-execute::

    mag_inclination, mag_declination = -25, 21

    tmp = rtp_grid = hm.reduction_to_pole(
        magnetic_grid,
        inclination=inclination,
        declination=declination,
        magnetization_inclination=mag_inclination,
        magnetization_declination=mag_declination,
    )
    rtp_grid

.. jupyter-execute::

    maxabs = vd.maxabs(rtp_grid, percentile=99.9)
    tmp = rtp_grid.plot(
        cmap="RdBu_r", vmin=-maxabs, vmax=maxabs, cbar_kwargs={'label':'nT'},
    )
    plt.gca().set_aspect("equal")
    plt.title("Reduced to the pole with remanence")
    plt.gca().ticklabel_format(style="sci", scilimits=(0, 0))
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

    cutoff_wavelength = 5e3  # meters

Then apply the two filters to our magnetic grid:

.. jupyter-execute::

    magnetic_low = hm.gaussian_lowpass(
        magnetic_grid, wavelength=cutoff_wavelength
    )
    magnetic_low

.. jupyter-execute::

    magnetic_high = hm.gaussian_highpass(
        magnetic_grid, wavelength=cutoff_wavelength
    )
    magnetic_high

Let's plot the results side by side:

.. jupyter-execute::

    fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 8))

    grids = [magnetic_grid, magnetic_low, magnetic_high]
    maxabs = vd.maxabs(*grids, percentile=99)
    for grid, ax in zip(grids, axes):
        tmp = grid.plot(
            ax=ax,
            add_colorbar=False,
            vmin=-maxabs,
            vmax=maxabs,
            cmap='RdBu_r',
        )
        ax.set_aspect("equal")
    axes[0].set_title("Original")
    axes[1].set_title("After low-pass filter")
    axes[2].set_title("After high-pass filter")
    fig.colorbar(tmp, ax=axes.ravel().tolist(), orientation="horizontal", label="nT", fraction=.03, pad=0.08)
    plt.show()


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
        magnetic_grid
    )
    tga_grid

And plot it:

.. jupyter-execute::

    cpt_lims = vd.minmax(tga_grid, min_percentile=1, max_percentile=99)
    tga_grid.plot(vmin=cpt_lims[0], vmax=cpt_lims[1],cbar_kwargs={'label':'nT/m'})
    plt.gca().set_aspect("equal")
    plt.title("Total gradient amplitude of the magnetic anomaly")
    plt.gca().ticklabel_format(style="sci", scilimits=(0, 0))
    plt.show()

----

.. grid:: 2

    .. grid-item-card:: :jupyter-download-script:`Download Python script <transformations>`
        :text-align: center

    .. grid-item-card:: :jupyter-download-nb:`Download Jupyter notebook <transformations>`
        :text-align: center
