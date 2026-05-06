#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xarray as xr
import matplotlib.pyplot as plt
import pyproj
import verde as vd
import harmonica as hm
import ensaio

fname = ensaio.fetch_lightning_creek_magnetic(version=1)
magnetic_grid = xr.load_dataarray(fname)
magnetic_grid


# In[2]:


tmp = magnetic_grid.plot(cmap="seismic", center=0, add_colorbar=False)
plt.gca().set_aspect("equal")
plt.title("Magnetic anomaly grid")
plt.gca().ticklabel_format(style="sci", scilimits=(0, 0))
plt.colorbar(tmp, label="nT")
plt.show()


# In[3]:


deriv_upward = hm.derivative_upward(magnetic_grid)
deriv_upward


# In[4]:


tmp = deriv_upward.plot(cmap="seismic", center=0, add_colorbar=False)
plt.gca().set_aspect("equal")
plt.title("Upward derivative of the magnetic anomaly")
plt.gca().ticklabel_format(style="sci", scilimits=(0, 0))
plt.colorbar(tmp, label="nT/m")
plt.show()


# In[5]:


deriv_easting = hm.derivative_easting(magnetic_grid)
deriv_easting


# In[6]:


deriv_northing = hm.derivative_northing(magnetic_grid)
deriv_northing


# In[7]:


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


# In[8]:


deriv_easting = hm.derivative_easting(magnetic_grid, method="fft")
deriv_easting


# In[9]:


deriv_northing = hm.derivative_northing(magnetic_grid, method="fft")
deriv_northing


# In[10]:


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


# In[11]:


change_in_height = 500  # meters
upward_continued = hm.upward_continuation(
    magnetic_grid, height_displacement=change_in_height
)
upward_continued


# In[12]:


upward_continued = upward_continued.assign_coords(
    {"height": magnetic_grid.height + change_in_height}
)
upward_continued


# In[13]:


tmp = upward_continued.plot(cmap="seismic", center=0, add_colorbar=False)
plt.gca().set_aspect("equal")
plt.title("Upward continued magnetic anomaly to 1000m")
plt.gca().ticklabel_format(style="sci", scilimits=(0, 0))
plt.colorbar(tmp, label="nT")
plt.show()


# In[14]:


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


# In[15]:


rtp_grid = hm.reduction_to_pole(
    magnetic_grid,
    inclination=inclination,
    declination=declination,
    magnetization_inclination=inclination,
    magnetization_declination=declination,
)
rtp_grid


# In[16]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
cbar_kwargs=dict(
    label="nT", orientation="horizontal", shrink=0.8, pad=0.08, aspect=42
)
magnetic_grid.plot(ax=ax1, cmap="seismic", center=0, cbar_kwargs=cbar_kwargs)
rtp_grid.plot(ax=ax2, cmap="seismic", center=0, cbar_kwargs=cbar_kwargs)
ax1.set_aspect("equal")
ax2.set_aspect("equal")
ax1.set_title("Magnetic anomaly")
ax2.set_title("Reduced to the pole")
ax1.ticklabel_format(style="sci", scilimits=(0, 0))
ax2.ticklabel_format(style="sci", scilimits=(0, 0))
plt.show()


# In[17]:


mag_inclination, mag_declination = -25, 21

tmp = rtp_grid = hm.reduction_to_pole(
    magnetic_grid,
    inclination=inclination,
    declination=declination,
    magnetization_inclination=mag_inclination,
    magnetization_declination=mag_declination,
)
rtp_grid


# In[18]:


tmp = rtp_grid.plot(cmap="seismic", center=0, add_colorbar=False)
plt.gca().set_aspect("equal")
plt.title("Reduced to the pole with remanence")
plt.gca().ticklabel_format(style="sci", scilimits=(0, 0))
plt.colorbar(tmp, label="nT")
plt.show()


# In[19]:


cutoff_wavelength = 5e3  # meters


# In[20]:


magnetic_low = hm.gaussian_lowpass(
    magnetic_grid, wavelength=cutoff_wavelength
)
magnetic_low


# In[21]:


magnetic_high = hm.gaussian_highpass(
    magnetic_grid, wavelength=cutoff_wavelength
)
magnetic_high


# In[22]:


fig, (ax1, ax2, ax3) = plt.subplots(
    nrows=1, ncols=3, sharey=True, figsize=(14, 8)
)

maxabs = vd.maxabs(magnetic_grid, magnetic_low, magnetic_high)
kwargs = dict(cmap="seismic", vmin=-maxabs, vmax=maxabs, add_colorbar=False)

tmp = magnetic_grid.plot(ax=ax1, **kwargs)
tmp = magnetic_low.plot(ax=ax2, **kwargs)
tmp = magnetic_high.plot(ax=ax3, **kwargs)

ax1.set_title("Original")
ax2.set_title("After low-pass filter")
ax3.set_title("After high-pass filter")
for ax in (ax1, ax2, ax3):
    ax.set_aspect("equal")
    ax.ticklabel_format(style="sci", scilimits=(0, 0))

plt.colorbar(
    tmp,
    ax=[ax1, ax2, ax3],
    label="nT",
    orientation="horizontal",
    aspect=42,
    shrink=0.8,
    pad=0.08,
)
plt.show()


# In[23]:


tga_grid = hm.total_gradient_amplitude(
    magnetic_grid
)
tga_grid


# In[24]:


tmp = tga_grid.plot(cmap="viridis", add_colorbar=False)
plt.gca().set_aspect("equal")
plt.title("Total gradient amplitude of the magnetic anomaly")
plt.gca().ticklabel_format(style="sci", scilimits=(0, 0))
plt.colorbar(tmp, label="nT/m")
plt.show()

