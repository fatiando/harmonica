#!/usr/bin/env python
# coding: utf-8

# In[1]:


import ensaio
import xarray as xr

fname = ensaio.fetch_lightning_creek_magnetic(version=1)
magnetic_grid = xr.load_dataarray(fname)
magnetic_grid


# In[2]:


import matplotlib.pyplot as plt

tmp = magnetic_grid.plot(cmap="seismic", center=0, add_colorbar=False)
plt.gca().set_aspect("equal")
plt.title("Magnetic anomaly grid")
plt.gca().ticklabel_format(style="sci", scilimits=(0, 0))
plt.colorbar(tmp, label="nT")
plt.show()


# In[3]:


pad_width = {
    "easting": magnetic_grid.easting.size // 3,
    "northing": magnetic_grid.northing.size // 3,
}


# In[4]:


import xrft

magnetic_grid_no_height = magnetic_grid.drop_vars("height")
magnetic_grid_padded = xrft.pad(magnetic_grid_no_height, pad_width)
magnetic_grid_padded


# In[5]:


tmp = magnetic_grid_padded.plot(cmap="seismic", center=0, add_colorbar=False)
plt.gca().set_aspect("equal")
plt.title("Padded magnetic anomaly grid")
plt.gca().ticklabel_format(style="sci", scilimits=(0, 0))
plt.colorbar(tmp, label="nT")
plt.show()


# In[6]:


import harmonica as hm

deriv_upward = hm.derivative_upward(magnetic_grid_padded)
deriv_upward


# In[7]:


deriv_upward = xrft.unpad(deriv_upward, pad_width)
deriv_upward


# In[8]:


tmp = deriv_upward.plot(cmap="seismic", center=0, add_colorbar=False)
plt.gca().set_aspect("equal")
plt.title("Upward derivative of the magnetic anomaly")
plt.gca().ticklabel_format(style="sci", scilimits=(0, 0))
plt.colorbar(tmp, label="nT/m")
plt.show()


# In[9]:


deriv_easting = hm.derivative_easting(magnetic_grid)
deriv_easting


# In[10]:


deriv_northing = hm.derivative_northing(magnetic_grid)
deriv_northing


# In[11]:


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


# In[12]:


deriv_easting = hm.derivative_easting(magnetic_grid_padded, method="fft")
deriv_easting = xrft.unpad(deriv_easting, pad_width)
deriv_easting


# In[13]:


deriv_northing = hm.derivative_northing(magnetic_grid_padded, method="fft")
deriv_northing = xrft.unpad(deriv_northing, pad_width)
deriv_northing


# In[14]:


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


# In[15]:


upward_continued = hm.upward_continuation(
    magnetic_grid_padded, height_displacement=500
)


# In[16]:


upward_continued = xrft.unpad(upward_continued, pad_width)
upward_continued


# In[17]:


tmp = upward_continued.plot(cmap="seismic", center=0, add_colorbar=False)
plt.gca().set_aspect("equal")
plt.title("Upward continued magnetic anomaly to 1000m")
plt.gca().ticklabel_format(style="sci", scilimits=(0, 0))
plt.colorbar(tmp, label="nT")
plt.show()


# In[18]:


inclination, declination = -52.98, 6.51


# In[19]:


rtp_grid = hm.reduction_to_pole(
    magnetic_grid_padded, inclination=inclination, declination=declination
)

# Unpad the reduced to the pole grid
rtp_grid = xrft.unpad(rtp_grid, pad_width)
rtp_grid


# In[20]:


tmp = rtp_grid.plot(cmap="seismic", center=0, add_colorbar=False)
plt.gca().set_aspect("equal")
plt.title("Magnetic anomaly reduced to the pole")
plt.gca().ticklabel_format(style="sci", scilimits=(0, 0))
plt.colorbar(tmp, label="nT")
plt.show()


# In[21]:


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


# In[22]:


tmp = rtp_grid.plot(cmap="seismic", center=0, add_colorbar=False)
plt.gca().set_aspect("equal")
plt.title("Reduced to the pole with remanence")
plt.gca().ticklabel_format(style="sci", scilimits=(0, 0))
plt.colorbar(tmp, label="nT")
plt.show()


# In[23]:


cutoff_wavelength = 5e3


# In[24]:


magnetic_low_freqs = hm.gaussian_lowpass(
    magnetic_grid_padded, wavelength=cutoff_wavelength
)
magnetic_high_freqs = hm.gaussian_highpass(
    magnetic_grid_padded, wavelength=cutoff_wavelength
)


# In[25]:


magnetic_low_freqs = xrft.unpad(magnetic_low_freqs, pad_width)
magnetic_high_freqs = xrft.unpad(magnetic_high_freqs, pad_width)


# In[26]:


magnetic_low_freqs


# In[27]:


magnetic_high_freqs


# In[28]:


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


# In[29]:


tga_grid = hm.total_gradient_amplitude(
    magnetic_grid_padded
)

# Unpad the total gradient amplitude grid
tga_grid = xrft.unpad(tga_grid, pad_width)
tga_grid


# In[30]:


import verde as vd

tmp = tga_grid.plot(cmap="viridis", add_colorbar=False)
plt.gca().set_aspect("equal")
plt.title("Total gradient amplitude of the magnetic anomaly")
plt.gca().ticklabel_format(style="sci", scilimits=(0, 0))
plt.colorbar(tmp, label="nT/m")
plt.show()

