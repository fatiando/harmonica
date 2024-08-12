#!/usr/bin/env python
# coding: utf-8

# In[1]:


import verde as vd

# Define boundaries of the rectangular prism (in meters)
west, east, south, north = -20, 20, -20, 20
bottom, top = -40, -20
prism = [west, east, south, north, bottom, top]


# In[2]:


# Define a regular grid of observation points (coordinates in meters)
coordinates = vd.grid_coordinates(
    region=(-40, 40, -40, 40), shape=(5, 5), extra_coords=100
)
easting, northing, upward = coordinates[:]

print("easting:", easting)
print("northing:", northing)
print("upward:", upward)


# In[3]:


# Define two points
point_1 = (30, 20, -67)
point_2 = (30, 20, -58)
print("Point 1 is higher than point 2?", point_1[2] > point_2[2])


# In[4]:


coordinates = vd.grid_coordinates(
    region=(-70, -65, -35, -30), shape=(6, 6), extra_coords=2e3
)
longitude, latitude, height = coordinates[:]
print("longitude:", longitude)
print("latitude:", latitude)
print("height:", height)


# In[5]:


import boule as bl

ellipsoid = bl.WGS84
print(ellipsoid)


# In[6]:


import pyproj

# Define a Mercator projection through pyproj
projection = pyproj.Proj(proj="merc", ellps="WGS84")

# Project the longitude and latitude coordinates of the grid points
longitude, latitude = coordinates[:2]
easting, northing = projection(longitude, latitude)

print("easting:", easting)
print("northing:", northing)


# In[7]:


coordinates = vd.grid_coordinates(
    region=(-70, -65, -35, -30),
    shape=(6, 6),
    extra_coords=ellipsoid.mean_radius,
)
longitude, sph_latitude, radius = coordinates[:]
print("longitude:", longitude)
print("spherical latitude:", sph_latitude)
print("radius:", radius)


# In[8]:


coordinates_geodetic = ellipsoid.spherical_to_geodetic(*coordinates)
longitude, latitude, height = coordinates_geodetic[:]
print("longitude:", longitude)
print("latitude:", latitude)
print("height:", height)


# In[9]:


coordinates_spherical = ellipsoid.geodetic_to_spherical(*coordinates_geodetic)
longitude, sph_latitude, radius = coordinates_spherical[:]
print("longitude:", longitude)
print("spherical latitude:", sph_latitude)
print("radius:", radius)

