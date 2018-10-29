# Sample data sets

These files are used as sample data in Harmonica:

* `global_gravity_earth.nc.xz`: Global gravity data set. The file contains the
  magnitude of the gradient of the gravitational potential of the Earth
  (including the centrifugal potential) on a regular grid of points located on
  the Earth surface. The file was downloaded from the [ICGEM Calculation
  Service](http://icgem.gfz-potsdam.de/calcgrid) using the EIGEN-6C4 gravity
  field model and the WGS84 Reference System. The data is stored in a netCDF
  file and `xz` compressed. The netCDF file can be loaded through the
  `xarray.open_dataset()` function, obtaining a `xarray.Dataset` object.

