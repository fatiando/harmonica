.. _changes:

Changelog
=========

Version 0.2.0
-------------

*Released on: 2021/04/09*

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4672400.svg
   :alt: Digital Object Identifier for the Zenodo archive
   :target: https://doi.org/10.5281/zenodo.4672400


New features:

- Add function to create a layer of prisms and add a new South Africa ETOPO1 dataset (`#186 <https://github.com/fatiando/harmonica/pull/186>`__)
- Optimize forward models by parallelizing outer loops for prisms and point masses and refactor the tesseroids forward modelling (`#205 <https://github.com/fatiando/harmonica/pull/205>`__)
- Add parallel flag to EQLs (`#207 <https://github.com/fatiando/harmonica/pull/207>`__)
- Parallelize EQLs predictions and Jacobian build (`#203 <https://github.com/fatiando/harmonica/pull/203>`__)
- Improve EQL harmonic classes by splitting classes and adding upward argument to prediction methods (`#190 <https://github.com/fatiando/harmonica/pull/190>`__)
- Add function to compute the distance between points given in geodetic coordinates (`#172 <https://github.com/fatiando/harmonica/pull/172>`__)
- Allow ``load_icgem_gdf`` to take open file objects (`#155 <https://github.com/fatiando/harmonica/pull/155>`__)
- Add new ``EQLHarmonicSpherical`` class to interpolate data using EQL in spherical coordinates (`#136 <https://github.com/fatiando/harmonica/pull/136>`__)


Maintenance:

- Extend support for Python 3.9 (`#219 <https://github.com/fatiando/harmonica/pull/219>`__)
- Separate the Actions jobs into categories (`#218 <https://github.com/fatiando/harmonica/pull/218>`__)
- Automatically check for license notice when checking code style (`#211 <https://github.com/fatiando/harmonica/pull/211>`__)
- Use the OSI version of item 3 in the license (`#206 <https://github.com/fatiando/harmonica/pull/206>`__)
- Add license and copyright notice to every .py file (`#201 <https://github.com/fatiando/harmonica/pull/201>`__)
- Replace ``versioneer`` with ``setuptools_scm`` (`#196 <https://github.com/fatiando/harmonica/pull/196>`__)
- Remove configuration files for unused CI: Stickler, Codacy and CodeClimate (`#197 <https://github.com/fatiando/harmonica/pull/197>`__)
- Replace TravisCI and Azure for GitHub Actions (`#189 <https://github.com/fatiando/harmonica/pull/189>`__)
- Fetch a sample data before testing locate because Pooch creates cache directory only after the first fetch (`#193 <https://github.com/fatiando/harmonica/pull/193>`__)
- Require Black>=20.8b1 (`#187 <https://github.com/fatiando/harmonica/pull/187>`__)
- Add CI builds for Python 3.8 (`#150 <https://github.com/fatiando/harmonica/pull/150>`__)
- Replace global Zenodo DOI on badges and citation (`#148 <https://github.com/fatiando/harmonica/pull/148>`__)
- Remove the GitHub templates from the repository and use the shared ones in fatiando/.github (`#149 <https://github.com/fatiando/harmonica/pull/149>`__)


Documentation:

- Add example for ``EQLHarmonicSpherical`` (`#152 <https://github.com/fatiando/harmonica/pull/152>`__)
- Replace Cartesian distance for Euclidean distance in docs (`#156 <https://github.com/fatiando/harmonica/pull/156>`__)


Bug fixes:

- Keep metadata of sample datasets in the Xarray objects (`#184 <https://github.com/fatiando/harmonica/pull/184>`__)
- Fix infinite loop on CIs after Numba v0.5.0 (`#180 <https://github.com/fatiando/harmonica/pull/180>`__)


This release contains contributions from:

- Santiago Soler
- Leonardo Uieda
- Nicholas Shea
- Rowan Cockett


Version 0.1.0
-------------

*Released on: 2020/02/27*

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3628742.svg
    :alt: Digital Object Identifier for the Zenodo archive
    :target: https://doi.org/10.5281/zenodo.3628742

Fist release of Harmonica. Forward modeling, inversion, and processing gravity
and magnetic data.

Forward models:

- Point masses in Cartesian coordinates. Gravitational potential with vertical
  (`#71 <https://github.com/fatiando/harmonica/pull/71>`__) and horizontal
  components of acceleration
  (`#119 <https://github.com/fatiando/harmonica/pull/119>`__).
- Point masses in spherical coordinates.
  (`#60 <https://github.com/fatiando/harmonica/pull/60>`__)
- Rectangular prisms. (`#86 <https://github.com/fatiando/harmonica/pull/86>`__)
- Tesseroids. (`#60 <https://github.com/fatiando/harmonica/pull/60>`__)


Available datasets:

- Great Britain aeromagnetic dataset.
  (`#111 <https://github.com/fatiando/harmonica/pull/111>`__)
- South Africa gravity station data.
  (`#99 <https://github.com/fatiando/harmonica/pull/99>`__)
- Geoid grid from EIGEN-6C4.
  (`#34 <https://github.com/fatiando/harmonica/pull/34>`__)
- Global topography from ETOPO1.
  (`#23 <https://github.com/fatiando/harmonica/pull/23>`__)
- Global gravity data from EIGEN-6C4.
  (`#12 <https://github.com/fatiando/harmonica/pull/12>`__)


Other features:

- Synthetic ground and airborne surveys based on real world data.
  (`#120 <https://github.com/fatiando/harmonica/pull/120>`__)
- Equivalent Layer for harmonic functions.
  (`#78 <https://github.com/fatiando/harmonica/pull/78>`__)
- Planar Bouguer correction.
  (`#38 <https://github.com/fatiando/harmonica/pull/38>`__)
- Airy Isostasy function.
  (`#17 <https://github.com/fatiando/harmonica/pull/17>`__)
- Function to read ICGEM data files.
  (`#11 <https://github.com/fatiando/harmonica/pull/11>`__)


This release contains contributions from:

- Leonardo Uieda
- Santiago Soler
- Vanderlei C Oliveira Jr
- Agustina Pesce
- Nicholas Shea
- ziebam
