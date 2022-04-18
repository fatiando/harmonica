.. _changes:

Changelog
=========

Version 0.4.0
-------------

*Released on: 2021/12/02*

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.5745400.svg
   :alt: Digital Object Identifier for the Zenodo archive
   :target: https://doi.org/10.5281/zenodo.5745400

New features:

- Allow ``EquivalentSources`` to define block-averaged sources through a new ``block_size`` argument [Soler2021]_. (`#260 <https://github.com/fatiando/harmonica/pull/260>`__)
- Add ``dtype`` argument to ``EquivalentSources``. Allows to select the data type used to allocate the Jacobian matrix. (`#278 <https://github.com/fatiando/harmonica/pull/278>`__)
- Add a new ``EquivalentSourcesGB`` class that implements gradient-boosted equivalent sources. Provides a method to estimate the amount of computer memory needed to allocate the largest Jacobian matrix [Soler2021]_. (`#275 <https://github.com/fatiando/harmonica/pull/275>`__)
- Allow ``tesseroid_gravity`` to compute gravitational fields of variable density tesseroids. Implements the density-based discretization algorithm and takes ``numba.njit`` decorated density functions as input [Soler2019]_. (`#269 <https://github.com/fatiando/harmonica/pull/269>`__)

Breaking changes:

- Rename ``point_mass_gravity`` to ``point_gravity``. Having mass and gravity in the same function name is redundant. The function name has the same structure as other forward modelling functions (``tesseroid_gravity`` and ``prism_gravity``). The old ``point_mass_gravity`` will be deprecated on the next release. (`#280 <https://github.com/fatiando/harmonica/pull/280>`__)

Bug fixes:

- Fix bug with the ``require_numba`` pytest mark and rename it to ``run_only_with_numba`` for improved readability. (`#273 <https://github.com/fatiando/harmonica/pull/273>`__)

Documentation:

- Fix typo on ``EquivalentSources`` docstring: replace ``bloc_size`` with ``block_size``. (`#276 <https://github.com/fatiando/harmonica/pull/276>`__)
- Minor improvements to the docs: fix bad references and links, replace Equivalent Layer for Equivalent Sources on API Index, fix bad RST syntax. (`#274 <https://github.com/fatiando/harmonica/pull/274>`__)

Maintenance:

- Rename the default branch: from ``master`` to ``main`` (`#287 <https://github.com/fatiando/harmonica/pull/287>`__)
- Replace ``pylint`` for ``flake8`` extensions. Add ``isort`` for autoformatting imports. (`#285 <https://github.com/fatiando/harmonica/pull/285>`__)
- Replace conda for pip on GitHub Actions and split requirements files for each separate task. (`#282 <https://github.com/fatiando/harmonica/pull/282>`__)
- Make GitHub Actions to check if license notice is present in source files. (`#277 <https://github.com/fatiando/harmonica/pull/277>`__)

This release contains contributions from:

- Santiago Soler


Version 0.3.3
-------------

*Released on: 2021/10/22*

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.5593112.svg
   :alt: Digital Object Identifier for the Zenodo archive
   :target: https://doi.org/10.5281/zenodo.5593112

Bug fix:

- Add ``EquivalentSources`` and ``EquivalentSourcesSph`` to API index. Replace the old equivalent layer classes. (`#270 <https://github.com/fatiando/harmonica/pull/270>`__)

This release contains contributions from:

- Santiago Soler


Version 0.3.2
-------------

*Released on: 2021/10/21*

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.5589989.svg
   :alt: Digital Object Identifier for the Zenodo archive
   :target: https://doi.org/10.5281/zenodo.5589989

Bug fixes:

- Fix import of Harmonica version on sample datasets: solves a problem whenbuilding docs for releases. Define the ``__version__`` variable inside a new ``version.py`` file. (`#264 <https://github.com/fatiando/harmonica/pull/264>`__)

This release contains contributions from:

- Santiago Soler


Version 0.3.1
-------------

*Released on: 2021/10/20*

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.5585665.svg
   :alt: Digital Object Identifier for the Zenodo archive
   :target: https://doi.org/10.5281/zenodo.5585665

Bug fix:

- Package ``requirements.txt`` and update the dependencies list: remove
  ``scipy`` and add ``scikit-learn``. Exclude ``license_notice.py`` and
  ``.flake8`` from the ``MANIFEST.in`` (`#261 <https://github.com/fatiando/harmonica/pull/261>`__)

This release contains contributions from:

- Santiago Soler


Version 0.3.0
-------------

*Released on: 2021/10/20*

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.5579324.svg
   :alt: Digital Object Identifier for the Zenodo archive
   :target: https://doi.org/10.5281/zenodo.5579324

Deprecations:

- Rename equivalent sources classes to ``EquivalentSources`` and ``EquivalentSourcesSph``. The old ``EQLHarmonic`` and ``EQLHarmonicSpherical`` will be removed on v0.5 (`#255 <https://github.com/fatiando/harmonica/pull/255>`__)
- Rename the ``relative_depth`` parameters in ``EquivalentSources`` to ``depth``. The old ``relative_depth`` parameter will be deleted on v0.5 (`#236 <https://github.com/fatiando/harmonica/pull/236>`__)

New features:

- Enable parallelization on tesseroids forward modelling and refactor its code (`#244 <https://github.com/fatiando/harmonica/pull/244>`__)
- Add option to set ``EquivalentSources`` points to constant depth (`#236 <https://github.com/fatiando/harmonica/pull/236>`__)
- Allow ``prism_layer`` to take Xarray objects as arguments (`#243 <https://github.com/fatiando/harmonica/pull/243>`__)

Maintenance:

- Generate version string on ``_version.py`` on build (`#237 <https://github.com/fatiando/harmonica/pull/237>`__)
- Run CIs only on the two ends of supported Python versions (`#256 <https://github.com/fatiando/harmonica/pull/256>`__)
- Transform ``require_numba`` decorator into a global variable (`#245 <https://github.com/fatiando/harmonica/pull/245>`__)

Documentation:

- Fix typo: replace bellow for below across docstrings (`#253 <https://github.com/fatiando/harmonica/pull/253>`__)
- Fix version display in the HTML title (`#249 <https://github.com/fatiando/harmonica/pull/249>`__)
- Remove unneeded line in prism_gravity example (`#248 <https://github.com/fatiando/harmonica/pull/248>`__)
- Update Fukushima2020 citation on References (`#246 <https://github.com/fatiando/harmonica/pull/246>`__)
- Change order of dims in example of ``prism_layer`` (`#241 <https://github.com/fatiando/harmonica/pull/241>`__)
- Fix class name on See also section in ``prism_layer`` (`#230 <https://github.com/fatiando/harmonica/pull/230>`__)
- Use the Jupyter book Sphinx theme instead of RTD (`#227 <https://github.com/fatiando/harmonica/pull/227>`__)

This release contains contributions from:

- Santiago Soler
- Leonardo Uieda


Version 0.2.1
-------------

*Released on: 2021/04/14*

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4685960.svg
   :alt: Digital Object Identifier for the Zenodo archive
   :target: https://doi.org/10.5281/zenodo.4685960


Minor changes:

- Rename prisms_layer to prism_layer (`#223 <https://github.com/fatiando/harmonica/pull/223>`__)


Bug fixes:

- Unpin Sphinx and fix documentation style (`#224 <https://github.com/fatiando/harmonica/pull/224>`__)


This release contains contributions from:

- Santiago Soler


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
