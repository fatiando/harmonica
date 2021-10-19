# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
# pylint: disable=missing-docstring,import-outside-toplevel,import-self
#
# Import functions/classes to make the public API
from . import datasets
from . import synthetic
from .io import load_icgem_gdf
from .isostasy import isostasy_airy
from .gravity_corrections import bouguer_correction
from .forward.point_mass import point_mass_gravity
from .forward.tesseroid import tesseroid_gravity
from .forward.prism import prism_gravity
from .forward.prism_layer import prism_layer, DatasetAccessorPrismLayer
from .equivalent_sources.cartesian import EquivalentSources, EQLHarmonic
from .equivalent_sources.spherical import (
    EquivalentSourcesSph,
    EQLHarmonicSpherical,
)

# This file is generated automatically by setuptools_scm
from . import _version

# Add a "v" to the version number
__version__ = f"v{_version.version}"


def test(doctest=True, verbose=True, coverage=False, figures=False):
    """
    Run the test suite.

    Uses `py.test <http://pytest.org/>`__ to discover and run the tests.

    Parameters
    ----------

    doctest : bool
        If ``True``, will run the doctests as well (code examples that start
        with a ``>>>`` in the docs).
    verbose : bool
        If ``True``, will print extra information during the test run.
    coverage : bool
        If ``True``, will run test coverage analysis on the code as well.
        Requires ``pytest-cov``.
    figures : bool
        If ``True``, will test generated figures against saved baseline
        figures.  Requires ``pytest-mpl`` and ``matplotlib``.

    Raises
    ------

    AssertionError
        If pytest returns a non-zero error code indicating that some tests have
        failed.

    """
    import pytest

    package = __name__
    args = []
    if verbose:
        args.append("-vv")
    if coverage:
        args.append("--cov={}".format(package))
        args.append("--cov-report=term-missing")
    if doctest:
        args.append("--doctest-modules")
    if figures:
        args.append("--mpl")
    args.append("--pyargs")
    args.append(package)
    status = pytest.main(args)
    assert status == 0, "Some tests have failed."
