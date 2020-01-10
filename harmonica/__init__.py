# pylint: disable=missing-docstring,import-outside-toplevel
# Import functions/classes to make the public API
from . import version
from . import datasets
from . import synthetic
from .io import load_icgem_gdf
from .isostasy import isostasy_airy
from .gravity_corrections import bouguer_correction
from .forward.point_mass import point_mass_gravity
from .forward.tesseroid import tesseroid_gravity
from .forward.prism import prism_gravity
from .equivalent_layer.harmonic import EQLHarmonic

# Get the version number through versioneer
__version__ = version.full_version


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
