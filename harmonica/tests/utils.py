"""
Decorators and useful functions for running tests
"""
import os
import pytest


def require_numba(function):  # pylint: disable=unused-argument
    """
    Decorator to tell pytest to run the test function only if Numba jit is enabled.

    Use this decorator on test functions that involve great computational load and don't
    want to run if Numba jit is disabled.
    The decorated test functions will be run and checked if pass or fail, but won't be
    taken into account for meassuring coverage.
    If the test function will run Numba code, but doesn't involve great computational
    load, we reccomend using the ``@pytest.mark.use_numba`` instead. Therefore the test
    function will be run twice: one with Numba jit enabled, and another one with Numba
    jit disable to check coverage.
    """
    # Check if env variable NUMBA_DISABLE_JIT is defined
    if os.getenv("NUMBA_DISABLE_JIT") is None:
        raise EnvironmentError(
            "Enviromental variable NUMBA_DISABLE_JIT is not defined."
            + " Cannot run tests unless it's set to 0 or 1"
            + " (for running non-compiled or compiled versios of jitted functions,"
            + " respectively)."
        )
    # Check if Numba is disabled
    numba_is_disabled = bool(os.getenv("NUMBA_DISABLE_JIT") != "0")

    @pytest.mark.use_numba
    @pytest.mark.skipif(numba_is_disabled, reason="Numba jit is disabled")
    def function_wrapper():
        return

    return function_wrapper
