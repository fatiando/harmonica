"""
Decorators and useful functions for running tests
"""
import os
import pytest


def require_numba(function):
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
    reason = "Numba jit is disabled"

    @pytest.mark.use_numba
    @pytest.mark.skipif(os.environ["NUMBA_DISABLE_JIT"] != "0", reason=reason)
    def function_wrapper():
        return

    return function_wrapper
