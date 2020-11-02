"""
Decorators and useful functions for running tests
"""
import os
import pytest


def require_numba(function):  # pylint: disable=unused-argument
    """
    Function decorator for pytest: run if Numba jit is enabled

    Functions decorator to tell pytest to run the test function only if Numba
    jit is enabled. To disable Numba jit the environmental variable
    ```NUMBA_DISABLE_JIT``` must be set to a value different than 0.

    Use this decorator on test functions that involve great computational load
    and don't want to run if Numba jit is disabled. The decorated test
    functions will be run and checked if pass or fail, but won't be taken into
    account for meassuring coverage. If the test function will run Numba code,
    but doesn't involve great computational load, we reccomend using the
    ``@pytest.mark.use_numba`` instead. Therefore the test function will be run
    twice: one with Numba jit enabled, and another one with Numba jit disable
    to check coverage.
    """
    # Check if Numba is disabled
    # (if NUMBA_DISABLE_JIT is not defined, we assume Numba jit is enabled)
    numba_is_disabled = bool(os.environ.get("NUMBA_DISABLE_JIT", default="0") != "0")

    @pytest.mark.use_numba
    @pytest.mark.skipif(numba_is_disabled, reason="Numba jit is disabled")
    def function_wrapper():
        function()

    return function_wrapper
