# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Decorators and useful functions for running tests
"""
import os

import pytest


def combine_decorators(*decorators):
    """
    Combine several decorators into a single one
    """

    def combination(func):
        for decorator in reversed(decorators):
            func = decorator(func)
        return func

    return combination


# Check if Numba is disabled
# (if NUMBA_DISABLE_JIT is not defined, we assume Numba jit is enabled)
NUMBA_IS_DISABLED = bool(os.environ.get("NUMBA_DISABLE_JIT", default="0") != "0")

# Decorator for pytest: run if Numba jit is enabled
#
# Tell pytest to run the test function only if Numba jit is enabled. To disable
# Numba jit the environmental variable ```NUMBA_DISABLE_JIT``` must be set to
# a value different than 0.
#
# Use this decorator on test functions that involve great computational load
# and don't want to run if Numba jit is disabled. The decorated test functions
# will be run and checked if pass or fail, but won't be taken into account for
# meassuring coverage. If the test function will run Numba code, but doesn't
# involve great computational load, we reccomend using the
# ``@pytest.mark.use_numba`` instead. Therefore the test function will be run
# twice: one with Numba jit enabled, and another one with Numba jit disable to
# check coverage.
run_only_with_numba = combine_decorators(
    pytest.mark.skipif(NUMBA_IS_DISABLED, reason="Numba jit is disabled"),
    pytest.mark.use_numba,
)
