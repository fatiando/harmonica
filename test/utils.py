# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Decorators and useful functions for running tests.
"""

import numpy as np


def root_mean_square_error(x, y):
    """
    Compute root mean square error between two arrays.

    Parameters
    ----------
    x : array
    y : array

    Returns
    -------
    rms : float
        Root mean square error between the two arrays
    """
    x = np.asarray(x)
    y = np.asarray(y)
    return np.sqrt(np.mean((x - y) ** 2))


def combine_decorators(*decorators):
    """
    Combine several decorators into a single one.
    """

    def combination(func):
        for decorator in reversed(decorators):
            func = decorator(func)
        return func

    return combination
