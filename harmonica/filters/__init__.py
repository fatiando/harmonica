# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Frequency domain filters meant to be applied on regular grids
"""
from ._filters import (
    derivative_easting_kernel,
    derivative_northing_kernel,
    derivative_upward_kernel,
    gaussian_highpass_kernel,
    gaussian_lowpass_kernel,
    reduction_to_pole_kernel,
    upward_continuation_kernel,
)
