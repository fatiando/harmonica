# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test the IGRF results against the ones calculated by the BGS.
"""
import pytest


from .._spherical_harmonics.igrf import load_igrf, fetch_igrf,
interpolate_coefficients, IGRF14
