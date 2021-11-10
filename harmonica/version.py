# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Define the __version__ variable as the version number with leading "v"
"""
# This file is generated automatically by setuptools_scm
from ._version import version

# Add a "v" to the version number
__version__ = f"v{version}"
