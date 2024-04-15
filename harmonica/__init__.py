# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
#
# Import functions/classes to make the public API
from ._equivalent_sources.cartesian import EquivalentSources
from ._equivalent_sources.gradient_boosted import EquivalentSourcesGB
from ._equivalent_sources.spherical import EquivalentSourcesSph
from ._forward.dipole import dipole_magnetic
from ._forward.point import point_gravity
from ._forward.prism_gravity import prism_gravity
from ._forward.prism_layer import DatasetAccessorPrismLayer, prism_layer
from ._forward.prism_magnetic import prism_magnetic
from ._forward.tesseroid import tesseroid_gravity
from ._forward.tesseroid_layer import DatasetAccessorTesseroidLayer, tesseroid_layer
from ._gravity_corrections import bouguer_correction
from ._io.icgem_gdf import load_icgem_gdf
from ._io.oasis_montaj_grd import load_oasis_montaj_grid
from ._isostasy import isostatic_moho_airy
from ._transformations import (
    derivative_easting,
    derivative_northing,
    derivative_upward,
    gaussian_highpass,
    gaussian_lowpass,
    reduction_to_pole,
    tilt_angle,
    total_gradient_amplitude,
    upward_continuation,
)
from ._utils import magnetic_angles_to_vec, magnetic_vec_to_angles
from ._version import __version__

# Append a leading "v" to the generated version by setuptools_scm
__version__ = f"v{__version__}"
