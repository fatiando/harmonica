"""
Test prisms layer
"""
import pytest
import numpy as np
import numpy.testing as npt

from .. import prisms_layer
from .. import prism_gravity


def test_prisms_layer_construction():
    """
    Check if a layer of prisms is property constructed

    To test:
        - spacing and shape
        - adjust
        - region vs boundary
        - top and bottom as None
        - multiple properties
        - no properties
    """
    region = (-1, 3, 7, 10)
    # Use spacing to define the layer
    spacing = 1
    layer = prisms_layer(region, spacing=spacing)
    assert "easting" in layer.coords
    assert "northing" in layer.coords
    assert "top" in layer.coords
    assert "bottom" in layer.coords
    npt.assert_allclose(layer.easting, [-1, 0, 1, 2, 3])
    npt.assert_allclose(layer.northing, [7, 8, 9, 10])
    # Use shape to define the layer
    shape = (4, 5)
    layer = prisms_layer(region, shape=shape)
    assert "easting" in layer.coords
    assert "northing" in layer.coords
    assert "top" in layer.coords
    assert "bottom" in layer.coords
    npt.assert_allclose(layer.easting, [-1, 0, 1, 2, 3])
    npt.assert_allclose(layer.northing, [7, 8, 9, 10])


def test_prisms_layer_top_bottom():
    """
    Check passing top and bottom boundaries to prisms layer
    """
    spacing = 1
    region = (-1, 3, 7, 10)
    # No top nor bottom
    layer = prisms_layer(region, spacing=spacing)
    npt.assert_allclose(layer.bottom, np.nan)
    npt.assert_allclose(layer.top, np.nan)
    assert layer.top.shape == (4, 5)
    assert layer.bottom.shape == (4, 5)
    # Only bottom
    layer = prisms_layer(region, spacing=spacing, bottom=0)
    npt.assert_allclose(layer.bottom, 0)
    npt.assert_allclose(layer.top, np.nan)
    assert layer.top.dims == ("northing", "easting")
    assert layer.bottom.dims == ("northing", "easting")
    assert layer.top.shape == (4, 5)
    assert layer.bottom.shape == (4, 5)
    # Only top
    layer = prisms_layer(region, spacing=spacing, top=100)
    npt.assert_allclose(layer.bottom, np.nan)
    npt.assert_allclose(layer.top, 100)
    assert layer.top.dims == ("northing", "easting")
    assert layer.bottom.dims == ("northing", "easting")
    assert layer.top.shape == (4, 5)
    assert layer.bottom.shape == (4, 5)
    # Both top and bottom
    layer = prisms_layer(region, spacing=spacing, bottom=-10, top=100)
    npt.assert_allclose(layer.bottom, -10)
    npt.assert_allclose(layer.top, 100)
    assert layer.top.dims == ("northing", "easting")
    assert layer.bottom.dims == ("northing", "easting")
    assert layer.top.shape == (4, 5)
    assert layer.bottom.shape == (4, 5)


def test_prisms_layer_properties():
    """
    Check passing physical properties to the prisms layer
    """
    spacing = 5e3
    region = (-20e3, 10e3, -30e3, -20e3)
    # No properties
    layer = prisms_layer(region, spacing=spacing)
    assert len(layer.data_vars) == 0
    # Single property
    layer = prisms_layer(region, spacing=spacing, properties={"density": 2670})
    npt.assert_allclose(layer.density, 2670)
    assert layer.density.dims == ("northing", "easting")
    # Multiple properties
    layer = prisms_layer(
        region, spacing=spacing, properties={"density": 2670, "magnetization": 1001}
    )
    npt.assert_allclose(layer.density, 2670)
    npt.assert_allclose(layer.magnetization, 1001)
    assert layer.density.dims == ("northing", "easting")
    assert layer.magnetization.dims == ("northing", "easting")


def test_prisms_layer_pixel_register():
    """
    Check layer construction under pixel_register values
    """
    spacing = 5e3
    region = (-20e3, 10e3, -30e3, -20e3)
    # pixel_register to False
    # The boundaries of the layer will be equal to region plus a padding equal
    # to half the size of the prisms
    layer = prisms_layer(region, spacing=spacing)
    assert layer.prisms_layer.shape == (3, 7)
    assert layer.prisms_layer.size == 21
    npt.assert_allclose(layer.prisms_layer.spacing, (5e3, 5e3))
    npt.assert_allclose(
        layer.prisms_layer.boundaries,
        (-22.5e3, 12.5e3, -32.5e3, -17.5e3),
    )
    # pixel_register to True
    # Now the boundaries should be equal to the passed region
    layer = prisms_layer(region, spacing=spacing, pixel_register=True)
    assert layer.prisms_layer.shape == (2, 6)
    assert layer.prisms_layer.size == 12
    npt.assert_allclose(layer.prisms_layer.spacing, (5e3, 5e3))
    npt.assert_allclose(layer.prisms_layer.boundaries, region)


def test_prisms_layer_uneven_spacing():
    """
    Check prisms sizes and layer boundaries after uneven spacing
    """
    # The prisms will be twice larger on the easting direction
    spacing = (5e3, 10e3)
    region = (0, 20e3, 0, 20e3)
    layer = prisms_layer(region, spacing=spacing)
    assert layer.prisms_layer.shape == (5, 3)
    npt.assert_allclose(layer.prisms_layer.boundaries, (-5e3, 25e3, -2.5e3, 22.5e3))


def test_prisms_layer_adjust():
    """
    Check prisms layer method after adjusting spacing or region
    """
    # Adjust spacing
    spacing = 5.2e3
    region = (0, 20e3, 0, 20e3)
    layer = prisms_layer(region, spacing=spacing, adjust="spacing")
    npt.assert_allclose(layer.prisms_layer.spacing, 5e3)
    npt.assert_allclose(layer.prisms_layer.boundaries, (-2.5e3, 22.5e3, -2.5e3, 22.5e3))
    # Adjust region
    spacing = 5e3
    region = (0, 21e3, 0, 21e3)
    layer = prisms_layer(region, spacing=spacing, adjust="region")
    npt.assert_allclose(layer.prisms_layer.spacing, 5e3)
    npt.assert_allclose(layer.prisms_layer.boundaries, (-2.5e3, 22.5e3, -2.5e3, 22.5e3))


def test_prisms_layer_get_prisms():
    """
    Check the get_prisms() method
    """
    spacing = 1
    region = (0, 1, 0, 1)
    layer = prisms_layer(region, spacing=spacing)
    layer["bottom"] = (layer.bottom.dims, np.arange(4).reshape(2, 2))
    layer["top"] = (layer.top.dims, (np.arange(4) + 10).reshape(2, 2))
    expected_prisms = [
        [-0.5, 0.5, -0.5, 0.5, 0, 10],
        [0.5, 1.5, -0.5, 0.5, 1, 11],
        [-0.5, 0.5, 0.5, 1.5, 2, 12],
        [0.5, 1.5, 0.5, 1.5, 3, 13],
    ]
    npt.assert_allclose(expected_prisms, layer.prisms_layer.get_prisms())


def test_prisms_layer_get_prism_by_index():
    """
    Check if the right prism is returned after index
    """
    spacing = 1
    region = (0, 1, 0, 1)
    layer = prisms_layer(region, spacing=spacing)
    layer["bottom"] = (layer.bottom.dims, np.arange(4).reshape(2, 2))
    layer["top"] = (layer.top.dims, (np.arange(4) + 10).reshape(2, 2))
    npt.assert_allclose(
        layer.prisms_layer.get_prism((0, 1)), [0.5, 1.5, -0.5, 0.5, 1, 11]
    )


@pytest.mark.use_numba
def test_prisms_layer_gravity():
    """
    Check the get_prisms() method
    """
    spacing = 1
    region = (0, 1, 0, 1)
    layer = prisms_layer(region, spacing=spacing, properties={"density": 2670})
    layer["bottom"] = (layer.bottom.dims, np.arange(4).reshape(2, 2))
    layer["top"] = (layer.top.dims, (np.arange(4) + 10).reshape(2, 2))
    prisms = [
        [-0.5, 0.5, -0.5, 0.5, 0, 10],
        [0.5, 1.5, -0.5, 0.5, 1, 11],
        [-0.5, 0.5, 0.5, 1.5, 2, 12],
        [0.5, 1.5, 0.5, 1.5, 3, 13],
    ]
    density = np.ones(4) * 2670
    coordinates = (0.5, 0.5, 20)

    for field in ("potential", "g_z"):
        npt.assert_allclose(
            layer.prisms_layer.gravity(coordinates, field=field),
            prism_gravity(coordinates, prisms, density, field=field),
        )
