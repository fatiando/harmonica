"""
Test prisms layer
"""
import pytest
import numpy as np
import numpy.testing as npt

from .. import prisms_layer


def test_prisms_layer():
    """
    Check if a layer of prisms is property constructed
    """
    easting = np.linspace(-1, 3, 5)
    northing = np.linspace(7, 10, 4)
    reference = 0
    surface = np.arange(20).reshape(4, 5)
    layer = prisms_layer((easting, northing), surface, reference)
    assert "easting" in layer.coords
    assert "northing" in layer.coords
    assert "top" in layer.coords
    assert "bottom" in layer.coords
    npt.assert_allclose(layer.easting, easting)
    npt.assert_allclose(layer.northing, northing)
    npt.assert_allclose(layer.top, surface)
    npt.assert_allclose(layer.bottom, reference)
    # Surface below reference on a single point
    surface[1, 1] = -1
    expected_top = surface.copy()
    expected_bottom = reference * np.ones_like(surface)
    expected_top[1, 1], expected_bottom[1, 1] = reference, surface[1, 1]
    layer = prisms_layer((easting, northing), surface, reference)
    assert "easting" in layer.coords
    assert "northing" in layer.coords
    assert "top" in layer.coords
    assert "bottom" in layer.coords
    npt.assert_allclose(layer.easting, easting)
    npt.assert_allclose(layer.northing, northing)
    npt.assert_allclose(layer.top, expected_top)
    npt.assert_allclose(layer.bottom, expected_bottom)


def test_prisms_layer_invalid_surface_reference():
    """
    Check if invalid surface and/or reference are caught
    """
    coordinates = np.linspace(-1, 3, 5), np.linspace(7, 10, 4)
    # Surface with wrong shape
    reference = 0
    surface = np.arange(20)
    with pytest.raises(ValueError):
        prisms_layer(coordinates, surface, reference)
    # Reference with wrong shape
    reference = np.zeros(20)
    surface = np.arange(20).reshape(4, 5)
    with pytest.raises(ValueError):
        prisms_layer(coordinates, surface, reference)


def test_prisms_layer_properties():
    """
    Check passing physical properties to the prisms layer
    """
    easting = np.linspace(-1, 3, 5)
    northing = np.linspace(7, 10, 4)
    reference = 0
    surface = np.arange(20).reshape(4, 5)
    density = 2670 * np.ones_like(surface)
    suceptibility = np.arange(20).reshape(4, 5)
    layer = prisms_layer(
        (easting, northing),
        surface,
        reference,
        properties={"density": density, "suceptibility": suceptibility},
    )
    npt.assert_allclose(layer.density, density)
    npt.assert_allclose(layer.suceptibility, suceptibility)


def test_prisms_layer_methods():
    """
    Check methods of the DatasetAccessorPrismsLayer class
    """
    easting = np.linspace(1, 3, 5)
    northing = np.linspace(7, 10, 4)
    reference = 0
    surface = np.arange(20).reshape(4, 5)
    layer = prisms_layer((easting, northing), surface, reference)
    assert layer.prisms_layer.dims == ("northing", "easting")
    assert layer.prisms_layer.spacing == (1, 0.5)
    assert layer.prisms_layer.boundaries == (
        easting[0] - 0.25,
        easting[-1] + 0.25,
        northing[0] - 0.5,
        northing[-1] + 0.5,
    )
    assert layer.prisms_layer.size == 20
    assert layer.prisms_layer.shape == (4, 5)


def test_prisms_layer_get_prisms():
    """
    Check the get_prisms() method
    """
    coordinates = (np.array([0, 1]), np.array([0, 1]))
    reference = np.arange(4).reshape(2, 2)
    surface = (np.arange(4) + 10).reshape(2, 2)
    layer = prisms_layer(coordinates, surface, reference)
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
    coordinates = (np.array([0, 1]), np.array([0, 1]))
    reference = np.arange(4).reshape(2, 2)
    surface = (np.arange(4) + 10).reshape(2, 2)
    layer = prisms_layer(coordinates, surface, reference)
    expected_prisms = [
        [[-0.5, 0.5, -0.5, 0.5, 0, 10], [0.5, 1.5, -0.5, 0.5, 1, 11]],
        [[-0.5, 0.5, 0.5, 1.5, 2, 12], [0.5, 1.5, 0.5, 1.5, 3, 13]],
    ]
    for i in range(2):
        for j in range(2):
            npt.assert_allclose(
                layer.prisms_layer.get_prism((i, j)), expected_prisms[i][j]
            )
