"""
Test the prism forward modeling code.
"""
import numpy as np
import numpy.testing as npt
import verde as vd

from ..forward.prism import prism_gravity


def test_forward_prism_around_vertical():
    "The vertical gravity component should be consistent around the prism"
    prism = (-300, 300, -300, 300, -300, 300)
    # Make the computation points surround the prism
    region = [-600, 600, -600, 600]
    spacing = 10
    distance = 310
    x, y = vd.grid_coordinates(region, spacing=spacing)
    top = prism
    # Top and bottom should be reversed
    npt.assert_allclose(top, -bottom, 10, 'top and bottom')

    # npt.assert_allclose(north, south, 10, 'Failed gz, north and south')
    # npt.assert_allclose(east, west, 10, 'Failed gz, east and west')
    # npt.assert_allclose(north, prism.gx(xp, yp, zp, model), 10,
                  # 'Failed gz, north and gx')
    # npt.assert_allclose(south, prism.gx(xp, yp, zp, model), 10,
                  # 'Failed gz, south and gx')
    # npt.assert_allclose(east, prism.gy(xp, yp, zp, model), 10,
                  # 'Failed gz, east and gy')
    # npt.assert_allclose(west, prism.gy(xp, yp, zp, model), 10,
                  # 'Failed gz, west and gy')
