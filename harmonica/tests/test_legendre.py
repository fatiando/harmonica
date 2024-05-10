"""
Test the associated Legendre function calculations.
"""
import numpy as np

from .._spherical_harmonics.legendre import pnm


def test_legendre_pnm_unnormalized():
    "Check if the first few degrees match analytical expressions"
    for angle in np.linspace(0, np.pi, 180):
        x = np.cos(angle)
        p = pnm(x, 4)
        # Make sure the upper diagonal is all NaNs
        for n in range(p.shape[0]):
            for m in range(n + 1, p.shape[1]):
                assert np.isnan(p[n, m])
        # Degree 0
        np.testing.assert_allclose(1, p[0][0])
        # Degree 1
        np.testing.assert_allclose(x, p[1][0])
        np.testing.assert_allclose(np.sqrt(1 - x**2), p[1][1], atol=1e-10)
        # Degree 2
        np.testing.assert_allclose(1 / 2 * (3 * x**2 - 1), p[2][0], atol=1e-10)
        np.testing.assert_allclose(3 * x * np.sqrt(1 - x**2), p[2][1], atol=1e-10)
        np.testing.assert_allclose(3 * (1 - x**2), p[2][2], atol=1e-10)
        # Degree 3
        np.testing.assert_allclose(1 / 2 * (5 * x**3 - 3 * x), p[3][0], atol=1e-10)
        np.testing.assert_allclose(
            -3 / 2 * (1 - 5 * x**2) * np.sqrt(1 - x**2), p[3][1], atol=1e-10
        )
        np.testing.assert_allclose(15 * x * (1 - x**2), p[3][2], atol=1e-10)
        np.testing.assert_allclose(15 * np.sqrt(1 - x**2) ** 3, p[3][3], atol=1e-10)
        # Degree 4
        np.testing.assert_allclose(
            1 / 8 * (35 * x**4 - 30 * x**2 + 3), p[4][0], atol=1e-10
        )
        np.testing.assert_allclose(
            5 / 2 * (7 * x**3 - 3 * x) * np.sqrt(1 - x**2), p[4][1], atol=1e-10
        )
        np.testing.assert_allclose(
            15 / 2 * (7 * x**2 - 1) * (1 - x**2), p[4][2], atol=1e-10
        )
        np.testing.assert_allclose(105 * x * np.sqrt(1 - x**2) ** 3, p[4][3], atol=1e-10)
        np.testing.assert_allclose(105 * np.sqrt(1 - x**2) ** 4, p[4][4], atol=1e-10)
