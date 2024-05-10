"""
Calculation of associated Legendre functions and their derivatives.
"""
import numba
import numpy as np
import math


def pnm(x, max_degree):
    """
    Calculate the associated Legendre functions Pnm until a maximum degree.

    The functions :math:`P_n^m` are solutions to Legendre's equation. We
    calculate their values using the recursive relations defined in
    Alken (2022). This function does not include the Condon-Shortly phase.

    Parameters
    ----------
    x : float
        The argument of :math:`P_n^m(x)`. Must be in the range [-1, 1].
    max_degree : int
        The maximum degree for the calculation.


    Returns
    -------
    p : 2D numpy.array
        Array with the values of :math:`P_n^m(x)` with shape
        ``(max_degree + 1, max_degree + 1)``. The degree n varies with the
        first axis and the order m varies with the second axis. For example,
        :math:`P_2^1(x)` is ``p[n, m]``. Array values where ``m > n`` are set
        to ``numpy.nan``.

    References
    ----------

    Alken, Patrick (2022). GSL Technical Report #1 - GSL-TR-001-20220827 -
      Implementation of associated Legendre functions in GSL.
      https://www.gnu.org/software/gsl/tr/tr001.pdf
    """
    p = np.full((max_degree + 1, max_degree + 1), np.nan)
    sqrt_x = np.sqrt(1 - x**2)
    p[0, 0] = 1
    p[1, 0] = x
    p[1, 1] = sqrt_x
    _pnm_unnormalized(x, sqrt_x, max_degree, p)
    return p


@numba.jit(nopython=True)
def _pnm_unnormalized(x, sqrt_x, max_degree, p):
    "Optimized version of the unnormalized calculation."
    for n in range(2, max_degree + 1):
        for m in range(0, n - 1):
            p[n, m] = ((2 * n - 1) * x * p[n-1, m] - (n + m - 1) * p[n-2, m]) / (n - m)
        m = n - 1
        p[n][m] = (2 * n - 1) / (n - m) * x * p[n - 1, m]
        p[n][n] = (2 * n - 1) * sqrt_x * p[n - 1, n - 1]


def schmidt_normalization(max_degree):
    """
    Calculate the Schmidt normalization factor for geomagnetic field models
    """
    s = np.full((max_degree + 1, max_degree + 1), np.nan)
    for n in range(max_degree + 1):
        for m in range(n + 1):
            if m == 0:
                s[n, m] = 1
            else:
                s[n, m] = np.sqrt(
                    2 * math.factorial(n - m) / math.factorial(n + m)
                )
    return s
