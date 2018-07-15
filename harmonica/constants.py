r"""
Definitions of commonly used physical constants. Uses :mod:`unyt` to assign units to the
constants.

======================== =============== ========================================
Name                     Symbol          Description
------------------------ --------------- ----------------------------------------
GRAVITATIONAL_CONST      :math:`G`       The gravitational constant.
======================== =============== ========================================

"""
from unyt import m, kg, s


GRAVITATIONAL_CONST = 0.00000000006673*m**3/(kg*s)
