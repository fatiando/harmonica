# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Utilities for forward modelling of rectangular prisms.
"""


def check_prisms(prisms):
    """
    Check if prisms boundaries are well defined.

    Parameters
    ----------
    prisms : 2d-array
        Array containing the boundaries of the prisms in the following order:
        ``w``, ``e``, ``s``, ``n``, ``bottom``, ``top``.
        The array must have the following shape: (``n_prisms``, 6), where
        ``n_prisms`` is the total number of prisms.
    """
    west, east, south, north, bottom, top = tuple(prisms[:, i] for i in range(6))
    err_msg = "Invalid prism or prisms. "
    bad_we = west > east
    bad_sn = south > north
    bad_bt = bottom > top
    if bad_we.any():
        err_msg += "The west boundary can't be greater than the east one.\n"
        for prism in prisms[bad_we]:
            err_msg += f"\tInvalid prism: {prism}\n"
        raise ValueError(err_msg)
    if bad_sn.any():
        err_msg += "The south boundary can't be greater than the north one.\n"
        for prism in prisms[bad_sn]:
            err_msg += f"\tInvalid prism: {prism}\n"
        raise ValueError(err_msg)
    if bad_bt.any():
        err_msg += "The bottom boundary can't be greater than the top one.\n"
        for prism in prisms[bad_bt]:
            err_msg += f"\tInvalid tesseroid: {prism}\n"
        raise ValueError(err_msg)
