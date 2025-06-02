# Copyright (c) 2018 The Harmonica Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Script to use the BGS API to get values of the IGRF for testing purposes.

"""

import datetime
import json

import numpy as np
import requests
import verde as vd
import xarray as xr

coordinates = vd.grid_coordinates((0, 359, -90, 90), spacing=10, extra_coords=0)

dates = ["1930-04-20", "1986-07-29", "2024-01-10", "2029-10-30"]

shape = (len(dates), *coordinates[0].shape)
bx = np.empty(shape)
by = np.empty(shape)
bz = np.empty(shape)

url = (
    "https://geomag.bgs.ac.uk/web_service/GMModels/igrf/14/?"
    "latitude={latitude}&longitude={longitude}&altitude={altitude}&date={date}&"
    "format=json"
)

for k in range(shape[0]):
    for i in range(shape[1]):
        for j in range(shape[2]):
            response = requests.get(
                url.format(
                    longitude=coordinates[0][i, j],
                    latitude=coordinates[1][i, j],
                    altitude=coordinates[2][i, j],
                    date=dates[k],
                )
            )
            response.raise_for_status()
            results = json.loads(response.text)
            field = results["geomagnetic-field-model-result"]["field-value"]
            bx[k, i, j] = field["east-intensity"]["value"]
            by[k, i, j] = field["north-intensity"]["value"]
            # The BGS gives vertical downward
            bz[k, i, j] = -field["vertical-intensity"]["value"]

np.save("bx.npy", bx)
np.save("by.npy", by)
np.save("bz.npy", bz)

bx = np.load("bx.npy")
by = np.load("by.npy")
bz = np.load("bz.npy")

time = [datetime.datetime.fromisoformat(d) for d in dates]
dims = ("time", "latitude", "longitude")
grid = xr.Dataset(
    {"bx": (dims, bx), "by": (dims, by), "bz": (dims, bz)},
    coords={
        "time": time,
        "longitude": coordinates[0][0, :],
        "latitude": coordinates[1][:, 0],
    },
)
grid.to_netcdf("igrf14-bgs.nc")
