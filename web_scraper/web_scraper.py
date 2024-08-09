import concurrent.futures
import os
import threading

from PIL import Image
import requests
from io import BytesIO

import numpy as np


def download_elevation_tile(url):
    """
    see https://github.com/tilezen/joerd/blob/master/docs/formats.md#terrarium for conversion details
    """
    response = requests.get(url)
    tile = np.array(Image.open(BytesIO(response.content)), dtype=np.float64)

    return (tile[:, :, 0] * 256.0 + tile[:, :, 1] + tile[:, :, 2] / 256.0) - 32768


def get_elevation_tile(x, y, zoom_level):

    url_dem = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{y}/{x}.png"

    tile_dem = download_elevation_tile(url_dem.format(z=zoom_level, x=y, y=x))

    return tile_dem
