from pyscript import display, document
import folium
import json
import pandas as pd
from ast import literal_eval

from pyodide.http import open_url, pyfetch


from PIL import Image
from io import BytesIO
import numpy as np
import asyncio

"""
url = (
    "https://raw.githubusercontent.com/python-visualization/folium/master/examples/data"
)
state_geo = f"{url}/us-states.json"
state_unemployment = f"{url}/US_Unemployment_Oct2012.csv"
state_data = pd.read_csv(open_url(state_unemployment))
geo_json = json.loads(open_url(state_geo).read())
"""


m = folium.Map(location=[48, -102], zoom_start=3, tiles="CartoDB positron")


async def fetch_and_load_image(url):
    # Step 1: Fetch the image using pyfetch
    response = await pyfetch(url, method="GET")

    # Step 2: Read the binary content from the response
    image_data = await response.bytes()

    # Step 3: Convert the binary data to a BytesIO object
    image_stream = BytesIO(image_data)

    # Step 4: Open the image using PIL
    image = Image.open(image_stream)

    # Now you can work with the image object
    return image


async def download_elevation_tile(url):
    """
    see https://github.com/tilezen/joerd/blob/master/docs/formats.md#terrarium for conversion details
    """

    # response = open_url(url).read().encode("latin1")
    # response = pyfetch(url)
    # tile = np.array(Image.open(BytesIO(response.content)), dtype=np.float64)

    tile = await fetch_and_load_image(url)

    # asyncio.gather(fetch_and_load_image(url) ) #up_down(), up_down(), up_down())
    # tile = np.array(Image.open(response), dtype=np.float64)

    return (
        tile  # (tile[:, :, 0] * 256.0 + tile[:, :, 1] + tile[:, :, 2] / 256.0) - 32768
    )


def get_elevation_tile(x, y, zoom_level):

    url_dem = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{y}/{x}.png"

    tile_dem = download_elevation_tile(url_dem.format(z=zoom_level, x=y, y=x))

    return tile_dem


async def submit_bounding_box(event):
    # input_text = document.querySelector("#english")
    # english = input_text.value
    input_text = document.querySelector("#bounding_box")
    output_div = document.querySelector("#output")
    map_element = document.querySelector("#folium")
    print(map_element)

    tile = await get_elevation_tile(0, 0, 0)
    print(tile)
    output_div.innerText = input_text.value

    # bounding_box.bounds = [[37.77, -102.42], [45.78, -110.41]]
    # print(map_element)  # .get_root().removeChild(map_element)
    # lat_min, lon_min, lat_max, lon_max = [
    #    literal_eval(i) for i in input_text.value.split(",")
    # ]

    folium.Rectangle(
        bounds=[[37.77, -122.42], [45.78, -100.41]],
        color="blue",
        fill=True,
        fill_color="blue",
        fill_opacity=0.2,
    ).add_to(m)


"""

# Add the rectangle to the map
bounding_box = folium.Rectangle(
    bounds=[[37.77, -122.42], [45.78, -100.41]],
    color="blue",
    fill=True,
    fill_color="cyan",
).add_to(m)

"""

display(m, target="folium")
