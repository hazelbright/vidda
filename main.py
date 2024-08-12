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
import trimesh

"""
url = (
    "https://raw.githubusercontent.com/python-visualization/folium/master/examples/data"
)
state_geo = f"{url}/us-states.json"
state_unemployment = f"{url}/US_Unemployment_Oct2012.csv"
state_data = pd.read_csv(open_url(state_unemployment))
geo_json = json.loads(open_url(state_geo).read())
"""

from js import Uint8Array, File, URL


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
    return np.array(image, dtype=np.float64)


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

    return (tile[:, :, 0] * 256.0 + tile[:, :, 1] + tile[:, :, 2] / 256.0) - 32768


async def get_elevation_tile(x, y, zoom_level):

    url_dem = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"

    tile_dem = await download_elevation_tile(url_dem.format(z=zoom_level, x=x, y=y))

    return (x, y, tile_dem)


def ll2wms(lat, lon, zoom_level):
    """Convert from lat/lon to wms tile coordinate"""
    n_tiles = 2**zoom_level

    lon_index = n_tiles * (lon + 180) / 360

    lat_rad = np.radians(lat)
    lat_index = (
        n_tiles * (1 - (np.log(np.tan(lat_rad) + (1 / np.cos(lat_rad))) / np.pi)) / 2
    )

    return lat_index, lon_index


def get_ranges(bbox_ll, zoom_level):

    y0, x0 = ll2wms(bbox_ll[0], bbox_ll[1], zoom_level=zoom_level)
    y1, x1 = ll2wms(bbox_ll[2], bbox_ll[3], zoom_level=zoom_level)

    x_range = np.arange(np.floor(min(x0, x1)), np.floor(max(x0, x1)) + 1, dtype=int)
    y_range = np.arange(np.floor(min(y0, y1)), np.floor(max(y0, y1)) + 1, dtype=int)

    return x_range, y_range


# 2. Generate Vertices and Faces
def heightmap_to_mesh(heightmap, dx=1.0, dy=1.0):
    """
    Converts a heightmap to a 3D mesh.

    Parameters:
    - heightmap: 2D numpy array of heights.
    - dx: Spacing between points in the x-direction.
    - dy: Spacing between points in the y-direction.

    Returns:
    - mesh: Trimesh object representing the 3D terrain.
    """
    n_rows, n_cols = heightmap.shape
    # Create grid of (x, y) coordinates
    xs = np.linspace(0, dx * (n_cols - 1), n_cols)
    ys = np.linspace(0, dy * (n_rows - 1), n_rows)
    xs, ys = np.meshgrid(xs, ys)

    # Flatten the arrays and create vertices
    vertices = np.column_stack(
        (
            xs.flatten(),
            heightmap.flatten(),
            ys.flatten(),
        )
    )

    # Create faces
    faces = []
    for i in range(n_rows - 1):
        for j in range(n_cols - 1):
            # Indices of the four corners of the quad
            idx_bl = i * n_cols + j
            idx_br = idx_bl + 1
            idx_tl = idx_bl + n_cols
            idx_tr = idx_tl + 1

            # First triangle of the quad
            # faces.append([idx_bl, idx_br, idx_tr])
            # Second triangle of the quad
            # faces.append([idx_bl, idx_tr, idx_tl])

            # First triangle of the quad
            faces.append([idx_tr, idx_br, idx_bl])
            # Second triangle of the quad
            faces.append([idx_tl, idx_tr, idx_bl])

    faces = np.array(faces)

    # Create the mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh


# Wrap the synchronous function in an async function
async def async_get_data(i, j):
    # Call the synchronous function in a thread to avoid blocking the event loop
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, get_elevation_tile, i, j, 11)
    return (i, j, result)


# Function to run the downloads asynchronously
async def download_data(indices):
    # Create a list of tasks
    tasks = []
    for i, j in indices:
        tasks.append(async_get_data(i, j))

    # Run the tasks concurrently
    results = await asyncio.gather(*tasks)

    # Return the results with their corresponding indices
    return results


async def submit_bounding_box(event):
    # input_text = document.querySelector("#english")
    # english = input_text.value
    input_text = document.querySelector("#bounding_box")
    output_div = document.querySelector("#output")
    # map_element = document.querySelector("#folium")
    # print(map_element)

    # if False:

    output_div.innerText = input_text.value

    # bounding_box.bounds = [[37.77, -102.42], [45.78, -110.41]]
    # print(map_element)  # .get_root().removeChild(map_element)

    bounding_box = [literal_eval(i) for i in input_text.value.split(",")]

    x_range, y_range = get_ranges(bounding_box, 11)
    print("x_range:", x_range)
    print("y_range:", y_range)

    tasks = []

    for i in x_range:
        for j in y_range:
            tasks.append(get_elevation_tile(i, j, 11))

            # print(i, j, tile)

    results = await asyncio.gather(*tasks)

    full_image = np.zeros((256 * x_range.shape[0], 256 * y_range.shape[0]), dtype=float)

    # parse results
    for result in results:
        i, j, tile = result

        full_image[
            (i - x_range[0]) * 256 : (i - x_range[0] + 1) * 256,
            (j - y_range[0]) * 256 : (j - y_range[0] + 1) * 256,
        ] = np.array(tile).T

    print(full_image)

    # print(results)

    if False:
        # results = await download_data(
        #    np.array(np.meshgrid(x_range, y_range)).reshape(-1, 2)
        # )

        results = await async_get_data(0, 0)

        print(results)

        folium.Rectangle(
            bounds=[[37.77, -122.42], [45.78, -100.41]],
            color="blue",
            fill=True,
            fill_color="blue",
            fill_opacity=0.2,
        ).add_to(m)

    if True:  # generate mesh
        vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        uv_coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])  # Texture coordinates

        # Load the texture image (your image array should be in RGB format)
        # texture_image = Image.open('your_image.png')

        # texture_image = Image.fromarray(np.uint8(full_image))

        # Convert the PIL image to a numpy array
        # texture = np.array(texture_image)

        # Create a Trimesh object
        # mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        full_image -= np.min(full_image)
        full_image *= full_image.max() ** -1
        full_image *= full_image.shape[0] * 0.1

        # print(full_image.max(), full_image.min())

        mesh = heightmap_to_mesh(
            full_image
        )  # np.random.uniform(-100, 100, full_image.shape))

        # Apply texture using visual.TextureVisuals
        # mesh.visual = trimesh.visual.TextureVisuals(uv=uv_coords, image=full_image)

        # print(mesh)
        file = File.new(
            [mesh.export(file_type="obj")], "generated_area.obj", {type: "model/obj"}
        )
        url = URL.createObjectURL(file)

        hidden_link = document.createElement("a")
        hidden_link.setAttribute("download", "generated_area.obj")
        hidden_link.setAttribute("href", url)
        hidden_link.click()

        # Save the mesh as a .obj file with the texture
        # mesh.export("textured_mesh.obj")


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

# 40,-74.5, 40.3, -74.2
