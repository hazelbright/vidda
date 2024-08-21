from pyscript import display, document
import io
from ast import literal_eval
import gc


from pyodide.http import pyfetch
from pyodide.ffi import to_js


from PIL import Image
from io import BytesIO
import numpy as np
import asyncio
import trimesh

import openeo

import zipfile


from js import File, URL, Blob, Uint8Array, ReadableStream


# m = folium.Map(location=[48, -102], zoom_start=3, tiles="CartoDB positron")


async def fetch_and_load_texture(x, y, zoom_level):
    headers = {
        "cache-control": "max-age=0",
        "sec-ch-ua": '" Not A;Brand";v="99", "Chromium";v="99", "Google Chrome";v="99"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest": "document",
        "sec-fetch-mode": "navigate",
        "sec-fetch-site": "none",
        "sec-fetch-user": "?1",
        "upgrade-insecure-requests": "1",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.82 Safari/537.36",
    }
    url_tex = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}".format(
        z=zoom_level, y=y, x=x
    )

    # url_tex = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
    # url_tex = "http://cawm.lib.uiowa.edu/tiles/{z}/{x}/{y}.png"
    # url_tex = "http://server.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile//{z}/{y}/{x}"
    # url_tex = "http://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}"
    # url_tex = "http://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}"
    response = await pyfetch(url_tex, method="GET")  # , headers=headers)

    image_data = await response.bytes()

    image_stream = BytesIO(image_data)

    image = Image.open(image_stream)

    # print(image)

    return np.array(image)[:, :, [2, 1, 0]]  # .swapaxes(0,1)


async def download_texture_tile(x, y, z):
    tile_tex = await fetch_and_load_texture(x, y, z)

    return (x, y, tile_tex)


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

    tile = await fetch_and_load_image(url)

    tile = np.array(tile, dtype=np.uint8)  # np.float64)

    red = tile[:, :, 0].astype(np.float64)
    green = tile[:, :, 1].astype(np.float64)
    blue = tile[:, :, 2].astype(np.float64)

    # Combine the channels into elevation values
    # elevation = (red * 256.0 + green + blue / 256.0) - 32768.0

    # return (tile[:, :, 0] * 256.0 + tile[:, :, 1] + tile[:, :, 2] / 256.0) - 32768
    return (red * 256.0 + green + blue / 256.0) - 32768.0
    # return eleva


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
def heightmap_to_mesh(
    heightmap, dx=0.1, dy=0.1, x_global_offset=0, y_global_offset=0
):  # , x_global_offset=0, y_global_offset=0):
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
    # print(dx * x_global_offset, dy * y_global_offset)
    # xs = np.linspace(0, dx * (n_cols - 1), n_cols) + dx * (y_global_offset * 256 - 1)
    # ys = np.linspace(0, dy * (n_rows - 1), n_rows) + dy * (x_global_offset * 256 - 1)

    xs = np.linspace(0, dx * n_cols, n_cols + 1)[:-1] + dx * (y_global_offset * 256)
    ys = np.linspace(0, dy * n_rows, n_rows + 1)[:-1] - dy * (x_global_offset * 256)

    xs, ys = np.meshgrid(xs, ys)

    # Flatten the arrays and create vertices
    vertices = np.column_stack(
        (
            xs.flatten(),
            heightmap.flatten(),
            ys.flatten(),
        )
    )

    n_rows, n_cols = heightmap.shape

    idx = np.arange(n_rows * n_cols).reshape((n_rows, n_cols))

    # Compute the indices of the vertices that form the faces
    idx_bl = idx[:-1, :-1].ravel()  # Bottom-left vertices
    idx_br = idx[:-1, 1:].ravel()  # Bottom-right vertices
    idx_tl = idx[1:, :-1].ravel()  # Top-left vertices
    idx_tr = idx[1:, 1:].ravel()  # Top-right vertices

    # Stack the indices to create faces
    # First triangle: [idx_tr, idx_br, idx_bl]
    faces_1 = np.stack([idx_tr, idx_br, idx_bl], axis=1)

    # Second triangle: [idx_tl, idx_tr, idx_bl]
    faces_2 = np.stack([idx_tl, idx_tr, idx_bl], axis=1)

    # Combine both triangles
    faces = np.vstack([faces_1, faces_2])

    faces = np.concatenate(
        [faces[: int(len(faces) / 2)], faces[int(len(faces) / 2) :]], axis=-1
    ).reshape(-1, 3)

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
    # input_text = document.querySelector("#bounding_box")
    # output_div = document.querySelector("#output")

    sw_input = document.querySelector(".sw")
    ne_input = document.querySelector(".ne")

    # print(sw_input.value, ne_input.value)
    sw0, sw1 = [literal_eval(i) for i in sw_input.value.split(",")]
    ne0, ne1 = [literal_eval(i) for i in ne_input.value.split(",")]

    bounding_box = [ne0, ne1, sw0, sw1]

    x_range, y_range = get_ranges(bounding_box, 11)

    print(x_range, y_range)

    meshing_tasks = []

    for x_chunk in np.array_split(x_range, int(len(x_range))):
        for y_chunk in np.array_split(y_range, int(len(y_range))):
            x_global_offset = x_chunk[0] - x_range[0]
            y_global_offset = y_chunk[0] - y_range[0]

            x_chunk_ = np.arange(x_chunk[0], x_chunk[-1] + 2)
            y_chunk_ = np.arange(y_chunk[0], y_chunk[-1] + 2)

            meshing_tasks.append(
                construct_mesh(
                    x_chunk_,
                    y_chunk_,
                    x_global_offset=x_global_offset,
                    y_global_offset=y_global_offset,
                )
            )

    meshes = await asyncio.gather(*meshing_tasks)

    if False:
        # process the whole shabang

        scene = trimesh.Scene(meshes)

        del meshes
        gc.collect()

        file = File.new(
            [Uint8Array.new(scene.export(file_type="glb"))],
            "generated_area.glb",
            {type: "model/gltf-binary"},
        )
        del scene
        gc.collect()

        url = URL.createObjectURL(file)

        hidden_link = document.createElement("a")
        hidden_link.setAttribute("download", "generated_area.glb")
        hidden_link.setAttribute("href", url)
        hidden_link.click()
    if False:
        # Create an in-memory ZIP file
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            # Add OBJ and MTL files to the ZIP
            i = 0
            while meshes:
                mesh = meshes.pop(0)
                # for i in range(len(meshes)):
                # obj_data = mesh.export(file_type="obj")
                zip_file.writestr("area_%i.glb" % i, mesh.export(file_type="glb"))
                i += 1

                del mesh
                gc.collect()

        print("Zipfile created")

        # Prepare the ZIP file for download
        zip_buffer.seek(0)

        zip_bytes = zip_buffer.getvalue()

        # Convert the bytes to a JavaScript Uint8Array
        uint8_array = Uint8Array.new(list(zip_bytes))

        # Create a Blob from the Uint8Array
        zip_blob = Blob.new([uint8_array], {"type": "application/zip"})

        zip_url = URL.createObjectURL(zip_blob)

        # Triggering the download by creating a hidden link and clicking it
        hidden_link = document.createElement("a")
        hidden_link.setAttribute("download", "model_files.zip")
        hidden_link.setAttribute("href", zip_url)
        hidden_link.click()
    if True:

        def create_zip_in_memory(meshes):
            # Create an in-memory bytes buffer
            zip_buffer = io.BytesIO()

            # Create a zip file in the buffer
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                i = 0
                while meshes:
                    mesh = meshes.pop(0)
                    # Add each mesh as a GLB file in the zip
                    zip_file.writestr(f"area_{i}.glb", mesh.export(file_type="glb"))
                    i += 1

                    del mesh
                    gc.collect()

            # Ensure everything is written to the buffer
            zip_file.close()

            # Move to the beginning of the buffer to prepare for reading
            zip_buffer.seek(0)
            return zip_buffer

        def stream_zip_content(zip_buffer, chunk_size=1024 * 64):
            while True:
                chunk = zip_buffer.read(chunk_size)
                if not chunk:
                    break
                yield to_js(Uint8Array.new(list(chunk)))

        def download_zip(meshes):
            # Create the zip in memory
            zip_buffer = create_zip_in_memory(meshes)

            # Create a generator to stream the zip content
            zip_stream = stream_zip_content(zip_buffer)

            # Create a Blob from the generator
            zip_blob = Blob.new(zip_stream, {"type": "application/zip"})

            # Create a download URL for the blob
            zip_url = URL.createObjectURL(zip_blob)

            # Trigger the download by creating a hidden link and clicking it
            hidden_link = document.createElement("a")
            hidden_link.setAttribute(
                "download", "%.5f_%.5f_%.5f_%.5f.zip" % tuple(bounding_box)
            )
            hidden_link.setAttribute("href", zip_url)
            hidden_link.click()

        # Assuming 'meshes' is the list of mesh objects
        download_zip(meshes)


async def construct_mesh(x_range, y_range, x_global_offset, y_global_offset):
    # input_text = document.querySelector("#english")

    tasks_elevation = []
    tasks_texture = []

    # tile_tex = await download_texture_tile(0, 0, 0)
    # print(tile_tex)

    for i in x_range:
        for j in y_range:
            tasks_elevation.append(get_elevation_tile(i, j, 11))
            tasks_texture.append(download_texture_tile(i, j, 11))

            # print(i, j, tile)

    results_elevation = await asyncio.gather(*tasks_elevation)

    print("Data downloaded")

    update_progress(10)

    full_elevation_image = np.zeros(
        (256 * x_range.shape[0], 256 * y_range.shape[0]), dtype=float
    )

    # parse results
    for result in results_elevation:
        i, j, tile = result

        full_elevation_image[
            (i - x_range[0]) * 256 : (i - x_range[0] + 1) * 256,
            (j - y_range[0]) * 256 : (j - y_range[0] + 1) * 256,
        ] = np.array(tile).T

    # full_elevation_image = full_elevation_image[::-1]

    full_elevation_image = full_elevation_image[
        : full_elevation_image.shape[0] - 255,
        : full_elevation_image.shape[1] - 255,
    ]

    del results_elevation
    gc.collect()

    update_progress(20)

    results_texture = await asyncio.gather(*tasks_texture)

    full_texture_image = np.zeros(
        (256 * x_range.shape[0], 256 * y_range.shape[0], 3), dtype=np.uint8
    )

    for result in results_texture:
        i, j, tile = result

        full_texture_image[
            (i - x_range[0]) * 256 : (i - x_range[0] + 1) * 256,
            (j - y_range[0]) * 256 : (j - y_range[0] + 1) * 256,
        ] = np.array(tile).swapaxes(0, 1)[
            :, :, [2, 1, 0]
        ]  # .T

    # full_texture_image = full_texture_image[::-1]

    full_texture_image = full_texture_image[
        : full_texture_image.shape[0] - 255, : full_texture_image.shape[1] - 255
    ]

    # full_texture_image = full_texture_image[::-1]
    full_elevation_image = full_elevation_image[::-1]

    del results_texture
    gc.collect()

    # full_elevation_image -= np.min(full_elevation_image)
    # full_elevation_image *= full_elevation_image.max() ** -1
    full_elevation_image *= 0.005

    mesh = heightmap_to_mesh(
        full_elevation_image,
        x_global_offset=x_global_offset,
        y_global_offset=y_global_offset,
    )  # np.random.uniform(-100, 100, full_image.shape))

    del full_elevation_image
    gc.collect()

    print("Mesh computed")

    # Generate a random RGB texture image
    texture_pil = Image.fromarray(full_texture_image)  # [::-1])

    del full_texture_image
    gc.collect()

    uv_coordinates = mesh.vertices[
        :, [0, 2]
    ]  # Use x and y components of vertices for UV

    uv_min = uv_coordinates.min(axis=0)
    uv_max = uv_coordinates.max(axis=0)
    uv = (uv_coordinates - uv_min) / (uv_max - uv_min)
    # im = Image.open("image.png")
    material = trimesh.visual.texture.SimpleMaterial(image=texture_pil, glossiness=None)

    del texture_pil
    gc.collect()

    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)

    return mesh


async def submit_bounding_box_(event):
    # input_text = document.querySelector("#english")

    # english = input_text.value
    input_text = document.querySelector("#bounding_box")
    # output_div = document.querySelector("#output")

    sw_input = document.querySelector(".sw")
    ne_input = document.querySelector(".ne")

    print(sw_input.value, ne_input.value)
    sw0, sw1 = [literal_eval(i) for i in sw_input.value.split(",")]
    ne0, ne1 = [literal_eval(i) for i in ne_input.value.split(",")]

    bounding_box = [ne0, ne1, sw0, sw1]

    x_range, y_range = get_ranges(bounding_box, 11)
    print("x_range:", x_range)
    print("y_range:", y_range)

    tasks_elevation = []
    tasks_texture = []

    # tile_tex = await download_texture_tile(0, 0, 0)
    # print(tile_tex)

    for i in x_range:
        for j in y_range:
            tasks_elevation.append(get_elevation_tile(i, j, 11))
            tasks_texture.append(download_texture_tile(i, j, 11))

            # print(i, j, tile)

    results_elevation = await asyncio.gather(*tasks_elevation)

    print("Data downloaded")
    update_progress(10)

    full_elevation_image = np.zeros(
        (256 * x_range.shape[0], 256 * y_range.shape[0]), dtype=float
    )

    # parse results
    for result in results_elevation:
        i, j, tile = result

        full_elevation_image[
            (i - x_range[0]) * 256 : (i - x_range[0] + 1) * 256,
            (j - y_range[0]) * 256 : (j - y_range[0] + 1) * 256,
        ] = np.array(tile).T

    del results_elevation
    gc.collect()

    update_progress(20)

    results_texture = await asyncio.gather(*tasks_texture)

    full_texture_image = np.zeros(
        (256 * x_range.shape[0], 256 * y_range.shape[0], 3), dtype=np.uint8
    )

    for result in results_texture:
        i, j, tile = result

        full_texture_image[
            (i - x_range[0]) * 256 : (i - x_range[0] + 1) * 256,
            (j - y_range[0]) * 256 : (j - y_range[0] + 1) * 256,
        ] = np.array(tile).swapaxes(0, 1)[
            :, :, [2, 1, 0]
        ]  # .T

    del results_texture
    gc.collect()

    # full_elevation_image -= np.min(full_elevation_image)
    # full_elevation_image *= full_elevation_image.max() ** -1
    full_elevation_image *= full_elevation_image.shape[0] * 0.0002

    mesh = heightmap_to_mesh(
        full_elevation_image
    )  # np.random.uniform(-100, 100, full_image.shape))

    del full_elevation_image
    gc.collect()

    print("Mesh computed")

    # Generate a random RGB texture image
    texture_pil = Image.fromarray(full_texture_image[::-1])

    del full_texture_image
    gc.collect()

    uv_coordinates = mesh.vertices[
        :, [0, 2]
    ]  # Use x and y components of vertices for UV

    uv_min = uv_coordinates.min(axis=0)
    uv_max = uv_coordinates.max(axis=0)
    uv = (uv_coordinates - uv_min) / (uv_max - uv_min)
    # im = Image.open("image.png")
    material = trimesh.visual.texture.SimpleMaterial(image=texture_pil, glossiness=None)

    del texture_pil
    gc.collect()

    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)

    print("Mesh constructed")

    # glb_data = mesh.export(file_type="glb")

    print("Generated binary data for download ")
    file = File.new(
        [Uint8Array.new(mesh.export(file_type="glb"))],
        "generated_area.glb",
        {type: "model/gltf-binary"},
    )
    del mesh
    gc.collect()

    url = URL.createObjectURL(file)

    hidden_link = document.createElement("a")
    hidden_link.setAttribute("download", "generated_area.glb")
    hidden_link.setAttribute("href", url)
    hidden_link.click()


def update_progress(value):
    progress_bar = document.getElementById("progress-bar")
    progress_bar.value = value
