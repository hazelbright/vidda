from pyscript import display, document
import folium
import io
from ast import literal_eval
import gc

from pyodide.http import pyfetch


from PIL import Image
from io import BytesIO
import numpy as np
import asyncio
import trimesh

import zipfile

"""
url = (
    "https://raw.githubusercontent.com/python-visualization/folium/master/examples/data"
)
state_geo = f"{url}/us-states.json"
state_unemployment = f"{url}/US_Unemployment_Oct2012.csv"
state_data = pd.read_csv(open_url(state_unemployment))
geo_json = json.loads(open_url(state_geo).read())
"""

from js import File, URL, Blob, Uint8Array


m = folium.Map(location=[48, -102], zoom_start=3, tiles="CartoDB positron")


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

    # response = open_url(url).read().encode("latin1")
    # response = pyfetch(url)
    # tile = np.array(Image.open(BytesIO(response.content)), dtype=np.float64)

    tile = await fetch_and_load_image(url)

    tile = np.array(tile, dtype=np.float64)

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

    """
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
    """

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

    # faces

    # Create the mesh

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # mesh.visual = trimesh.visual.TextureVisuals(
    ##    uv=np.array([xs, ys]).reshape(2, -1).T,
    #    image=np.random.randint(0, 255, (3, n_rows, n_cols)),
    # )

    # Create UV coordinates - they should be normalized to [0, 1] range
    uv_x = np.linspace(0, 1, n_cols)
    uv_y = np.linspace(0, 1, n_rows)
    uv_xv, uv_yv = np.meshgrid(uv_x, uv_y)
    uv_coords = np.column_stack((uv_xv.flatten(), uv_yv.flatten()))

    # Generate a random RGB texture image
    texture_image = Image.fromarray(
        np.random.randint(230, 255, (n_rows, n_cols, 3), dtype=np.uint8)
    )

    def get_texture(my_uvs, img):
        # img is PIL Image
        uvs = my_uvs
        material = trimesh.visual.texture.SimpleMaterial(image=img)
        texture = trimesh.visual.TextureVisuals(uv=uvs, image=img, material=material)
        return texture

    # texture_visual = get_texture(uv_coords, texture_image)

    # mesh.visual = trimesh.visual.TextureVisuals(
    #    uv=uv_coords, image=Image.fromarray(texture_image)
    # )

    # Assign texture to the mesh using TextureVisuals
    # mesh.visual = texture_visual  # trimesh.visual.TextureVisuals(uv=uv_coords, image=texture_image)

    # mesh.visual = trimesh.visual.TextureVisuals(uv=uv_coords, image=texture)

    return mesh  # , texture_visual


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

    print("Data combined")

    # print(full_image)

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
        # vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        # faces = np.array([[0, 1, 2], [0, 2, 3]])
        # uv_coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])  # Texture coordinates

        # Load the texture image (your image array should be in RGB format)
        # texture_image = Image.open('your_image.png')

        # texture_image = Image.fromarray(np.uint8(full_image))

        # Convert the PIL image to a numpy array
        # texture = np.array(texture_image)

        # Create a Trimesh object
        # mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        print(
            "elevation range:",
            np.max(full_elevation_image),
            np.min(full_elevation_image),
        )
        # full_elevation_image -= np.min(full_elevation_image)
        # full_elevation_image *= full_elevation_image.max() ** -1
        full_elevation_image *= full_elevation_image.shape[0] * 0.0001

        # print(full_image.max(), full_image.min())

        mesh = heightmap_to_mesh(
            full_elevation_image
        )  # np.random.uniform(-100, 100, full_image.shape))

        del full_elevation_image
        gc.collect()

        print("Mesh computed")

        # Apply texture using visual.TextureVisuals
        # mesh.visual = trimesh.visual.TextureVisuals(uv=uv_coords, image=full_image)

        # print(mesh)

        if False:

            file = File.new(
                [mesh.export(file_type="obj")],
                "generated_area.obj",
                {type: "model/obj"},
            )

            url = URL.createObjectURL(file)

            hidden_link = document.createElement("a")
            hidden_link.setAttribute("download", "generated_area.obj")
            hidden_link.setAttribute("href", url)
            hidden_link.click()

        if True:

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
            material = trimesh.visual.texture.SimpleMaterial(
                image=texture_pil, glossiness=None
            )

            del texture_pil
            gc.collect()

            # material = trimesh.visual.texture.PBRMaterial(image=texture_pil)

            # mesh = mesh.simplify_quadric_decimation(face_count=1000)

            # print("simplified mesh")

            # color_visuals = trimesh.visual.TextureVisuals(uv=uv, material=material)
            mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)

            print("Mesh constructed")

            if False:

                obj_data = mesh.export(file_type="obj")
                # mtl_data = mesh.visual.material.export(file_type="mtl")

                print("Obj exported")

                mtl_data = f"""newmtl material_0
    Ka 0.00000000 0.00000000 0.00000000
    Kd 0.00000000 0.00000000 0.00000000
    Ks 0.00000000 0.00000000 0.00000000
    Ns 0.00000000
    map_Kd material_0.png
                """
                # mtl_data = mesh.visual.material.to_mtl()

                """    np.random.randint(
                        100,
                        255,
                        (full_image.shape[0], full_image.shape[1], 3),
                        dtype=np.uint8,
                    )
                )"""

                # Save the texture image as PNG in memory
                texture_buffer = io.BytesIO()
                texture_pil.save(texture_buffer, format="PNG")
                texture_buffer.seek(0)
                texture_data = texture_buffer.getvalue()

                # Create an in-memory ZIP file
                zip_buffer = io.BytesIO()

                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    # Add OBJ and MTL files to the ZIP
                    zip_file.writestr("model.obj", obj_data)
                    zip_file.writestr("material_0.png", texture_data)
                    zip_file.writestr("model.mtl", mtl_data)

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

            # Convert buffer to bytes


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
# 37,-75.5,38,-74.9
