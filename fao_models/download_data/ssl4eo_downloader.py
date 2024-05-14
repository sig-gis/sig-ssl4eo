""" Sample and download Satellite tiles with Google Earth Engine

### run the script:

## Install and authenticate Google Earth Engine (https://developers.google.com/earth-engine/guides/python_install) # noqa: E501

## match and download pre-sampled locations
python ssl4eo_downloader.py \
    --save_path /Volumes/External/pc530/training \
    --collection COPERNICUS/S2 \ 
    --meta_cloud_name CLOUDY_PIXEL_PERCENTAGE \
    --cloud_pct 10 \ 
    --year 2018  \ 
    --radius 1320 \ 
    --bands B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B10 B11 B12  \ 
    --crops 44 264 264 264 132 132 132 264 132 44 44 132 132 \ 
    --dtype uint16 \ 
    --num_workers 12 \ 
    --log_freq 100 \ 
    --match_file data/match_training_sample.csv  


## resume from interruption (e.g. 20 ids processed)
python ssl4eo_downloader.py \
    -- ... \
    --resume ./data/checked_locations.csv \
    --indices_range 20 250000


## Example: download Landsat-8, match SSL4EO-S12 locations but keep same patch size
python ssl4eo_downloader.py \
    --save_path ./data \
    --collection LANDSAT/LC08/C02/T1_TOA \
    --meta_cloud_name CLOUD_COVER \
    --cloud_pct 20 \
    --dates 2021-12-21 2021-09-22 2021-06-21 2021-03-20 \
    --radius 1980 \
    --bands B1 B2 B3 B4 B5 B6 B7 B8 B9 B10 B11 \
    --crops 132 132 132 132 132 132 132 264 264 132 132 \
    --dtype float32 \
    --num_workers 8 \
    --log_freq 100 \
    --match_file ./data/ssl4eo-s12_center_coords.csv \
    --indices_range 0 250000

### Notes
# By default, the script will sample and download Sentinel-2 L1C tiles (13 bands) with cloud cover less than 20%. # noqa: E501
# The script will download 250k little-overlap locations, 4 tiles for each location, one for each season (in a two-year buffer). # noqa: E501
# You may want to extend the buffer to more years by modifying the `get_period()` and `filter_collection()` functions. # noqa: E501

"""

import argparse
import csv
import json
import os
import time
import warnings
from collections import OrderedDict
from datetime import datetime
from multiprocessing.dummy import Lock, Pool
from typing import Any, Dict, List, Optional, Tuple

import ee
import numpy as np
import rasterio
import google.auth
from google.api_core import exceptions, retry

import urllib3
from rasterio.transform import Affine
from tqdm import tqdm

warnings.simplefilter("ignore", UserWarning)


def date2str(date: datetime) -> str:
    return date.strftime("%Y-%m-%d")


def maskS2clouds(args: Any, image: ee.Image) -> ee.Image:
    qa = image.select(args.qa_band)
    cloudBitMask = 1 << args.qa_cloud_bit
    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloudBitMask).eq(0)
    return image.updateMask(mask)


def get_collection(
    collection_name: str, meta_cloud_name: str, cloud_pct: float
) -> ee.ImageCollection:
    collection = ee.ImageCollection(collection_name)
    collection = collection.filter(ee.Filter.lt(meta_cloud_name, cloud_pct))
    # Uncomment the following line if you want to apply cloud masking.
    # collection = collection.map(maskS2clouds, args)
    return collection


def filter_collection(
    collection: ee.ImageCollection,
    coords: List[float],
    start_date: ee.Date,
    end_date: ee.Date,
) -> ee.ImageCollection:
    filtered = collection.filterDate(start_date, end_date)

    filtered = filtered.filterBounds(ee.Geometry.Point(coords))  # filter region

    if filtered.limit(1).size().getInfo() == 0:
        raise ee.EEException(
            f"ImageCollection.filter: No suitable images found in ({coords[1]:.4f}, {coords[0]:.4f}) between {start_date.getInfo()} and {end_date.getInfo()}."  # noqa: E501
        )
    return filtered


def center_crop(
    img: np.ndarray[Any, np.dtype[Any]], out_size: Tuple[int, int]
) -> np.ndarray[Any, np.dtype[Any]]:
    image_height, image_width = img.shape[:2]
    crop_height, crop_width = out_size
    crop_top = (image_height - crop_height + 1) // 2
    crop_left = (image_width - crop_width + 1) // 2
    return img[crop_top : crop_top + crop_height, crop_left : crop_left + crop_width]


def adjust_coords(
    coords: List[List[float]], old_size: Tuple[int, int], new_size: Tuple[int, int]
) -> List[List[float]]:
    xres = (coords[1][0] - coords[0][0]) / old_size[1]
    yres = (coords[0][1] - coords[1][1]) / old_size[0]
    xoff = int((old_size[1] - new_size[1] + 1) * 0.5)
    yoff = int((old_size[0] - new_size[0] + 1) * 0.5)
    return [
        [coords[0][0] + (xoff * xres), coords[0][1] - (yoff * yres)],
        [
            coords[0][0] + ((xoff + new_size[1]) * xres),
            coords[0][1] - ((yoff + new_size[0]) * yres),
        ],
    ]


def get_properties(image: ee.Image) -> Any:
    return image.getInfo()


def get_patch(
    collection: ee.ImageCollection,
    center_coord: List[float],
    radius: float,
    bands: List[str],
    crop: Optional[Dict[str, Any]] = None,
    dtype: str = "float32",
    sort_by: str = "system:time_start",
    sort_acceding: bool = False,
) -> Dict[str, Any]:
    image = collection.sort(sort_by, sort_acceding).first()  # get most recent
    region = (
        ee.Geometry.Point(center_coord).buffer(radius).bounds()
    )  # sample region bound
    patch = image.select(*bands).sampleRectangle(region, defaultValue=0)

    features = patch.getInfo()  # the actual download

    raster = OrderedDict()
    for band in bands:
        img = np.atleast_3d(features["properties"][band])
        if crop is not None:
            img = center_crop(img, out_size=crop[band])
        raster[band] = img.astype(dtype)

    coords0 = np.array(features["geometry"]["coordinates"][0])
    coords = [
        [coords0[:, 0].min(), coords0[:, 1].max()],
        [coords0[:, 0].max(), coords0[:, 1].min()],
    ]
    if crop is not None:
        band = bands[0]
        old_size = (
            len(features["properties"][band]),
            len(features["properties"][band][0]),
        )
        new_size = raster[band].shape[:2]
        coords = adjust_coords(coords, old_size, new_size)

    return OrderedDict(
        {"raster": raster, "coords": coords, "metadata": get_properties(image)}
    )


# TODO: [x] add google retry
# [] handel errors
# [x] pass in year instead of dates
# [x] refactor or rm get_period
# [x] simplify filter collection
@retry.Retry()
def get_patch_by_match(
    idx: int,
    collection: ee.ImageCollection,
    bands: List[str],
    crops: Dict[str, Any],
    dtype: str,
    year: str | int,
    radius: float,
    debug: bool = False,
    match_coords: Dict[str, Any] = {},
) -> Tuple[Optional[List[Dict[str, Any]]], List[float]]:
    # (lon,lat) of idx patch
    coords = match_coords[str(idx)]

    start_date = ee.Date.fromYMD(int(year), 1, 1)
    end_date = start_date.advance(1, "years")

    try:
        filtered_collection = filter_collection(
            collection, coords, start_date=start_date, end_date=end_date
        )
        patches = get_patch(
            filtered_collection,
            coords,
            radius,
            bands=bands,
            crop=crops,
            dtype=dtype,
            sort_acceding=True,
            sort_by="CLOUDY_PIXEL_PERCENTAGE",  # TODO: pull from cli?
        )

    except (ee.EEException, urllib3.exceptions.HTTPError) as e:
        print(e)
        return None, coords

    return patches, coords


def save_geotiff(
    img: np.ndarray[Any, np.dtype[Any]], coords: List[List[float]], filename: str
) -> None:
    height, width, channels = img.shape
    xres = (coords[1][0] - coords[0][0]) / width
    yres = (coords[0][1] - coords[1][1]) / height
    transform = Affine.translation(
        coords[0][0] - xres / 2, coords[0][1] + yres / 2
    ) * Affine.scale(xres, -yres)
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": channels,
        "crs": "+proj=latlong",
        "transform": transform,
        "dtype": img.dtype,
        "compress": "lzw",
        "predictor": 2,
    }
    with rasterio.open(filename, "w", **profile) as f:
        f.write(img.transpose(2, 0, 1))


def save_patch(
    raster: Dict[str, Any],
    coords: List[List[float]],
    metadata: Dict[str, Any],
    path: str,
) -> None:
    patch_id = metadata["properties"]["system:index"]
    patch_path = os.path.join(path, patch_id)
    os.makedirs(patch_path, exist_ok=True)

    for band, img in raster.items():
        save_geotiff(img, coords, os.path.join(patch_path, f"{band}.tif"))

    with open(os.path.join(patch_path, "metadata.json"), "w") as f:
        json.dump(metadata, f)


class Counter:
    def __init__(self, start: int = 0) -> None:
        self.value = start
        self.lock = Lock()

    def update(self, delta: int = 1) -> int:
        with self.lock:
            self.value += delta
            return self.value


def fix_random_seeds(seed: int = 42) -> None:
    np.random.seed(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path", type=str, default="./data/", help="dir to save data"
    )
    # collection properties
    parser.add_argument(
        "--collection", type=str, default="COPERNICUS/S2", help="GEE collection name"
    )
    parser.add_argument(
        "--qa_band", type=str, default="QA60", help="qa band name (optional)"
    )  # optional
    parser.add_argument(
        "--qa_cloud_bit", type=int, default=10, help="qa band cloud bit (optional)"
    )  # optional
    parser.add_argument(
        "--meta_cloud_name",
        type=str,
        default="CLOUDY_PIXEL_PERCENTAGE",
        help="meta data cloud percentage name",
    )
    parser.add_argument(
        "--cloud_pct", type=int, default=20, help="cloud percentage threshold"
    )

    # patch properties
    parser.add_argument(
        "--year",
        type=str,
        default="2018",
        help="The year from which to grab samples from",
    )
    parser.add_argument(
        "--radius", type=int, default=1320, help="patch radius in meters"
    )
    parser.add_argument(
        "--bands",
        type=str,
        nargs="+",
        default=[
            "B1",
            "B2",
            "B3",
            "B4",
            "B5",
            "B6",
            "B7",
            "B8",
            "B8A",
            "B9",
            "B10",
            "B11",
            "B12",
        ],
        help="bands to download",
    )
    parser.add_argument(
        "--crops",
        type=int,
        nargs="+",
        default=[44, 264, 264, 264, 132, 132, 132, 264, 132, 44, 44, 132, 132],
        help="crop size for each band",
    )
    parser.add_argument("--dtype", type=str, default="float32", help="data type")

    # download settings
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers")
    parser.add_argument("--log_freq", type=int, default=10, help="print frequency")
    parser.add_argument(
        "--resume", type=str, default=None, help="resume from a previous run"
    )
    # sampler options
    # op1: match pre-sampled coordinates and indexes
    parser.add_argument(
        "--match_file",
        type=str,
        default=None,
        help="match pre-sampled coordinates and indexes",
    )

    # number of locations to download
    parser.add_argument(
        "--indices_range",
        type=int,
        nargs=2,
        default=[0, 250000],
        help="indices to download",
    )
    # debug
    parser.add_argument("--debug", action="store_true", help="debug mode")

    args = parser.parse_args()

    fix_random_seeds(seed=42)

    # initialize ee
    PROJECT = "pc530-fao-fra-rss"
    credentials, _ = google.auth.default()
    ee.Initialize(
        credentials,
        project=PROJECT,
        opt_url="https://earthengine-highvolume.googleapis.com",
    )
    # get data collection (remove clouds)
    collection = get_collection(args.collection, args.meta_cloud_name, args.cloud_pct)

    year = args.year
    bands = args.bands
    dtype = args.dtype
    crops = {}
    for i, band in enumerate(bands):
        crops[band] = (args.crops[i], args.crops[i])

    # if resume
    ext_coords = {}
    ext_flags = {}
    if args.resume:
        ext_path = args.resume
        with open(ext_path) as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                key = row[0]
                val1 = float(row[1])
                val2 = float(row[2])
                ext_coords[key] = (val1, val2)  # lon, lat
                ext_flags[key] = int(row[3])  # success or not
    else:
        ext_path = os.path.join(args.save_path, "checked_locations.csv")

    # if match from pre-sampled coords (e.g. SSL4EO-S12)
    if args.match_file:
        match_coords = {}
        with open(args.match_file) as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                key = row[0]
                val1 = float(row[1])
                val2 = float(row[2])
                match_coords[key] = (val1, val2)  # lon, lat

    else:
        raise NotImplementedError

    start_time = time.time()
    counter = Counter()

    def worker(idx: int) -> None:
        if str(idx) in ext_coords.keys():
            if args.match_file:  # skip all processed ids
                return
            else:
                if ext_flags[str(idx)] != 0:  # only skip downloaded ids
                    return

        if args.match_file:
            patch, center_coord = get_patch_by_match(
                idx,
                collection,
                bands,
                crops,
                dtype,
                year,
                radius=args.radius,
                debug=args.debug,
                match_coords=match_coords,
            )
        else:
            raise NotImplementedError

        if patch is not None:
            if args.save_path is not None:
                # s2c
                location_path = os.path.join(args.save_path, "imgs", f"{idx:06d}")
                os.makedirs(location_path, exist_ok=True)

                save_patch(
                    raster=patch["raster"],
                    coords=patch["coords"],
                    metadata=patch["metadata"],
                    path=location_path,
                )

            count = counter.update(1)
            if count % args.log_freq == 0:
                print(f"Downloaded {count} images in {time.time() - start_time:.3f}s.")
        else:
            print("no suitable image for location %d." % (idx))

        # add to existing checked locations
        with open(ext_path, "a") as f:
            writer = csv.writer(f)
            if patch is not None:
                if args.match_file:
                    success = 2
                else:
                    success = 1
            else:
                success = 0
            data = [idx, center_coord[0], center_coord[1], success]
            writer.writerow(data)

        return

    # set indices
    if args.match_file is not None:
        indices = []
        for key in match_coords.keys():
            indices.append(int(key))
        indices = indices[args.indices_range[0] : args.indices_range[1]]
    elif args.indices_range is not None:
        indices = list(range(args.indices_range[0], args.indices_range[1]))
    else:
        print("Please set up indices.")
        raise NotImplementedError

    if args.num_workers == 0:
        for i in indices:
            worker(i)
    else:
        # parallelism data
        with Pool(processes=args.num_workers) as p:
            p.map(worker, indices)
