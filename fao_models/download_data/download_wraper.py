from pathlib import Path

try:
    from ssl4eo_downloader import get_patch_by_match, get_collection, save_patch
except:
    from fao_models.download_data.ssl4eo_downloader import (
        get_patch_by_match,
        get_collection,
        save_patch,
    )
import tempfile


def single_patch(
    coords: tuple[float, float],
    id: str | int,
    year: int,
    dst: str | Path,
    collection: str = "COPERNICUS/S2",
    meta_cloud_name: str = "CLOUDY_PIXEL_PERCENTAGE",
    cloud_pct: int = 10,
    radius: int = 1320,
    bands: list[str] = [],
    crop_dimensions: list[str] = [],
    dtype: str = "float32",
):
    # get data collection (remove clouds)
    collection = get_collection(collection, meta_cloud_name, cloud_pct)
    _crops = list(map(lambda i: (i, i), crop_dimensions))
    crops = dict(zip(bands, _crops))

    patch, _ = get_patch_by_match(
        0, collection, bands, crops, dtype, year, radius, match_coords={"0": coords}
    )

    if patch is None:
        raise RuntimeError("no suitable image found")
    td = Path(dst)
    td.name
    root = td / str(id) / str(id)

    save_patch(
        raster=patch["raster"],
        coords=patch["coords"],
        metadata=patch["metadata"],
        path=root,
    )
    return root


if __name__ == "__main__":
    import ee
    import google.auth

    bands = [
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
    ]
    crops = [44, 264, 264, 264, 132, 132, 132, 264, 132, 44, 44, 132, 132]
    dst = "tmp-del"
    PROJECT = "pc530-fao-fra-rss"
    credentials, _ = google.auth.default()
    ee.Initialize(
        credentials,
        project=PROJECT,
        opt_url="https://earthengine-highvolume.googleapis.com",
    )
    a = single_patch(
        (-45.974, -21.146), 2019, dst=dst, bands=bands, crop_dimensions=crops
    )
    print(a)
