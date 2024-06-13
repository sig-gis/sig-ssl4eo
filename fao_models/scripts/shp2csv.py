import geopandas as gpd


def shp2csv(src, dst):
    """Converts a shapefile of polygons to a CSV. Centroid is calculated
    by projecting to Equal Area Cylindrical and is then re-projected back to
    the input CRS.

    Args:
        src (str): A SHP file or other file readable by geopandas.
        dst (str): The CSV destination.
    """
    gdf = gpd.read_file(src)
    # https://gis.stackexchange.com/a/401815/95209
    gdf["centroid"] = gdf.to_crs("+proj=cea").centroid.to_crs(gdf.crs)
    gdf["long"] = gdf["centroid"].x
    gdf["lat"] = gdf["centroid"].y
    gdf.to_csv(dst)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=str, help="source shp file")
    parser.add_argument("dst", type=str, help="destination csv file")
    args = parser.parse_args()

    shp2csv(args.src, args.dst)
