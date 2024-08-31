# combine.py
import os
import glob
from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely import Point

search_str = 'data/vectors/fao/processed/tz/**.csv*'
def ceo_csv_combine(search_str):
    list_ = glob.glob(search_str)
    list_ = [file for file in list_ if os.path.isfile(file)]# ignore tmp directories 
    def _to_gdf(df, longitude: str, latitude: str)->gpd.GeoDataFrame:
        """Converts a DataFrame to a GeoDataFrame with Point geometries.

        Args:
            df (pd.DataFrame): The DataFrame to convert.

        Returns:
            gpd.GeoDataFrame: The converted GeoDataFrame.
        """
        geometry = [Point(lonlat) for lonlat in zip(df[longitude], df[latitude])]
        return gpd.GeoDataFrame(df, geometry=geometry, crs=4326)
    
    res = []
    for i in list_:
        tmp = pd.read_csv(i)
        res.append(tmp)

    allres = pd.concat(res)

    _path = Path(list_[0])
    print('_path',_path)
    out_folder = _path.parent / _path.name.__str__().split('.csv')[0]
    if not out_folder.exists():
        print(f"making new folder: {out_folder}")
        out_folder.mkdir(parents=True)
    print('out_folder',out_folder)
    out_file = out_folder.joinpath(out_folder.with_suffix(".shp").name)
    print('out_file',out_file)
    gdf = _to_gdf(allres,longitude='long',latitude='lat')
    save_cols = ['PLOTID', 'SAMPLEID', 'prob_label','pred_label','success']
    gdf['PLOTID'] =gdf.id
    gdf['SAMPLEID'] =gdf.id
    out = gpd.GeoDataFrame(gdf[save_cols], geometry=gdf.geometry)
    out.to_file(out_file)
