# combine.py
import glob
from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely import Point

search_str = 'data/vectors/fao/processed/tz/**.csv*'
def ceo_csv_combine(search_str):
    list_ = glob.glob(search_str)
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
    filename = f"{_path.parent / _path.name.__str__().split('.csv')[0]}.shp"
    gdf = _to_gdf(allres,longitude='long',latitude='lat')
    save_cols = ['PLOTID', 'SAMPLEID', 'prob_label','pred_label','success']
    gdf['PLOTID'] =gdf.id
    gdf['SAMPLEID'] =gdf.id
    out = gpd.GeoDataFrame(gdf[save_cols], geometry=gdf.geometry)
    out.to_file(filename)