# combine.py
import glob
import pandas as pd
import geopandas as gpd
from shapely import Point

list_ = glob.glob('data/vectors/fao/processed/tz/**.csv*')
def _to_gdf(df, longitude: str, latitude: str)->gpd.GeoDataFrame:
    """Converts a DataFrame to a GeoDataFrame with Point geometries.

    Args:
        df (pd.DataFrame): The DataFrame to convert.

    Returns:
        gpd.GeoDataFrame: The converted GeoDataFrame.
    """
    geometry = [Point(lonlat) for lonlat in zip(df[longitude], df[latitude])]
    return gpd.GeoDataFrame(df, geometry=geometry, crs=4326)
print(list_)
res = []
for i in list_:
    tmp = pd.read_csv(i)
    res.append(tmp)

allres = pd.concat(res)
print(allres)
gdf = _to_gdf(allres,longitude='long',latitude='lat')
save_cols = ['PLOTID', 'SAMPLEID', 'prob_label','pred_label','success']
gdf['PLOTID'] =gdf.id
gdf['SAMPLEID'] =gdf.id
out = gpd.GeoDataFrame(gdf[save_cols], geometry=gdf.geometry)
out.to_file('tz-ssl4eo-predictions.shp')