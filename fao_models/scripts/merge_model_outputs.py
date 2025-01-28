import geopandas as gpd
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Merge two shapefiles with model predictions based on PLOTID")
    parser.add_argument('--input1', type=str, required=True, help='Path to the first input shapefile')
    parser.add_argument('--input2', type=str, required=True, help='Path to the second input shapefile')
    parser.add_argument('--output', type=str, required=True, help='Path to the output shapefile')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    shp1 = gpd.read_file(args.input1)
    shp2 = gpd.read_file(args.input2)

    out_shp = shp1.merge(shp2, on='PLOTID')
    out_shp = out_shp[['PLOTID', 'ssl4_prob', 'ssl4_pred','r50_prob','r50_pred','geometry_x']]
    out_shp = out_shp.rename(columns={'geometry_x':'geometry'})
    out_shp.to_file(args.output)

