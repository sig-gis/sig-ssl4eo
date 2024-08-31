from fao_models import ForestClassifierBeam
from pathlib import Path
from types import SimpleNamespace
import shutil
import tqdm

from fao_models.scripts.tz_shards2shp import ceo_csv_combine

inputs_txt = "shplist_MDG.txt"
config = "fao_models_runs/predict_sepal_16w.yml"
output_root = Path("fao_models/data/vectors/fao/processed")
with open(inputs_txt) as f:
    shps = f.readlines(-1)
    shps = [shp.rstrip('\n') for shp in shps]

for input_shp in tqdm.tqdm(shps[1:]):
    print(f"input_shp:{input_shp}")
    the_split = str(input_shp).split('raw')[1].strip('\/')
    output_csv = output_root/ Path(the_split).parent / f"{Path(the_split).stem }.csv"
    print(f"output_csv:{output_csv}")
    args = SimpleNamespace(input=input_shp, output=output_csv.__str__(), model_config=config)
    ForestClassifierBeam.pipeline(beam_options=None, dotargs=args)
    ceo_csv_combine(f"{output_csv.parent.__str__()}/**.csv*")
    shutil.rmtree('TMP/imgs')
