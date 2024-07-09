# SSL4EO - FAO Forest Classifier

### Setup (Under Construction)

#### Environment
- Clone the repo and cd to the repo root directory (assume you're in root for all steps below), 
- Create a fresh virtual environment (we are using conda/mamba)
- Once in the fresh env, run `pip install -r requirements.txt` to get all dependencies*
- Install repo as an editable package: `pip install -e .`
- Run `earthengine authenticate` to make sure you have GEE creds
- *NOTE: I get a pip error on my machine that leads me to believe something is amiss in this requirements.txt related to geopandas dependencies. Mght be best to install all you can in conda/mamba (`mamba install -c conda-forge geopandas rasterio earthengine-api pyyaml`) then pip install the rest of the dependencies that aren't on conda.
- **NOTE2: keep trying to run `fao_models/ForestClassifierBeam.py --help` to identify any missing packages. 

### Inference 

At present, we use an apache_beam pipeline to parallelize image classification for large sets of FAO FRA plots. FRA officers send us shapefiles of FRA plots, our pipeline performs the following steps (simplification):
* Read shapefile and convert the geodataframe into an apache beam PCollection, where each element contains a unique plot id and a lon,lat tuple
* Load the model checkpoint that we'll use for inference
* for each element, download the required imagery, perform inference on it, and append prediction to a new csv file containing plot id and prediction label

#### Required Downloads
- Download the linear dino backbone and our latest fine-tuned linear model weights files to `_models` dir
In root:
```
mkdir _models
gsutil cp gs://forest-nonforest/models/linear-dino-2/checkpoint.pth.tar ./_models/. 
gsutil cp gs://forest-nonforest/models/b13-vits16-dino/B13_vits16_dino_0099_ckpt.pth ./_models/.

```
Then you can run the beam pipline as a cli script
#### Example usage
```bash
python fao_models/ForestClassifierBeam.py -i fao_models/data/vectors/fao/intermediate/test_del.csv -o TEST-fao-csv.csv -mc fao_models_runs/test.yml
```

### Creating csv to process **Deprecated?**
**just know these modules exist, but we don't use them currently in this manner now, as the pipeline itself reads shapefiles, and we have created some postprocessing logic in the batch inference scripts (see next section!)**

If you need to process a shp file of polygons you can use `shp2csv.py` as a command-line tool or as a module to create a csv of centroids.

Example usage (python file):
```bash
python fao_models/scripts/shp2csv.py fao_models/data/vectors/fao/raw/ALL_centroids_completed_v1_/ALL_centroids_completed_v1_.shp fao_models/data/vectors/fao/ALL_centroids_completed_v1_no_index.csv
```

Example usage (python module):
```bash
python -m fao_models.scripts.shp2csv fao_models/data/vectors/fao/raw/TZ_workshop_NEW_centr/TZ_workshop_NEW_centr.shp fao_models/data/vectors/fao/intermediate/TZ_workshop_NEW_centr.csv
```

### Setting up a batch inference script

In practice so far, we have ended up setting up one-off batch scripts to setup multiple beam pipelines for multiple shapefiles at once. There are two example python scripts demonstrating this. Importantly, these scripts perform some nice data management postprocessing steps after a pipeline job finishes, including merging all sharded .csv files into one .csv file, and converting that merged .csv file into a shapefile. We utlimately have ended up delivering these final shapefiles per FRA plot shapefile that we receive. 

Look at [batch_adolfo_copy.py](batch_adolfo_copy.py) for an example running it locally, and [batch_mdg_predict_sepal.py](batch_mdg_predict_sepal.py) for a similar example but running it on SEPAL. 

The key things we need for this is a .txt file ([shplist.txt](shplist.txt)) containing the list of shapefiles you want to run inference on, and a model config .yml, which we describe below. They are both hardcoded as paths into the scripts. We see these scripts as single-use for now, and not a long-term solution.

Example usage (python file):
```bash
python batch_mdg_predict_sepal.py
```

### Setting up config
Inference requires a few more parameters than training. Best practice is to keep all parameters in a single config file. So if you're running inference on multiple different datasets then try to make a full config for each set and track them via git. It can get messy but will help if you need to go back and troubleshoot!

Required parameters:
- Project parameters
    - eeproject (str): The earth engine project to use.
    - predict_id (str): the unique id field that the input shapefile uses. This is usually either PLOTID, SAMPLEID, or something simliar.

- Beam parameters
    - runner (str): The runner to use. (DirectRunner).
    - direct_num_workers (str): The num of workers.
    - direct_running_mode (str): The running mode.
- Imagery parameters
    - tmp (str): The path to save images to locally.
    - bands (list[str]): The bands to download.
    - crops (list[int]): The crop sizes for each band. 
- Model parameters
    - model_name (str): The model name to load. (supported names found in _models.py) 
    - model_root(str | Path): The path to the pertained SSL4EO weights
    - model_head_root (str | Path): The path to the pertained classifier head model. (linear in this case)
    - arch (str): The model architecture. (e.g. vit_small)
    - avgpool_patchtokens (bool): Whether or not to average pool patches (false).
    - patch_size (int): Number of patches. (16)
    - n_last_blocks (int): 4
    - checkpoint_key (str): teacher

For good examples, look at [/fao_models_runs/predict_sepal_16w.yml](fao_models_runs/predict_sepal_16w.yml) for running inference on SEPAL and [/fao_models/runs/test_local_pred_adolfo.yml](fao_models_runs/test_local_pred_adolfo.yml)


### Training A Model (Under Construction)

#### Download Weights
- link

#### Download labels (aka match file)
Labels are defined from the FAO 2019 survey data. 
- script link
- example usage

### Example usage
```bash
python fao_models/cli.py ./fao_models_runs/test.yml --test
```

