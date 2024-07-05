# SSL4EO - FAO Forest Classifier

## Environment Setup (Inference.. under construction)

- Create a fresh virtual environment
- Once in the fresh env, run `pip install -r requirements.txt` (NOTE: update that file by exporting from sepal workspace)
- Clone the repo, then from root, `pip install -e .`
- run `earthengine authenticate` to make sure you have GEE creds
- Download the linear dino backbone and our latest fine-tuned linear model weights files
-    In your terminal:
```
mkdir _models
gsutil cp gs://forest-nonforest/models/linear-dino-2/checkpoint.pth.tar ./_models/. 
gsutil cp gs://forest-nonforest/models/b13-vits16-dino/B13_vits16_dino_0099_ckpt.pth ./_models/.

```

## Training A Model

### Download Weights
- link

### Download labels (aka match file)
Labels are defined from the FAO 2019 survey data. 
- script link
- example usage

### Setting up a config

### Example usage
```bash
python fao_models/cli.py ./fao_models_runs/test.yml --test
```

## Inference

### Installing the package
From the root dir (`sig-ssl4eo`) you can install the package in developer mode or regularly:

- Developer mode
    - `pip install -e .`
- Regular mode
    - `pip install .`

### Creating csv to process
If you need to process a shp file of polygons you can use `shp2csv.py` to create a csv of centroids.

Example usage (python file):
```bash
python fao_models/scripts/shp2csv.py fao_models/data/vectors/fao/raw/ALL_centroids_completed_v1_/ALL_centroids_completed_v1_.shp fao_models/data/vectors/fao/ALL_centroids_completed_v1_no_index.csv
```

Example usage (python module):
```bash
python -m fao_models.scripts.shp2csv fao_models/data/vectors/fao/raw/TZ_workshop_NEW_centr/TZ_workshop_NEW_centr.shp fao_models/data/vectors/fao/intermediate/TZ_workshop_NEW_centr.csv
```

### Setting up config
Inference requires a few more parameters than training. Bet practice is to keep all parameters in a single config file. So if you're running inference on multiple different datasets then try to make a full config for each set and track them via git. It can get messy but will help if you need to go back and troubleshoot!

Required parameters:
- Project parameters
    - eeproject (str): The earth engine project to use.
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

### Example usage
```bash
python fao_models/ForestClassifierBeam.py -i fao_models/data/vectors/fao/intermediate/test_del.csv -o TEST-fao-csv.csv -mc fao_models_runs/test.yml
```
