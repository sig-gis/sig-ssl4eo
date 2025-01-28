import collections
import argparse
from pathlib import Path
from types import SimpleNamespace
import logging
import os
import pandas as pd
import geopandas as gpd
import copy

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.io import ReadFromCsv, WriteToText
from fao_models.common import load_yml
from fao_models._types import Config
from fao_models.scripts import shp2csv

logging.basicConfig(
    filename="forest-classifier-beam.log",
    encoding="utf-8",
    format="%(asctime)s - %(message)s",
    level=logging.INFO,
)


# https://github.com/kubeflow/examples/blob/master/LICENSE
class DictToCSVString(beam.DoFn):
    """Convert incoming dict to a CSV string.

    This DoFn converts a Python dict into
    a CSV string.

    Args:
      fieldnames: A list of strings representing keys of a dict.
    """

    def __init__(self, fieldnames):
        # super(DictToCSVString, self).__init__()

        self.fieldnames = fieldnames

    def process(self, element, *_args, **_kwargs) -> collections.abc.Iterator[str]:
        """Convert a Python dict instance into CSV string.

        This routine uses the Python CSV DictReader to
        robustly convert an input dict to a comma-separated
        CSV string. This also handles appropriate escaping of
        characters like the delimiter ",". The dict values
        must be serializable into a string.

        Args:
          element: A dict mapping string keys to string values.
            {
              "key1": "STRING",
              "key2": "STRING"
            }

        Yields:
          A string representing the row in CSV format.
        """
        import io
        import csv

        fieldnames = self.fieldnames
        filtered_element = {
            key: value for (key, value) in element.items() if key in fieldnames
        }
        with io.StringIO() as stream:
            writer = csv.DictWriter(stream, fieldnames)
            writer.writerow(filtered_element)
            csv_string = stream.getvalue().strip("\r\n")

        yield csv_string


class Predict(beam.DoFn):
    def __init__(self, config: Config):
        self._config = config
        logging.info(f"config :{self._config.__dict__}")

    def setup(self):
        self.load_model()

    def load_model(self):
        """load model"""
        from fao_models.models._models import get_model
        from fao_models.models.dino.utils import restart_from_checkpoint
        import os

        c = self._config
        self.model, self.linear_classifier = get_model(**c.__dict__)
        restart_from_checkpoint(
            os.path.join(c.model_head_root),
            state_dict=self.linear_classifier,
        )

    def process(self, element):
        import torch
        from fao_models.datasets.ssl4eo_dataset import SSL4EO

        if element["img_root"] == "RuntimeError":
            element["ssl4_prob"] = 0
            element["ssl4_pred"] = 0
            element["success"] = False
            yield element

        else:
            dataset = SSL4EO(
                root=element["img_root"].parent,
                mode="s2c",
                normalize=False,  # todo add normalized to self._config.
            )

            image = dataset[0]
            image = torch.unsqueeze(torch.tensor(image), 0).type(torch.float32)

            self.linear_classifier.eval()
            with torch.no_grad():
                intermediate_output = self.model.get_intermediate_layers(
                    image, self._config.n_last_blocks
                )
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)

            output = self.linear_classifier(output)
            element["ssl4_prob"] = output.detach().cpu().item()
            element["ssl4_pred"] = round(element["ssl4_prob"])
            element["success"] = True
            yield element


class GetImagery(beam.DoFn):
    def __init__(self, dst, project, bands, crops, year, uid):
        self.dst = dst
        self.PROJECT = project
        self.BANDS = bands
        self.CROPS = crops
        self.year = year
        self.uid = uid
        # TODO: change caps to lower

    def setup(self):
        import ee
        import google.auth

        credentials, _ = google.auth.default()
        ee.Initialize(
            credentials,
            project=self.PROJECT,
            opt_url="https://earthengine-highvolume.googleapis.com",
        )

    def get_image_dst(self, root_dst, id):
        from pathlib import Path

        return Path(root_dst) / str(id) / str(id)

    def check_exsits(self, dst, id):
        root = self.get_image_dst(dst, id)
        return root.exists()

    def process(self, element):
        """download imagery"""
        from fao_models.download_data.download_wraper import single_patch
        from pathlib import Path

        uid = element.__getattribute__(self.uid)
        try:

            sample = element
            logging.info(f"start {uid}")
            coords = (sample.long, sample.lat)
            local_root = Path(self.dst)
            if self.check_exsits(local_root / "imgs", uid):
                logging.info(f"{uid} already exists skipping download...")
                img_root = self.get_image_dst(local_root / "imgs", uid)
            else:
                img_root = single_patch(
                    coords,
                    id=uid,
                    dst=local_root / "imgs",
                    year=self.year,
                    bands=self.BANDS,
                    crop_dimensions=self.CROPS,
                )

            logging.info(f"end {uid}")
            yield {
                "img_root": img_root,
                "long": sample.long,
                "lat": sample.lat,
                "PLOTID": uid,
            }
        except RuntimeError:
            logging.warning(f"no image found for sample: {uid}")
            # no image found
            yield {
                "img_root": "RuntimeError",
                "long": sample.long,
                "lat": sample.lat,
                "PLOTID": uid,
            }


def pipeline(beam_options, dotargs: SimpleNamespace):
    logging.info("Pipeline is starting.")
    import time
    from fao_models._types import Config

    st = time.time()
    if beam_options is not None:
        beam_options = PipelineOptions(**load_yml(beam_options))
    conf = Config(**load_yml(dotargs.model_config))
    cols = ["PLOTID", "long", "lat", "ssl4_prob", "ssl4_pred", "success"]

    options = PipelineOptions(
        runner=conf.beam_params.runner,  # or 'DirectRunner'
        direct_num_workers=conf.beam_params.direct_num_workers,
        direct_running_mode=conf.beam_params.direct_running_mode,
    )

    if dotargs.input.endswith('.shp'):
        _cur = Path(dotargs.input)
        _parent = _cur.parent
        _new = _parent/ f"{_cur.stem}.csv"
        shp2csv.shp2csv(_cur, _new)
        dotargs.input = _new.__str__()
        print(dotargs)

    with beam.Pipeline(options=options) as p:
        forest_pipeline = (
            p
            | "read input data" >> ReadFromCsv(dotargs.input, splittable=True)
            | "Reshuffle to prevent fusion" >> beam.Reshuffle()
            | "download imagery"
            >> beam.ParDo(
                GetImagery(
                    dst=conf.imagery_params.tmp,
                    project=conf.project_params.eeproject,
                    bands=conf.imagery_params.bands,
                    crops=conf.imagery_params.crops,
                    year=conf.imagery_params.predict_year,
                    uid=conf.project_params.predict_id,
                )
            ).with_output_types(dict)
            | "predict" >> beam.ParDo(Predict(config=conf)).with_output_types(dict)
            | "to csv str" >> beam.ParDo(DictToCSVString(cols))
            | "write to csv" >> WriteToText(dotargs.output, header=",".join(cols))
        )

    print(f"pipeline took {time.time()-st}")


def run():
    argparse.FileType()

    argparse.FileType()
    parser = argparse.ArgumentParser(description="Run Beam Pipeline for FAO Forest Classifier")
     
    # Input/Output arguments group
    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument("--input", "-i", type=str, required=True, help="Input shapefile path")
    io_group.add_argument("--output", "-o", type=str, required=True, help="Output CSV file path")
    io_group.add_argument("--model-config", "-mc", type=str, required=True, help="Model configuration file path")

    # Beam options group
    beam_group = parser.add_argument_group("Beam Options")
    beam_group.add_argument("--beam-runner", type=str, required=False, help="Pipeline runner (e.g., DirectRunner, DataflowRunner)")

    # Dataflow options group
    dataflow_group = parser.add_argument_group("Dataflow Options")
    dataflow_group.add_argument("--project", "-p", type=str, required=False, help="GCP project ID")
    dataflow_group.add_argument("--region", "-r", type=str, required=False, help="GCP region")
    dataflow_group.add_argument("--temp_location", "-tl", type=str, required=False, help="GCP temp location")
    dataflow_group.add_argument("--staging_location", "-sl", type=str, required=False, help="GCP staging location")
    dataflow_group.add_argument("--job_name", "-jn", type=str, required=False, help="Dataflow job name")

    args = parser.parse_args()
    print(args)

    beam_runner = args.beam_runner
    
    if beam_runner == "DataflowRunner":
        pipeline_options = PipelineOptions(
            runner=beam_runner,
            project=args.project,
            job_name=args.job_name,
            temp_location=args.temp_location,
            region=args.region)
        pipeline(beam_options=pipeline_options, dotargs=args)
    else:
        pipeline(beam_options=None, dotargs=args)

    logging.info(f"merging outputs to one dataframe")
    cur = Path(args.input)

    parent = cur.parent
    files = [(parent/ file) for file in os.listdir(parent) if file.startswith(Path(args.output).stem)]

    # merge all .csv shard files
    df = pd.concat([pd.read_csv(file) for file in files])

    # join it with the input shapefile 
    shp = gpd.read_file(args.input)
    shp['PLOTID'] = shp['PLOTID'].astype('int64')
    shp.to_file(parent/ f"{cur.stem}_input_shp_intmd.shp")
    
    joined = gpd.GeoDataFrame(shp.merge(df, on='PLOTID'), geometry='geometry')
    
    # save the geodataframe as a shapefile
    joined.to_file(args.output)

if __name__ == "__main__":
    run()
