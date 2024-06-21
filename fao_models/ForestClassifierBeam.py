import collections
import argparse
from types import SimpleNamespace
import logging

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.io import ReadFromCsv, WriteToText
from fao_models.common import load_yml
from fao_models._types import Config

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
            element["prob_label"] = 0
            element["pred_label"] = 0
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
            element["prob_label"] = output.detach().cpu().item()
            element["pred_label"] = round(element["prob_label"])
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
                "id": uid,
            }
        except RuntimeError:
            logging.warning(f"no image found for sample: {uid}")
            # no image found
            yield {
                "img_root": "RuntimeError",
                "long": sample.long,
                "lat": sample.lat,
                "id": uid,
            }


def pipeline(beam_options, dotargs: SimpleNamespace):
    logging.info("Pipeline is starting.")
    import time
    from fao_models._types import Config

    st = time.time()
    if beam_options is not None:
        beam_options = PipelineOptions(**load_yml(beam_options))
    conf = Config(**load_yml(dotargs.model_config))
    cols = ["id", "long", "lat", "prob_label", "pred_label", "success"]

    options = PipelineOptions(
        runner=conf.beam_params.runner,  # or 'DirectRunner'
        direct_num_workers=conf.beam_params.direct_num_workers,
        direct_running_mode=conf.beam_params.direct_running_mode,
    )

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

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument("--model-config", "-mc", type=str, required=True)
    group = parser.add_argument_group("pipeline-options")
    group.add_argument("--beam-config", "-bc", type=str)
    args = parser.parse_args()

    pipeline(beam_options=args.beam_config, dotargs=args)


if __name__ == "__main__":
    run()
