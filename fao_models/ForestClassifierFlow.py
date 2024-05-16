from pathlib import Path

from metaflow import FlowSpec, step, Parameter
import pandas as pd

BANDS = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B9",
    "B10",
    "B11",
    "B12",
]
CROPS = [44, 264, 264, 264, 132, 132, 132, 264, 132, 44, 44, 132, 132]
PROJECT = "pc530-fao-fra-rss"


class SSL4EOFlow(FlowSpec):

    samples = Parameter(name="samples", type=str, help="The path to a csv of samples.")
    config = Parameter(name="config", type=str, help="The path to a model config.")
    dst = Parameter(name="dst", type=str, help="The path to save the resulting csv.")

    def check_dst(self, dst):
        dst = Path(dst)
        if dst.is_file() == False:
            dst = dst / "flowout.csv"
        return dst

    @step
    def start(self):
        """load csv, initialize config."""
        from common import load_yml
        from _types import Config

        samples = pd.read_csv(self.samples)
        self._samples = samples.to_dict(orient="records")
        self._config = Config(**load_yml(self.config))
        self._dst = self.check_dst(self.dst)

        self.next(self.load_model)

    @step
    def load_model(self):
        """load model"""
        from models._models import get_model
        from models.dino.utils import restart_from_checkpoint
        import os

        c = self._config
        self.model, self.linear_classifier = get_model(**c.__dict__)
        restart_from_checkpoint(
            os.path.join(c.model_head_root),
            state_dict=self.linear_classifier,
        )

        self.next(self.get_imagery, foreach="_samples")

    @step
    def get_imagery(self):
        """download imagery"""
        from download_data.download_wraper import single_patch

        import ee
        import google.auth

        credentials, _ = google.auth.default()
        ee.Initialize(
            credentials,
            project=PROJECT,
            opt_url="https://earthengine-highvolume.googleapis.com",
        )

        #
        self.sample = self.input
        coords = (self.sample["long"], self.sample["lat"])
        local_root = self._dst.parent
        img_root = single_patch(
            coords,
            id=self.sample["id"],
            dst=local_root / "imgs",
            year=2019,
            bands=BANDS,
            crop_dimensions=CROPS,
        )
        self.sample["img_root"] = img_root
        self.next(self.run_model)

    @step
    def run_model(self):
        import torch
        from datasets.ssl4eo_dataset import SSL4EO

        dataset = SSL4EO(
            root=self.sample["img_root"].parent,
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
        self.sample["prob_label"] = output.detach().cpu().item()
        self.sample["pred_label"] = round(self.sample["prob_label"])
        self.next(self.join_results)

    @step
    def join_results(self, inputs):
        self.out = pd.DataFrame(
            [i.sample for i in inputs],
        ).sort_values("id")
        self.merge_artifacts(inputs, include=["_dst"])
        columns_save = ["id", "long", "lat", "prob_label", "pred_label", "label"]
        self.out[columns_save].to_csv(self._dst, index=False)

        self.next(self.end)

    @step
    def end(self):
        print("fin")


if __name__ == "__main__":
    SSL4EOFlow()
