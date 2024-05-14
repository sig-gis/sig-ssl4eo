# flow outline
# inputs:
#   -csv_plots
#   -config:
#   -output_dir
# output:
#   -csv_predicted_plots

# step1
# - load the csv
# for each plot
# -step2
# load model
# predict
# return json/objct
# step 4
# update csv with results
# step 5 fin

from metaflow import FlowSpec, step, Parameter
import pandas as pd
import rasterio as rio


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


class SSL4EOFlow(FlowSpec):

    samples = Parameter(name="samples", type=str, help="The path to a csv of samples.")
    config = Parameter(name="config", type=str, help="The path to a model config.")

    def load_model(self):
        print("i ran")
        return pd.read_csv("fao_models/data/match_dev.csv").to_dict(orient="records")

    @step
    def start(self):
        """load csv, initialize config."""
        from common import load_yml
        from _types import Config

        samples = pd.read_csv(self.samples)
        self._samples = samples.to_dict(orient="records")
        self._config = Config(**load_yml(self.config))

        self.next(self.load_model)

    @step
    def load_model(self):
        from main import load_base_model
        from models.classification import linear

        c = self._config
        self.model, embed_dim = load_base_model(
            pretrained=c.model_root,
            checkpoint_key=c.checkpoint_key,
            arch=c.arch,
            patch_size=c.patch_size,
            n_last_blocks=c.n_last_blocks,
            avgpool_patchtokens=c.avgpool_patchtokens,
        )
        self.linear_classifier = linear.LinearClassifier(embed_dim, num_labels=1)

        self.next(self.get_imagery, foreach="_samples")

    @step
    def get_imagery(self):
        from download_data.download_wraper import single_patch
        from pathlib import Path
        import ee
        import google.auth

        PROJECT = "pc530-fao-fra-rss"
        credentials, _ = google.auth.default()
        ee.Initialize(
            credentials,
            project=PROJECT,
            opt_url="https://earthengine-highvolume.googleapis.com",
        )

        # download imagery
        self.sample = self.input
        coords = (self.sample["long"], self.sample["lat"])
        local_root = Path(__file__).parent
        img_root = single_patch(
            coords,
            id=self.sample["id"],
            dst=local_root / "testing123",
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

        _dataset = SSL4EO(
            root=self.sample["img_root"].parent,
            mode="s2c",
            normalize=False,  # todo add normalized to self._config.
        )

        image = _dataset[0]
        image = torch.unsqueeze(torch.tensor(image), 0).type(torch.float32)
        self.linear_classifier.eval()
        with torch.no_grad():
            intermediate_output = self.model.get_intermediate_layers(
                image, self._config.n_last_blocks
            )
            output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)

        output = self.linear_classifier(output)
        self.sample["label"] = output.detach().cpu()
        self.next(self.join_res)

    @step
    def join_res(self, inputs):
        print("join res:::::", [i.sample for i in inputs])
        self.next(self.end)

    @step
    def end(self):
        print("fin")


if __name__ == "__main__":
    SSL4EOFlow()
