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
    b: str = "lol"

    def load_model(self):
        print("i ran")
        return pd.read_csv("fao_models/data/match_dev.csv").to_dict(orient="records")

    @step
    def start(self):

        samples = pd.read_csv(self.samples)
        # download imagery
        self._samples = samples.to_dict(orient="records")

        self.next(self.get_imagery, foreach="_samples")

    @step
    def get_imagery(self):
        from download_data.download_wraper import single_patch
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
        self.img_root = single_patch(
            coords, dst="testing123", year=2019, bands=BANDS, crop_dimensions=CROPS
        )
        print(self.sample)
        self.next(self.run_model)

    @step
    def run_model(self):
        print(self.img_root)
        self.next(self.join_res)

    @step
    def join_res(self, inputs):
        # print("in join", inputs.sample)
        # print("in join", inputs.img_root)
        self.next(self.end)

    @step
    def end(self):
        print(self.b)
        print("fin")


if __name__ == "__main__":
    SSL4EOFlow()
