import ee
import requests

from clean_csv import *

PROJECT = "pc530-fao-fra-rss"
ee.Initialize(project=PROJECT)

hex = ee.FeatureCollection(
    "projects/pc530-fao-fra-rss/assets/reference/hexWCenPropertiesTropics"
)
hex_forest = hex.filter(
    ee.Filter.And(ee.Filter.eq("FOREST", 1), ee.Filter.eq("LU18CEN", "Forest"))
).map(lambda f: f.set("label", 1))
hex_nonforest = hex.filter(
    ee.Filter.And(ee.Filter.eq("FOREST", 0), ee.Filter.neq("LU18CEN", "Forest"))
).map(lambda f: f.set("label", 0))
hex_samples = (
    ee.FeatureCollection([hex_forest, hex_nonforest]).flatten().randomColumn(seed=42)
)


# .Point(lng, lat).
def to_centroid(feat: ee.Feature):
    feat = feat.centroid()
    coords = feat.geometry().coordinates()
    return feat.set({"lng": coords.get(0), "lat": coords.get(1)})


hex_samples = hex_samples.map(to_centroid)

# split into train test and validate 70, 20, 10
training_sample = hex_samples.filter("random <= 0.7")
testing_sample = hex_samples.filter("random > 0.7 and random <=.9")
validation_sample = hex_samples.filter("random > 0.9")


# print(
#     training_sample.size().getInfo(),
#     testing_sample.size().getInfo(),
#     validation_sample.size().getInfo(),
# )
def downloadv2(collection: ee.FeatureCollection, full_namepath: str):
    url = collection.getDownloadURL()
    df = load_csv(url)
    add_new_index(df)
    df = select_columns(df, ["id", "lng", "lat", "label"])
    save(df, full_namepath)


def download(collection: ee.FeatureCollection, full_namepath: str):
    url = collection.getDownloadURL()
    response = requests.get(url)
    response.raise_for_status()
    with open(full_namepath, "wb") as fd:
        fd.write(response.content)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    path = Path(sys.argv[1])
    print(path)
    downloadv2(training_sample.limit(100), path / "dev_training_sample.csv")
    downloadv2(testing_sample.limit(100), path / "dev_testing_sample.csv")
    downloadv2(validation_sample.limit(100), path / "dev_validation_sample.csv")
