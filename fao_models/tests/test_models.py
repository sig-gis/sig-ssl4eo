# quick tests refactor to unittest/pytest later
import sys
import os.path
import pytest

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)


# get model should load a full model (base mode, head)
def test_get_model():
    from models._models import get_model
    from models.dino.vision_transformer import VisionTransformer
    from models.classification.linear import LinearClassifier
    from _types import Config
    from common import load_yml

    f = "fao_models/tests/config/test.yml"
    c = Config(**load_yml(f))
    print()
    b, h = get_model(**c.__dict__)
    assert isinstance(b, VisionTransformer), "VIT base failed"
    assert isinstance(h, LinearClassifier), "linear head failed"


def dataset_init():
    from datasets.ssl4eo_dataset import SSL4EO

    dataset = SSL4EO(root="fao_models/tests/data/imgs", mode="s2c", normalize=False)
    return dataset


def test_expected_shape():
    _dataset = dataset_init()
    data = _dataset[0]
    assert data.shape == (13, 264, 264)


def test_expected_size():
    _dataset = dataset_init()
    assert len(_dataset) == 2


def test_get_correct_label():
    from datasets.ssl4eo_dataset import SSL4EO

    dataset = SSL4EO(
        root="fao_models/tests/data/imgs",
        label="fao_models/tests/data/match_dev.csv",
        mode="s2c",
        normalize=False,
    )

    expected_labels = {"000001": 1, "000000": 0}
    # labels are stored in desc order
    _, label_000001 = dataset[0]
    _, label_000000 = dataset[1]
    assert label_000001 == expected_labels["000001"]
    assert label_000000 == expected_labels["000000"]
