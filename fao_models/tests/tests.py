# quick tests refactor to unittest/pytest later
import sys
import os.path

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


def test_intermediate_shape():
    from models._models import get_model
    from models.dino.vision_transformer import VisionTransformer
    from models.classification.linear import LinearClassifier
    from _types import Config
    from common import load_yml

    f = "/Users/johndilger/Documents/projects/SSL4EO-S12/test.yml"
    c = Config(**load_yml(f))

    b, h = get_model("linear-dino", **c.__dict__)
    print(b)


if __name__ == "__main__":
    test_get_model()
