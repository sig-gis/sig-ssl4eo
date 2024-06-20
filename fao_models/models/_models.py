from fao_models.models.dino import vision_transformer as vits
from fao_models.models.dino import utils
from fao_models.models.classification.linear import LinearClassifier
from pathlib import Path


def load_base_model(
    arch: str,
    patch_size: int,
    n_last_blocks: int,
    avgpool_patchtokens: bool,
    pretrained: str | Path,
    checkpoint_key: str,
):
    if arch in vits.__dict__.keys():
        model = vits.__dict__[arch](patch_size=patch_size, num_classes=0, in_chans=13)
        embed_dim = model.embed_dim * (n_last_blocks + int(avgpool_patchtokens))
    else:
        raise NotImplementedError(f"Unknown architecture: {arch}")

    model.cpu()
    model.eval()
    # load weights to evaluate
    utils.load_pretrained_weights(model, pretrained, checkpoint_key, arch, patch_size)

    print(f"Model {arch} built.")
    return model, embed_dim


def get_linear_dino(**c):
    base_model, embed_dim = load_base_model(
        pretrained=c["model_root"],
        checkpoint_key=c["checkpoint_key"],
        arch=c["arch"],
        patch_size=c["patch_size"],
        n_last_blocks=c["n_last_blocks"],
        avgpool_patchtokens=c["avgpool_patchtokens"],
    )
    linear_model = LinearClassifier(embed_dim, 1)
    return base_model, linear_model


model_names = {"linear-dino": get_linear_dino}


def get_model(model_name: str = "", **config_params):
    base, head = model_names[model_name](**config_params)

    return base, head
