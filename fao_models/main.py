from datasets.ssl4eo_dataset import SSL4EO


root = "./ssl4eo-s12_100patches/"
# ds = SSL4EO(root=root, mode=["s1", "s2a", "s2c"])

# ds = Bigearthnet(root=)
# print(len(ds))
# from linear BE dino
model_root = "B13_vits16_dino_0099_ckpt.pth"
# ============ building network ... ============
# if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
import argparse
import sys
import torch
from torch import nn
from torchvision import models as torchvision_models

from models.dino import utils
from models.dino import vision_transformer as vits

# models.dino import vision_transformer as vits

parser = argparse.ArgumentParser()
parser.add_argument("--arch", default="vit_small", type=str, help="Architecture")
parser.add_argument(
    "--patch_size", default=16, type=int, help="Patch resolution of the model."
)
parser.add_argument(
    "--pretrained",
    default=model_root,
    type=str,
    help="Path to pretrained weights to evaluate.",
)
parser.add_argument(
    "--n_last_blocks",
    default=4,
    type=int,
    help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""",
)
parser.add_argument(
    "--avgpool_patchtokens",
    default=False,
    type=utils.bool_flag,
    help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""",
)
parser.add_argument(
    "--checkpoint_key",
    default="teacher",
    type=str,
    help='Key to use in the checkpoint (example: "teacher")',
)

args = parser.parse_args()
if args.arch in vits.__dict__.keys():
    model = vits.__dict__[args.arch](
        patch_size=args.patch_size, num_classes=0, in_chans=13
    )
    embed_dim = model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
# otherwise, we check if the architecture is in torchvision models
elif args.arch in torchvision_models.__dict__.keys():
    model = torchvision_models.__dict__[args.arch]()
    embed_dim = model.fc.weight.shape[1]
    model.conv1 = torch.nn.Conv2d(
        13, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    model.fc = nn.Identity()
    # model.fc = torch.nn.Linear(2048,19)
# if the network is a XCiT
elif "xcit" in args.arch:
    model = torch.hub.load("facebookresearch/xcit:main", args.arch, num_classes=0)
    embed_dim = model.embed_dim
else:
    print(f"Unknow architecture: {args.arch}")
    sys.exit(1)
model.cpu()
model.eval()
# load weights to evaluate
utils.load_pretrained_weights(
    model, args.pretrained, args.checkpoint_key, args.arch, args.patch_size
)
print(f"Model {args.arch} built.")
