import typer
from typing_extensions import Annotated
import torch

from common import load_yml
from datasets.ssl4eo_dataset import SSL4EO, random_subset
from main import eval_linear
from _types import Config


def main(config: str, test: Annotated[bool, typer.Option()] = False):
    args = Config(**load_yml(config))

    _data_train = SSL4EO(
        root=args.imgs_training, mode="s2c", label=args.labels_training, normalize=False
    )
    _data_test = SSL4EO(
        root=args.imgs_testing, mode="s2c", label=args.labels_testing, normalize=False
    )
    if test:
        _data_train = random_subset(_data_train, args.random_subset_frac, args.seed)
        _data_test = random_subset(_data_train, 0.01, args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")
    print(f"testing size:{len(_data_test)}")
    eval_linear(
        training_data=_data_train,
        dataset_val=_data_test,
        arch=args.arch,
        model_root=args.model_root,
        avgpool_patchtokens=args.avgpool_patchtokens,
        patch_size=args.patch_size,
        n_last_blocks=args.n_last_blocks,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_key=args.checkpoint_key,
        checkpoints_dir=args.checkpoints_dir,
        resume=args.resume,
        device=device,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    typer.run(main)
