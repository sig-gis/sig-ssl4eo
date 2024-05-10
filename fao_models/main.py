import sys
import os
from pathlib import Path
import json
from typing import Literal

import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score
import typer
from typing_extensions import Annotated
import yaml
from torchmetrics.classification import BinaryF1Score

from datasets.ssl4eo_dataset import SSL4EO, random_subset
from models.dino import utils
from models.dino import vision_transformer as vits
from models.classification import linear
from _types import Config


def train(model, linear_classifier, optimizer, loader, epoch, n, avgpool, device, arch):
    linear_classifier.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    for images, target in metric_logger.log_every(loader, 20, header):
        inp = images.type(torch.float32).to(device)
        target = target.to(device).float()

        # forward
        with torch.no_grad():
            if "vit" in arch:
                intermediate_output = model.get_intermediate_layers(inp, n)
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                if avgpool:
                    output = torch.cat(
                        (
                            output.unsqueeze(-1),
                            torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(
                                -1
                            ),
                        ),
                        dim=-1,
                    )
                    output = output.reshape(output.shape[0], -1)
            else:
                output = model(inp)
        output = linear_classifier(output)

        # compute cross entropy loss
        loss = nn.BCELoss()(output, torch.unsqueeze(target, 1))

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # log
        # torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_network(val_loader, model, linear_classifier, n, avgpool, device, arch):
    linear_classifier.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    for images, target in metric_logger.log_every(val_loader, 20, header):
        inp = images.type(torch.float32).to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).float()

        # forward
        with torch.no_grad():
            if "vit" in arch:
                intermediate_output = model.get_intermediate_layers(inp, n)
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                if avgpool:
                    output = torch.cat(
                        (
                            output.unsqueeze(-1),
                            torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(
                                -1
                            ),
                        ),
                        dim=-1,
                    )
                    output = output.reshape(output.shape[0], -1)
            else:
                output = model(inp)
        output = linear_classifier(output)
        loss = nn.BCELoss()(output, torch.unsqueeze(target, 1))

        acc1 = average_precision_score(target.cpu(), output.cpu(), average="micro") * 100.0
        acc2 = BinaryF1Score()(output.cpu(), torch.unsqueeze(target, 1).cpu())

        batch_size = inp.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["average_precision_score"].update(
            acc1.item(), n=batch_size
        )
        metric_logger.meters["binary_f1_score"].update(acc2.item(), n=batch_size)

    print(
        f"* Average Precision: {metric_logger.average_precision_score.global_avg:.3f} F1: {metric_logger.binary_f1_score.global_avg:.3f} loss: {metric_logger.loss.global_avg:.3f}"
    )
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


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
        print(f"Unknow architecture: {arch}")
        sys.exit(1)
    model.cpu()
    model.eval()
    # load weights to evaluate
    utils.load_pretrained_weights(model, pretrained, checkpoint_key, arch, patch_size)

    print(f"Model {arch} built.")
    return model, embed_dim


def eval_linear(
    training_data: torch.utils.data.Dataset,
    dataset_val: torch.utils.data.Dataset,
    arch: Literal["vit_small"],
    device: str,
    pretrained: str | Path,
    avgpool_patchtokens: bool,
    patch_size: int,
    n_last_blocks: int,
    lr: float,
    batch_size: int,
    checkpoints_dir: str | Path,
    resume: bool,
    epochs: int,
    num_workers: int,
    checkpoint_key: str = "teacher",
):
    train_loader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
    )

    model, embed_dim = load_base_model(
        pretrained=pretrained,
        checkpoint_key=checkpoint_key,
        arch=arch,
        patch_size=patch_size,
        n_last_blocks=n_last_blocks,
        avgpool_patchtokens=avgpool_patchtokens,
    )
    linear_classifier = linear.LinearClassifier(embed_dim, num_labels=1)

    # attach to gpu/cpu
    model = model.to(device)
    linear_classifier = linear_classifier.to(device)
    print("model is cuda?", next(model.parameters()).is_cuda)
    print("classifier is cuda?", next(linear_classifier.parameters()).is_cuda)
    # TODO: add transforms back into ds
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # cvtransforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # cvtransforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    optimizer = torch.optim.SGD(
        linear_classifier.parameters(),
        lr,  # linear scaling rule
        momentum=0.9,
        weight_decay=0,  # we do not apply weight decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0)
    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_acc": 0.0}
    if resume:
        utils.restart_from_checkpoint(
            os.path.join(checkpoints_dir, "checkpoint.pth.tar"),
            run_variables=to_restore,
            state_dict=linear_classifier,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]
    os.makedirs(checkpoints_dir, exist_ok=True)
    for epoch in range(start_epoch, epochs):
        train_stats = train(
            model,
            linear_classifier,
            optimizer,
            train_loader,
            epoch,
            n_last_blocks,
            avgpool_patchtokens,
            device=device,
            arch=arch,
        )
        scheduler.step()

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
        }
        # if epoch == epochs - 1:
        test_stats = validate_network(
            val_loader,
            model,
            linear_classifier,
            n_last_blocks,
            avgpool_patchtokens,
            device=device,
            arch=arch,
        )
        print(
            f"Accuracy at epoch {epoch} of the network on the {len(dataset_val)} test images: {test_stats['average_precision_score']:.1f}%"
        )
        best_acc = max(best_acc, test_stats["average_precision_score"])
        print(f"Max accuracy so far: {best_acc:.2f}%")
        log_stats = {
            **{k: v for k, v in log_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
        }
        if utils.is_main_process():
            with (Path(checkpoints_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": linear_classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
            }
            torch.save(save_dict, os.path.join(checkpoints_dir, "checkpoint.pth.tar"))
    print(
        "Training of the supervised linear classifier on frozen features completed.\n"
        "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc)
    )


def load_yml(_input: Path | str):
    # TODO mv to common.py refactor other scripts to use
    with open(_input, "r") as f:
        args = yaml.safe_load(f)

    # #tests for later maybe
    # assert a1 == a2, "PAth and str are not same"
    return args


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
        _data_test = random_subset(_data_train, args.random_subset_frac, args.seed)
        # print("train info", _data_train.info)
        # print("test info", _data_test.info)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")
    print(f"testing size:{len(_data_test)}")
    eval_linear(
        training_data=_data_train,
        dataset_val=_data_test,
        arch=args.arch,
        pretrained=args.model_root,
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
