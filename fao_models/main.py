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
from torchmetrics.classification import BinaryF1Score
from torchmetrics.classification import BinaryPrecision
from torchmetrics.classification import BinaryRecall

from models.dino import utils
from models._models import get_model


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

        acc1 = (
            average_precision_score(target.cpu(), output.cpu(), average="micro") * 100.0
        )
        _output = output.cpu()
        _target = torch.unsqueeze(target, 1).cpu()
        acc2 = BinaryF1Score()(_output, _target)
        acc3 = BinaryPrecision()(_output, _target)
        acc4 = BinaryRecall()(_output, _target)

        batch_size = inp.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["average_precision_score"].update(
            acc1.item(), n=batch_size
        )
        metric_logger.meters["binary_f1_score"].update(acc2.item(), n=batch_size)
        metric_logger.meters["binary_precision_score"].update(acc3.item(), n=batch_size)
        metric_logger.meters["binary_recall_score"].update(acc4.item(), n=batch_size)

    print(
        f"* Average Precision: {metric_logger.average_precision_score.global_avg:.3f} "
        f"* F1: {metric_logger.binary_f1_score.global_avg:.3f} "
        f"* Precision {metric_logger.binary_precision_score.global_avg:.3f} "
        f"* Recall {metric_logger.binary_recall_score.global_avg:.3f} "
        f"* loss: {metric_logger.loss.global_avg:.3f} "
    )
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def eval_linear(
    training_data: torch.utils.data.Dataset,
    dataset_val: torch.utils.data.Dataset,
    arch: Literal["vit_small"],
    device: str,
    model_root: str | Path,
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

    model, linear_classifier = get_model(
        model_name="linear-dino",
        model_root=model_root,
        checkpoint_key=checkpoint_key,
        arch=arch,
        patch_size=patch_size,
        n_last_blocks=n_last_blocks,
        avgpool_patchtokens=avgpool_patchtokens,
    )
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
