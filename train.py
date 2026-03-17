import os
import argparse
import sys
import csv
import time
from datetime import datetime
from typing import Dict, Tuple, List

os.environ["HF_HOME"] = "/nfs2/users/yszhang/tool_checkpoints/huggingface"
os.environ["TORCH_HOME"] = "/nfs2/users/yszhang/tool_checkpoints/huggingface"

parser = argparse.ArgumentParser()
parser.add_argument("--train_parquet", nargs="+", default=[os.path.join("datasets", "train.parquet")])
parser.add_argument("--val_parquet", nargs="+", default=[os.path.join("datasets", "val.parquet")])
parser.add_argument("--test_parquet", nargs="+", default=None)
parser.add_argument("--save_dir", type=str, default="./checkpoints")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--bs", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument(
    "--backbone",
    type=str,
    default="resnet18",
    choices=[
        "resnet18",
        "resnet50",
        "resnet18_trunc1",
        "resnet18_trunc2",
        "resnet18_trunc3",
        "customized_cnn",
    ],
)
parser.add_argument("--base_channels", type=int, default=32, help="Base channels for customized CNN backbone")
parser.add_argument("--no_pretrained", action="store_true")
parser.add_argument("--resume_path", type=str, default=None)
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--cuda_id", type=int, default=0)
parser.add_argument(
    "--model_selection_mode",
    type=str,
    default="mean",
    choices=["mean", "single"],
    help="How to select best model: mean over all validation sets, or use one specific validation set.",
)
parser.add_argument(
    "--model_selection_idx",
    type=int,
    default=0,
    help="Validation set index used when model_selection_mode='single'.",
)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_id)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from dataloader import build_dataloaders


class ResNet18Truncated(nn.Module):
    def __init__(self, num_classes: int = 8, depth: int = 3, pretrained: bool = True):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base = models.resnet18(weights=weights)

        if depth == 1:
            self.features = nn.Sequential(
                base.conv1,
                base.bn1,
                base.relu,
                base.maxpool,
                base.layer1,
            )
            out_channels = 64

        elif depth == 2:
            self.features = nn.Sequential(
                base.conv1,
                base.bn1,
                base.relu,
                base.maxpool,
                base.layer1,
                base.layer2,
            )
            out_channels = 128

        elif depth == 3:
            self.features = nn.Sequential(
                base.conv1,
                base.bn1,
                base.relu,
                base.maxpool,
                base.layer1,
                base.layer2,
                base.layer3,
            )
            out_channels = 256

        else:
            raise ValueError("depth must be 1, 2 or 3")

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(out_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MultiScaleConcatBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, branch_ratio: int = 4):
        super().__init__()

        branch_channels = max(out_channels // branch_ratio, 8)

        self.branch3 = DepthwiseSeparableConv(in_channels, branch_channels, kernel_size=3, stride=stride)
        self.branch5 = DepthwiseSeparableConv(in_channels, branch_channels, kernel_size=5, stride=stride)
        self.branch7 = DepthwiseSeparableConv(in_channels, branch_channels, kernel_size=7, stride=stride)

        fused_channels = branch_channels * 3

        self.fuse = nn.Sequential(
            nn.Conv2d(fused_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x3 = self.branch3(x)
        x5 = self.branch5(x)
        x7 = self.branch7(x)
        x = torch.cat([x3, x5, x7], dim=1)
        x = self.fuse(x)
        return x


class MultiScaleTinyCNN(nn.Module):
    def __init__(self, num_classes: int = 8, base_channels: int = 32):
        super().__init__()

        self.features = nn.Sequential(
            MultiScaleConcatBlock(3, base_channels, stride=2),
            MultiScaleConcatBlock(base_channels, base_channels * 2, stride=2),
            MultiScaleConcatBlock(base_channels * 2, base_channels * 4, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(base_channels * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class DegradationClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 8,
        backbone: str = "resnet18",
        pretrained: bool = True,
        resume_path: str = None,
        base_channels: int = 32,
    ):
        super().__init__()

        if backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
            self.model = model

        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            model = models.resnet50(weights=weights)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
            self.model = model

        elif backbone == "resnet18_trunc1":
            self.model = ResNet18Truncated(num_classes=num_classes, depth=1, pretrained=pretrained)

        elif backbone == "resnet18_trunc2":
            self.model = ResNet18Truncated(num_classes=num_classes, depth=2, pretrained=pretrained)

        elif backbone == "resnet18_trunc3":
            self.model = ResNet18Truncated(num_classes=num_classes, depth=3, pretrained=pretrained)

        elif backbone == "customized_cnn":
            self.model = MultiScaleTinyCNN(num_classes=num_classes, base_channels=base_channels)

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        if resume_path is not None:
            print(f"Loading checkpoint from {resume_path}")
            checkpoint = torch.load(resume_path, map_location="cpu")
            state_dict = checkpoint["model_state_dict"]
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)
            print("Checkpoint loaded.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@torch.no_grad()
def multilabel_metrics_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    micro_f1 = 2 * precision * recall / (precision + recall + 1e-8)

    exact_match = (preds == targets).all(dim=1).float().mean()
    label_acc = (preds == targets).float().mean()

    return {
        "micro_precision": precision.item(),
        "micro_recall": recall.item(),
        "micro_f1": micro_f1.item(),
        "exact_match": exact_match.item(),
        "label_acc": label_acc.item(),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    threshold: float,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_samples = 0
    all_logits = []
    all_targets = []

    progress_bar = tqdm(loader, desc="Train", leave=False)

    for batch in progress_bar:
        lq, degra_tensor = batch

        lq = lq.to(device, non_blocking=True)
        degra_tensor = degra_tensor.float().to(device, non_blocking=True)

        optimizer.zero_grad()

        logits = model(lq)
        loss = criterion(logits, degra_tensor)

        loss.backward()
        optimizer.step()

        batch_size = lq.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_logits.append(logits.detach())
        all_targets.append(degra_tensor.detach())

        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    metrics = multilabel_metrics_from_logits(all_logits, all_targets, threshold=threshold)
    metrics["loss"] = total_loss / max(total_samples, 1)
    return metrics


@torch.no_grad()
def evaluate_one_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
    desc: str = "Eval",
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_samples = 0
    all_logits = []
    all_targets = []

    progress_bar = tqdm(loader, desc=desc, leave=False)

    for batch in progress_bar:
        lq, degra_tensor = batch

        lq = lq.to(device, non_blocking=True)
        degra_tensor = degra_tensor.float().to(device, non_blocking=True)

        logits = model(lq)
        loss = criterion(logits, degra_tensor)

        batch_size = lq.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_logits.append(logits)
        all_targets.append(degra_tensor)

        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    metrics = multilabel_metrics_from_logits(all_logits, all_targets, threshold=threshold)
    metrics["loss"] = total_loss / max(total_samples, 1)
    return metrics


def get_model_selection_score(
    val_metrics_list: List[Dict[str, float]],
    mode: str = "mean",
    idx: int = 0,
) -> float:
    if len(val_metrics_list) == 0:
        raise ValueError("val_metrics_list is empty.")

    if mode == "mean":
        return sum(v["micro_f1"] for v in val_metrics_list) / len(val_metrics_list)

    if mode == "single":
        if idx < 0 or idx >= len(val_metrics_list):
            raise ValueError(
                f"model_selection_idx={idx} is out of range for {len(val_metrics_list)} validation loaders."
            )
        return val_metrics_list[idx]["micro_f1"]

    raise ValueError(f"Unsupported model_selection_mode: {mode}")


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total_params, trainable_params


def print_run_info(
    train_loader: DataLoader,
    val_loaders: List[DataLoader],
    test_loaders: List[DataLoader],
    model: nn.Module,
    device: torch.device,
):
    print("\n===== Run Configuration =====")
    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"Visible CUDA devices: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("GPU name: N/A (running on CPU)")

    total_params, trainable_params = count_parameters(model)

    print(f"Backbone: {args.backbone}")
    print(f"Resume checkpoint: {args.resume_path or 'None'}")
    print(f"Epochs: {args.epochs}, batch size: {args.bs}, lr: {args.lr}, weight decay: {args.weight_decay}")
    print(f"Threshold: {args.threshold}, seed: {args.seed}")
    print(f"Model selection mode: {args.model_selection_mode}")
    if args.model_selection_mode == "single":
        print(f"Model selection val index: {args.model_selection_idx}")

    print(f"Train samples: {len(train_loader.dataset)}, train batches: {len(train_loader)}")

    print(f"Validation loaders: {len(val_loaders)}")
    for i, val_loader in enumerate(val_loaders):
        print(f"Val[{i}] samples: {len(val_loader.dataset)}, batches: {len(val_loader)}")

    print(f"Test loaders: {len(test_loaders)}")
    for i, test_loader in enumerate(test_loaders):
        print(f"Test[{i}] samples: {len(test_loader.dataset)}, batches: {len(test_loader)}")

    print(f"Model parameters: total={total_params:,}, trainable={trainable_params:,}")
    print("=============================\n")


def save_checkpoint(
    save_path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    best_selection_score: float,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_selection_score": best_selection_score,
        },
        save_path,
    )


def load_checkpoint_for_eval(model: nn.Module, ckpt_path: str, device: torch.device) -> dict:
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def print_metrics(prefix: str, metrics: Dict[str, float]):
    print(
        f"{prefix} | "
        f"loss={metrics['loss']:.4f}, "
        f"micro_f1={metrics['micro_f1']:.4f}, "
        f"precision={metrics['micro_precision']:.4f}, "
        f"recall={metrics['micro_recall']:.4f}, "
        f"label_acc={metrics['label_acc']:.4f}, "
        f"exact_match={metrics['exact_match']:.4f}"
    )


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def setup_logging(save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(save_dir, f"train_{timestamp}.log")
    log_file = open(log_path, "a", buffering=1, encoding="utf-8")

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    return log_path, log_file, original_stdout, original_stderr


def append_metrics_to_csv(csv_path: str, row: dict):
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    os.makedirs(args.save_dir, exist_ok=True)

    log_path, log_file, original_stdout, original_stderr = setup_logging(args.save_dir)
    metrics_csv_path = os.path.join(args.save_dir, "metrics.csv")
    test_metrics_csv_path = os.path.join(args.save_dir, "test_metrics.csv")

    print(f"Log file: {log_path}")

    try:
        set_seed(args.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_loader, val_loaders, test_loaders = build_dataloaders(args)

        model = DegradationClassifier(
            num_classes=8,
            backbone=args.backbone,
            pretrained=not args.no_pretrained,
            resume_path=args.resume_path,
            base_channels=args.base_channels,
        ).to(device)

        print_run_info(
            train_loader=train_loader,
            val_loaders=val_loaders,
            test_loaders=test_loaders,
            model=model,
            device=device,
        )

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_selection_score = -1.0
        best_epoch = -1

        best_ckpt_path = os.path.join(args.save_dir, "best_model.pt")
        last_ckpt_path = os.path.join(args.save_dir, "last_model.pt")

        for epoch in range(1, args.epochs + 1):
            start_time = time.time()
            print(f"Epoch [{epoch}/{args.epochs}]")

            train_metrics = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                threshold=args.threshold,
            )

            print_metrics("Train", train_metrics)

            val_metrics_list = []
            for i, val_loader in enumerate(val_loaders):
                val_metrics = evaluate_one_loader(
                    model=model,
                    loader=val_loader,
                    criterion=criterion,
                    device=device,
                    threshold=args.threshold,
                    desc=f"Val[{i}]",
                )
                val_metrics_list.append(val_metrics)
                print_metrics(f"Val[{i}]", val_metrics)

            save_checkpoint(
                save_path=last_ckpt_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_selection_score=best_selection_score,
            )

            mean_val_f1 = sum(v["micro_f1"] for v in val_metrics_list) / len(val_metrics_list)

            selection_score = get_model_selection_score(
                val_metrics_list=val_metrics_list,
                mode=args.model_selection_mode,
                idx=args.model_selection_idx,
            )

            if selection_score > best_selection_score:
                best_selection_score = selection_score
                best_epoch = epoch
                save_checkpoint(
                    save_path=best_ckpt_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_selection_score=best_selection_score,
                )
                print(f"Saved new best checkpoint to: {best_ckpt_path}")

            elapsed = time.time() - start_time
            print(f"Epoch [{epoch}/{args.epochs}] finished in {elapsed:.1f}s\n")

            row = {
                "epoch": epoch,
                "backbone": args.backbone,
                "train_loss": train_metrics["loss"],
                "train_micro_f1": train_metrics["micro_f1"],
                "train_precision": train_metrics["micro_precision"],
                "train_recall": train_metrics["micro_recall"],
                "train_label_acc": train_metrics["label_acc"],
                "train_exact_match": train_metrics["exact_match"],
                "mean_val_micro_f1": mean_val_f1,
                "model_selection_mode": args.model_selection_mode,
                "model_selection_idx": args.model_selection_idx if args.model_selection_mode == "single" else -1,
                "selection_score": selection_score,
                "best_selection_score_so_far": best_selection_score,
                "best_epoch_so_far": best_epoch,
                "elapsed_sec": elapsed,
            }

            for i, val_metrics in enumerate(val_metrics_list):
                row[f"val{i}_loss"] = val_metrics["loss"]
                row[f"val{i}_micro_f1"] = val_metrics["micro_f1"]
                row[f"val{i}_precision"] = val_metrics["micro_precision"]
                row[f"val{i}_recall"] = val_metrics["micro_recall"]
                row[f"val{i}_label_acc"] = val_metrics["label_acc"]
                row[f"val{i}_exact_match"] = val_metrics["exact_match"]

            append_metrics_to_csv(metrics_csv_path, row)

        print("\nTraining finished.")
        print(f"Best selection score: {best_selection_score:.4f}")
        print(f"Best epoch: {best_epoch}")
        print(f"Metrics CSV saved to: {metrics_csv_path}")

        if test_loaders is not None and len(test_loaders) > 0:
            print("\n===== Final Test with Best Checkpoint =====")

            checkpoint = load_checkpoint_for_eval(model, best_ckpt_path, device=device)
            loaded_best_epoch = checkpoint.get("epoch", best_epoch)
            loaded_best_selection_score = checkpoint.get("best_selection_score", best_selection_score)

            print(f"Loaded best checkpoint from: {best_ckpt_path}")
            print(f"Best checkpoint epoch: {loaded_best_epoch}")
            print(f"Best checkpoint selection score: {loaded_best_selection_score:.4f}")

            test_row = {
                "backbone": args.backbone,
                "best_epoch": loaded_best_epoch,
                "best_selection_score": loaded_best_selection_score,
                "model_selection_mode": args.model_selection_mode,
                "model_selection_idx": args.model_selection_idx if args.model_selection_mode == "single" else -1,
                "threshold": args.threshold,
                "save_dir": args.save_dir,
            }

            for i, test_loader in enumerate(test_loaders):
                test_metrics = evaluate_one_loader(
                    model=model,
                    loader=test_loader,
                    criterion=criterion,
                    device=device,
                    threshold=args.threshold,
                    desc=f"Test[{i}]",
                )
                print_metrics(f"Test[{i}]", test_metrics)

                test_row[f"test{i}_loss"] = test_metrics["loss"]
                test_row[f"test{i}_micro_f1"] = test_metrics["micro_f1"]
                test_row[f"test{i}_precision"] = test_metrics["micro_precision"]
                test_row[f"test{i}_recall"] = test_metrics["micro_recall"]
                test_row[f"test{i}_label_acc"] = test_metrics["label_acc"]
                test_row[f"test{i}_exact_match"] = test_metrics["exact_match"]

            append_metrics_to_csv(test_metrics_csv_path, test_row)
            append_metrics_to_csv(metrics_csv_path, {"epoch": "final_test", **test_row})

            print(f"Test metrics CSV saved to: {test_metrics_csv_path}")
            print("===== End Final Test =====\n")
        else:
            print("\nNo test loaders provided. Skipping final test evaluation.")

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


if __name__ == "__main__":
    main()
