import os
import argparse



os.environ['HF_HOME'] = '/nfs2/users/yszhang/tool_checkpoints/huggingface'
os.environ["TORCH_HOME"] = "/nfs2/users/yszhang/tool_checkpoints/huggingface"

parser = argparse.ArgumentParser()
parser.add_argument("--train_parquet", nargs="+", required=True)
parser.add_argument("--val_parquet", nargs="+", required=True)
parser.add_argument("--save_dir", type=str, default="./checkpoints")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--bs", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-4)
# parser.add_argument("--num_classes", type=int, default=7)
parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet50", "resnet18_trunc2", "resnet18_trunc3"])
parser.add_argument("--resume_path", type=str, default=None)
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--cuda_id", type=int, default=0)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_id)

import time
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from dataloader import build_dataloaders


class ResNet18Truncated(nn.Module):
    def __init__(self, num_classes=7, depth=3, pretrained=True):
        """
        depth:
            2 -> conv1 + layer1 + layer2
            3 -> conv1 + layer1 + layer2 + layer3
        """
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base = models.resnet18(weights=weights)

        if depth == 2:
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
            raise ValueError("depth must be 2 or 3")

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class DegradationClassifier(nn.Module):
    def __init__(self, num_classes: int = 7, backbone: str = "resnet18", pretrained: bool = True, resume_path: str = None):
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

        elif backbone == "resnet18_trunc2":
            self.model = ResNet18Truncated(num_classes, depth=2, pretrained=pretrained)

        elif backbone == "resnet18_trunc3":
            self.model = ResNet18Truncated(num_classes, depth=3, pretrained=pretrained)

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        if resume_path is not None:
            print(f"Loading checkpoint from {resume_path}")
            
            checkpoint = torch.load(resume_path, map_location="cpu")
            state_dict = checkpoint["model_state_dict"]
            state_dict = {
                k.replace("model.", ""): v
                for k, v in state_dict.items()
            }
            self.model.load_state_dict(state_dict)

            print("Checkpoint loaded.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def multilabel_metrics_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    logits: [B, K]
    targets: [B, K], values in {0,1}
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    micro_f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # exact match ratio: all labels in one sample must be correct
    exact_match = (preds == targets).all(dim=1).float().mean()

    # label-wise accuracy over all positions
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
        LQ, degra_tensor = batch

        LQ = LQ.to(device, non_blocking=True)
        degra_tensor = degra_tensor.float().to(device, non_blocking=True)

        optimizer.zero_grad()

        logits = model(LQ)
        loss = criterion(logits, degra_tensor)

        loss.backward()
        optimizer.step()

        batch_size = LQ.size(0)
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
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_logits = []
    all_targets = []

    progress_bar = tqdm(loader, desc="Val", leave=False)

    for batch in progress_bar:
        LQ, degra_tensor = batch

        LQ = LQ.to(device, non_blocking=True)
        degra_tensor = degra_tensor.float().to(device, non_blocking=True)

        logits = model(LQ)
        loss = criterion(logits, degra_tensor)

        batch_size = LQ.size(0)
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
    val_loaders: list,
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
    print(f"Train samples: {len(train_loader.dataset)}, train batches: {len(train_loader)}")
    print(f"Validation loaders: {len(val_loaders)}")
    for i, val_loader in enumerate(val_loaders):
        print(f"Val[{i}] samples: {len(val_loader.dataset)}, batches: {len(val_loader)}")
    print(f"Model parameters: total={total_params:,}, trainable={trainable_params:,}")
    print("=============================\n")


def save_checkpoint(
    save_path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    best_val_f1: float,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_f1": best_val_f1,
        },
        save_path,
    )


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


def main():
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loaders = build_dataloaders(args)

    model = DegradationClassifier(
        num_classes=args.num_classes,
        backbone=args.backbone,
        pretrained=True,
        resume_path=args.resume_path,
    ).to(device)

    print_run_info(
        train_loader=train_loader,
        val_loaders=val_loaders,
        model=model,
        device=device,
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_f1 = -1.0
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

        val_metrics_list = []

        print_metrics("Train", train_metrics)

        for i, val_loader in enumerate(val_loaders):

            val_metrics = validate_one_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                threshold=args.threshold,
            )

            val_metrics_list.append(val_metrics)

            print_metrics(f"Val[{i}]", val_metrics)

        save_checkpoint(
            save_path=last_ckpt_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_f1=best_val_f1,
        )

        mean_val_f1 = sum(v["micro_f1"] for v in val_metrics_list) / len(val_metrics_list)
        if mean_val_f1 > best_val_f1:
            best_val_f1 = mean_val_f1
            save_checkpoint(
                save_path=best_ckpt_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_f1=best_val_f1,
            )
            print(f"Saved new best checkpoint to: {best_ckpt_path}")

        elapsed = time.time() - start_time
        print(f"Epoch [{epoch}/{args.epochs}] finished in {elapsed:.1f}s\n")

    print("\nTraining finished.")
    print(f"Best val micro-F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    main()
