"""Train the ExitNav room classifier."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

try:
    from .storage import TRAINING_DIR
except ImportError:
    from storage import TRAINING_DIR

MODELS_DIR = Path(__file__).parent.parent / "models"
LABELS_FILE = MODELS_DIR / "labels.json"
MODEL_FILE = MODELS_DIR / "exitnav_model.pt"

LogFn = Callable[[str], None]


def _import_torch():
    try:
        import torch
        import torchvision

        return torch, torchvision
    except ImportError:
        print("Missing PyTorch. Install dependencies with: pip install -r requirements.txt")
        sys.exit(1)


def _log(log: LogFn | None, message: str) -> None:
    (log or print)(message)


def _create_model(models, nn, num_classes: int, log: LogFn | None):
    try:
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_small(weights=weights)
        _log(log, "Model: MobileNetV3-Small with ImageNet weights.")
    except Exception as exc:
        _log(log, f"ImageNet weights unavailable, using fresh weights: {exc}")
        model = models.mobilenet_v3_small(weights=None)

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def train(
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 5e-4,
    img_size: int = 224,
    log: LogFn | None = None,
) -> float:
    torch, _ = _import_torch()
    from torch import nn
    from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
    from torchvision import datasets, transforms, models

    epochs = max(1, int(epochs))
    batch_size = max(1, int(batch_size))
    lr = max(1e-7, float(lr))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    classes = sorted(d.name for d in TRAINING_DIR.iterdir() if d.is_dir())
    if len(classes) < 2:
        raise RuntimeError(
            f"Need at least 2 classes in {TRAINING_DIR}. Found: {classes or 'none'}"
        )

    _log(log, f"Training data: {TRAINING_DIR}")
    _log(log, f"Classes ({len(classes)}): {', '.join(classes)}")

    label_map = {c: i for i, c in enumerate(classes)}
    with open(LABELS_FILE, "w", encoding="utf-8") as f:
        json.dump({"classes": classes, "label_map": label_map}, f, ensure_ascii=False, indent=2)
    _log(log, f"Labels saved: {LABELS_FILE}")

    train_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(8),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    split_ds = datasets.ImageFolder(str(TRAINING_DIR), transform=None)
    if len(split_ds) < 2:
        raise RuntimeError(f"Need at least 2 training images in {TRAINING_DIR}.")

    targets = list(split_ds.targets)
    class_counts = {classes[i]: targets.count(i) for i in range(len(classes))}
    _log(log, "Class counts: " + ", ".join(f"{name}={count}" for name, count in class_counts.items()))

    generator = torch.Generator().manual_seed(42)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for class_idx in range(len(classes)):
        indices = torch.tensor([i for i, target in enumerate(targets) if target == class_idx])
        perm = indices[torch.randperm(len(indices), generator=generator)].tolist()
        if len(perm) < 2:
            train_indices.extend(perm)
            continue
        val_count = max(1, int(round(len(perm) * 0.15)))
        val_indices.extend(perm[:val_count])
        train_indices.extend(perm[val_count:])

    if not val_indices:
        raise RuntimeError("Need at least 2 images in one class to build a validation split.")

    train_base = datasets.ImageFolder(str(TRAINING_DIR), transform=train_tf)
    val_base = datasets.ImageFolder(str(TRAINING_DIR), transform=val_tf)
    train_ds = Subset(train_base, train_indices)
    val_ds = Subset(val_base, val_indices)
    n_train = len(train_indices)
    n_val = len(val_indices)

    train_targets = [targets[i] for i in train_indices]
    train_class_counts = torch.bincount(torch.tensor(train_targets), minlength=len(classes)).float()
    class_weights = (train_class_counts.sum() / train_class_counts.clamp_min(1.0)).to(torch.float32)
    sample_weights = [float(class_weights[target]) for target in train_targets]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )

    # num_workers=0 avoids Windows/FastAPI background-thread deadlocks.
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory
    )
    _log(
        log,
        f"Dataset: {n_train} train / {n_val} val images, batch {batch_size}, img {img_size}px.",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(log, f"Device: {device}")

    model = _create_model(models, nn, len(classes), log).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = -1.0
    total_batches = max(1, len(train_loader))
    _log(log, f"Starting training: {epochs} epochs, {total_batches} batches/epoch, lr {lr:g}.")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch_idx, (imgs, labels) in enumerate(train_loader, start=1):
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            batch_size_actual = imgs.size(0)
            total_loss += loss.item() * batch_size_actual
            correct += (out.argmax(1) == labels).sum().item()
            total += batch_size_actual

            if batch_idx == 1 or batch_idx == total_batches or batch_idx % 5 == 0:
                _log(
                    log,
                    f"Epoch {epoch}/{epochs} batch {batch_idx}/{total_batches} "
                    f"loss {loss.item():.4f}",
                )

        train_acc = correct / max(1, total)
        train_loss = total_loss / max(1, total)

        model.eval()
        vcorrect, vtotal = 0, 0
        per_class_correct = [0 for _ in classes]
        per_class_total = [0 for _ in classes]
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                out = model(imgs)
                pred = out.argmax(1)
                matches = pred == labels
                vcorrect += matches.sum().item()
                vtotal += imgs.size(0)
                for class_idx in range(len(classes)):
                    mask = labels == class_idx
                    per_class_total[class_idx] += mask.sum().item()
                    per_class_correct[class_idx] += (matches & mask).sum().item()
        val_acc = vcorrect / max(1, vtotal)
        scheduler.step()

        _log(
            log,
            f"Epoch {epoch}/{epochs} done | loss {train_loss:.4f} | "
            f"train acc {train_acc:.3f} | val acc {val_acc:.3f}",
        )
        per_class_log = []
        for class_idx, name in enumerate(classes):
            if per_class_total[class_idx]:
                acc = per_class_correct[class_idx] / per_class_total[class_idx]
                per_class_log.append(f"{name} {acc:.3f}")
        if per_class_log:
            _log(log, "Validation by class: " + " | ".join(per_class_log))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "classes": classes,
                    "val_acc": val_acc,
                    "img_size": img_size,
                },
                MODEL_FILE,
            )
            _log(log, f"Best model saved: val acc {val_acc:.3f} -> {MODEL_FILE}")

    _log(log, f"Training finished. Best val accuracy: {best_val_acc:.3f}")
    return best_val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ExitNav room model training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch, lr=args.lr)
