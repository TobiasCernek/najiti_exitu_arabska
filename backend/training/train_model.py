"""
train_model.py – Trénink klasifikátoru místností (MobileNetV3-Small).

Struktura trénovacích dat:
    backend/data/training/
        ucebna_101/   ← framy z učebny 101
        ucebna_102/
        chodba_1np/
        ...

Použití:
    python train_model.py
    python train_model.py --epochs 30 --batch 32 --lr 0.001
"""

import argparse
import json
import sys
from pathlib import Path

TRAINING_DIR = Path(__file__).parent.parent / "data" / "training"
MODELS_DIR   = Path(__file__).parent.parent / "models"
LABELS_FILE  = MODELS_DIR / "labels.json"
MODEL_FILE   = MODELS_DIR / "exitnav_model.pt"

# --------------------------------------------------------------------------- #
# Lazy imports – dáme hezkou chybu pokud chybí balíčky
# --------------------------------------------------------------------------- #
def _import_torch():
    try:
        import torch
        import torchvision
        return torch, torchvision
    except ImportError:
        print("Chybí PyTorch. Nainstalujte:\n  pip install torch torchvision")
        sys.exit(1)


def train(epochs: int = 20, batch_size: int = 16, lr: float = 5e-4, img_size: int = 224):
    torch, torchvision = _import_torch()
    from torch import nn
    from torch.utils.data import DataLoader, random_split
    from torchvision import datasets, transforms, models

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Zjisti třídy (složky v training/)
    # ------------------------------------------------------------------ #
    classes = sorted([d.name for d in TRAINING_DIR.iterdir() if d.is_dir()])
    if len(classes) < 2:
        print(f"Potřebuji alespoň 2 třídy (složky) v {TRAINING_DIR}. Nalezeno: {classes}")
        sys.exit(1)
    print(f"Třídy ({len(classes)}): {classes}")

    # Uložení mapování label → index
    label_map = {c: i for i, c in enumerate(classes)}
    with open(LABELS_FILE, "w", encoding="utf-8") as f:
        json.dump({"classes": classes, "label_map": label_map}, f, ensure_ascii=False, indent=2)
    print(f"Labels uloženy → {LABELS_FILE}")

    # ------------------------------------------------------------------ #
    # 2. Dataset a augmentace
    # ------------------------------------------------------------------ #
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    full_ds = datasets.ImageFolder(str(TRAINING_DIR), transform=train_tf)
    n_val   = max(1, int(len(full_ds) * 0.15))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    val_ds.dataset.transform = val_tf  # type: ignore

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    print(f"Dataset: {n_train} trén. / {n_val} val. snímků")

    # ------------------------------------------------------------------ #
    # 3. Model – MobileNetV3-Small (lehký, rychlý, vhodný pro produkci)
    # ------------------------------------------------------------------ #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Zařízení: {device}")

    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    # Nahraď klasifikační hlavu
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ------------------------------------------------------------------ #
    # 4. Tréninkový cyklus
    # ------------------------------------------------------------------ #
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct    += (out.argmax(1) == labels).sum().item()
            total      += imgs.size(0)
        train_acc  = correct / total
        train_loss = total_loss / total

        # --- Validation ---
        model.eval()
        vcorrect, vtotal = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                vcorrect += (out.argmax(1) == labels).sum().item()
                vtotal   += imgs.size(0)
        val_acc = vcorrect / vtotal

        scheduler.step()

        print(f"Epoch {epoch:3d}/{epochs} │ "
              f"loss {train_loss:.4f} │ train acc {train_acc:.3f} │ val acc {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "classes":   classes,
                "val_acc":   val_acc,
                "img_size":  img_size,
            }, MODEL_FILE)
            print(f"  ✓ Nejlepší model uložen (val acc {val_acc:.3f}) → {MODEL_FILE}")

    print(f"\nTrénink dokončen. Nejlepší val accuracy: {best_val_acc:.3f}")
    return best_val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ExitNav – trénink modelu místností")
    parser.add_argument("--epochs", type=int,   default=20,   help="Počet epoch")
    parser.add_argument("--batch",  type=int,   default=16,   help="Velikost batche")
    parser.add_argument("--lr",     type=float, default=5e-4, help="Learning rate")
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch, lr=args.lr)
