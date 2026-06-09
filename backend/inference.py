"""
inference.py – Rozpoznávání místnosti z obrázku pomocí natrénovaného modelu.
"""

import base64
import io
import json
import sys
from pathlib import Path
from typing import Optional

MODELS_DIR  = Path(__file__).parent / "models"
MODEL_FILE  = MODELS_DIR / "exitnav_model.pt"
LABELS_FILE = MODELS_DIR / "labels.json"

# Lazy load – PyTorch se načte jen jednou
_model    = None
_classes  = None
_device   = None
_img_size = 224
_transform = None


def _load_model():
    global _model, _classes, _device, _img_size, _transform

    try:
        import torch
        from torchvision import transforms, models
        from torch import nn
    except ImportError:
        raise RuntimeError("Chybí PyTorch. Nainstalujte: pip install torch torchvision")

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model nenalezen: {MODEL_FILE}\n"
            "Nejprve natrénujte model přes admin panel."
        )

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_FILE, map_location=_device)
    _classes  = checkpoint["classes"]
    _img_size = checkpoint.get("img_size", 224)

    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(_classes))
    model.load_state_dict(checkpoint["model"])
    model.to(_device)
    model.eval()
    _model = model

    _transform = transforms.Compose([
        transforms.Resize((_img_size, _img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    print(f"[inference] Model načten. Třídy: {_classes} | Zařízení: {_device}")

    # Synchronizuj labels.json s daty v checkpointu
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LABELS_FILE, "w", encoding="utf-8") as f:
        json.dump({"classes": _classes}, f, ensure_ascii=False)


def predict_from_base64(b64_image: str, top_k: int = 3) -> dict:
    """
    Přijme base64 JPEG/PNG string (s nebo bez data URI prefixu),
    vrátí předpověď třídy a top-k pravděpodobnosti.

    Returns:
        {
            "label": "ucebna_101",
            "confidence": 0.92,
            "top": [{"label": ..., "prob": ...}, ...]
        }
    """
    import torch
    from PIL import Image

    global _model
    if _model is None:
        _load_model()

    # Odstraň data URI prefix pokud je přítomen
    if "," in b64_image:
        b64_image = b64_image.split(",", 1)[1]

    img_bytes = base64.b64decode(b64_image)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    tensor = _transform(img).unsqueeze(0).to(_device)  # type: ignore

    with torch.no_grad():
        logits = _model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    top_probs, top_idxs = probs.topk(min(top_k, len(_classes)))  # type: ignore
    top = [
        {"label": _classes[i], "prob": round(float(p), 4)}
        for i, p in zip(top_idxs.tolist(), top_probs.tolist())
    ]
    margin = top[0]["prob"] - (top[1]["prob"] if len(top) > 1 else 0.0)

    return {
        "label":      top[0]["label"],
        "confidence": top[0]["prob"],
        "margin": round(float(margin), 4),
        "uncertain": top[0]["prob"] < 0.62 or margin < 0.12,
        "top":        top,
    }


def is_model_available() -> bool:
    """Model je dostupný pokud existuje soubor .pt (labels.json se generuje automaticky)."""
    return MODEL_FILE.exists()


def get_classes() -> Optional[list[str]]:
    """Vrátí třídy z labels.json nebo přímo z modelu."""
    if LABELS_FILE.exists():
        with open(LABELS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)["classes"]
    # Fallback: načti přímo z checkpointu bez načtení celého modelu
    try:
        import torch
        checkpoint = torch.load(MODEL_FILE, map_location="cpu")
        return checkpoint.get("classes")
    except Exception:
        return None


def reset_model_cache() -> None:
    """Force the next prediction to load the newest checkpoint from disk."""
    global _model, _classes, _device, _img_size, _transform
    _model = None
    _classes = None
    _device = None
    _img_size = 224
    _transform = None
