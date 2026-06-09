"""Extract video frames for ExitNav training."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import cv2
except ImportError:
    print("Missing opencv-python. Install dependencies with: pip install -r requirements.txt")
    sys.exit(1)

try:
    from .storage import TRAINING_DIR
except ImportError:
    from storage import TRAINING_DIR


def _safe_label(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip())
    cleaned = cleaned.strip("._-")
    if not cleaned:
        raise ValueError("Label cannot be empty.")
    return cleaned


def _frame_fingerprint(frame) -> int:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    avg = small.mean()
    bits = (small > avg).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _quality_reason(frame, min_blur: float, min_brightness: float) -> str | None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    if brightness < min_brightness:
        return f"dark ({brightness:.1f})"
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur < min_blur:
        return f"blurry ({blur:.1f})"
    return None


def extract_frames(
    video_path: str,
    label: str,
    fps: float = 1.0,
    output_root: Path = TRAINING_DIR,
    max_frames: int = 2000,
    filter_quality: bool = True,
    min_blur: float = 35.0,
    min_brightness: float = 18.0,
    duplicate_distance: int = 3,
) -> int:
    """Extract frames from a video into a class folder and return saved count."""
    safe_label = _safe_label(label)
    output_root.mkdir(parents=True, exist_ok=True)
    out_dir = output_root / safe_label
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fps = max(0.1, float(fps))
    step = max(1, int(round(video_fps / fps)))

    existing = sorted(out_dir.glob("frame_*.jpg"))
    start_idx = len(existing)

    saved = 0
    skipped_dark_blurry = 0
    skipped_duplicate = 0
    frame_idx = 0
    last_fingerprint: int | None = None
    while saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            if filter_quality:
                reason = _quality_reason(frame, min_blur, min_brightness)
                if reason:
                    skipped_dark_blurry += 1
                    frame_idx += 1
                    continue
                fingerprint = _frame_fingerprint(frame)
                if (
                    last_fingerprint is not None
                    and _hamming_distance(last_fingerprint, fingerprint) <= duplicate_distance
                ):
                    skipped_duplicate += 1
                    frame_idx += 1
                    continue
                last_fingerprint = fingerprint
            filename = out_dir / f"frame_{start_idx + saved:05d}.jpg"
            ok = cv2.imwrite(str(filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 84])
            if not ok:
                raise RuntimeError(f"Could not write frame: {filename}")
            saved += 1
        frame_idx += 1

    cap.release()
    print(
        f"[extract_frames] Label '{safe_label}': saved {saved} frames -> {out_dir} "
        f"(skipped quality={skipped_dark_blurry}, duplicates={skipped_duplicate})"
    )
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames for ExitNav training")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--label", required=True, help="Room/class label")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second")
    parser.add_argument("--max", type=int, default=2000, help="Maximum frames")
    parser.add_argument("--keep-low-quality", action="store_true", help="Disable blur/dark/duplicate filtering")
    args = parser.parse_args()

    n = extract_frames(
        args.video,
        args.label,
        args.fps,
        max_frames=args.max,
        filter_quality=not args.keep_low_quality,
    )
    print(f"Done: {n} frames for '{args.label}'")
