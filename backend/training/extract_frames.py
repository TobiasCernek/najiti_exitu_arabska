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
) -> dict:
    """Extract frames from a video into a class folder and return saved count.

    Frame selection is timestamp-based (using each decoded frame's
    presentation time) rather than purely index/step-based. This avoids
    under-sampling on containers/codecs (commonly .mov / HEVC) where
    ``CAP_PROP_FPS`` is unreliable or where OpenCV reports a frame count
    that doesn't match how many frames can actually be decoded.
    """
    safe_label = _safe_label(label)
    output_root.mkdir(parents=True, exist_ok=True)
    out_dir = output_root / safe_label
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    reported_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    reported_frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    fps = max(0.1, float(fps))
    interval_sec = 1.0 / fps

    existing = sorted(out_dir.glob("frame_*.jpg"))
    start_idx = len(existing)

    saved = 0
    skipped_dark_blurry = 0
    skipped_duplicate = 0
    frame_idx = 0
    last_fingerprint: int | None = None
    next_target_sec = 0.0
    last_timestamp_sec = 0.0
    fallback_to_index_step = False

    while saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        timestamp_sec = timestamp_ms / 1000.0 if timestamp_ms and timestamp_ms > 0 else None

        if timestamp_sec is None:
            # Some codecs/containers never report a valid POS_MSEC (stays 0).
            # Fall back to evenly-spaced index sampling using the reported
            # FPS (or a sane default if that's also missing/zero).
            fallback_to_index_step = True
            effective_fps = reported_fps if reported_fps > 1 else 25.0
            step = max(1, int(round(effective_fps / fps)))
            take = frame_idx % step == 0
        else:
            last_timestamp_sec = timestamp_sec
            take = timestamp_sec + 1e-6 >= next_target_sec

        if take:
            if filter_quality:
                reason = _quality_reason(frame, min_blur, min_brightness)
                if reason:
                    skipped_dark_blurry += 1
                    frame_idx += 1
                    if not fallback_to_index_step:
                        next_target_sec += interval_sec
                    continue
                fingerprint = _frame_fingerprint(frame)
                if (
                    last_fingerprint is not None
                    and _hamming_distance(last_fingerprint, fingerprint) <= duplicate_distance
                ):
                    skipped_duplicate += 1
                    frame_idx += 1
                    if not fallback_to_index_step:
                        next_target_sec += interval_sec
                    continue
                last_fingerprint = fingerprint
            filename = out_dir / f"frame_{start_idx + saved:05d}.jpg"
            ok = cv2.imwrite(str(filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 84])
            if not ok:
                raise RuntimeError(f"Could not write frame: {filename}")
            saved += 1
            if not fallback_to_index_step:
                next_target_sec += interval_sec
        frame_idx += 1

    cap.release()

    duration_hint = (
        f", duration~{last_timestamp_sec:.1f}s" if last_timestamp_sec else ""
    )
    print(
        f"[extract_frames] Label '{safe_label}': saved {saved} frames -> {out_dir} "
        f"(decoded frames={frame_idx}, reported_fps={reported_fps:.2f}, "
        f"reported_frame_count={reported_frame_count:.0f}{duration_hint}, "
        f"index_fallback={fallback_to_index_step}, "
        f"skipped quality={skipped_dark_blurry}, duplicates={skipped_duplicate})"
    )

    warning = None
    if frame_idx <= 5 and saved <= 1:
        warning = (
            f"Z videa se podařilo přečíst jen {frame_idx} snímek(ů). Soubor pravděpodobně "
            "používá kontejner/kodek (často .mov/HEVC), který OpenCV neumí plně dekódovat. "
            "Zkuste video převést do .mp4 (H.264) a nahrát znovu."
        )
        print(f"[extract_frames] WARNING: {warning}")
    elif (
        last_timestamp_sec >= 1.0
        and saved < max(1, int(last_timestamp_sec * fps * 0.5))
    ):
        warning = (
            f"Extrakce uložila jen {saved} snímků z videa o délce ~{last_timestamp_sec:.1f}s "
            f"při {fps:g} fps (čekalo se min. ~{int(last_timestamp_sec * fps)}). "
            "Část snímků mohla být zahozena jako tmavá/rozmazaná/duplicitní, nebo "
            "video metadata (délka/fps) neodpovídají skutečnému obsahu."
        )
        print(f"[extract_frames] WARNING: {warning}")

    return {
        "saved": saved,
        "decoded_frames": frame_idx,
        "reported_fps": round(reported_fps, 2),
        "reported_frame_count": int(reported_frame_count),
        "duration_sec": round(last_timestamp_sec, 2),
        "index_fallback": fallback_to_index_step,
        "skipped_quality": skipped_dark_blurry,
        "skipped_duplicate": skipped_duplicate,
        "warning": warning,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames for ExitNav training")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--label", required=True, help="Room/class label")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second")
    parser.add_argument("--max", type=int, default=2000, help="Maximum frames")
    parser.add_argument("--keep-low-quality", action="store_true", help="Disable blur/dark/duplicate filtering")
    args = parser.parse_args()

    result = extract_frames(
        args.video,
        args.label,
        args.fps,
        max_frames=args.max,
        filter_quality=not args.keep_low_quality,
    )
    print(f"Done: {result['saved']} frames for '{args.label}'")
    if result.get("warning"):
        print(f"Warning: {result['warning']}")
