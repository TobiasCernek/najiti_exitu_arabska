"""
extract_frames.py – Extrakce framů z videa pro trénink.

Použití:
    python extract_frames.py --video path/to/video.mp4 --label ucebna_101 --fps 1
    python extract_frames.py --video path/to/video.mp4 --label chodba_1np --fps 2

Framy jsou ukládány do:
    backend/data/training/<label>/frame_XXXX.jpg
"""

import argparse
import sys
from pathlib import Path

try:
    import cv2
except ImportError:
    print("Chybí balíček opencv-python. Nainstalujte: pip install opencv-python")
    sys.exit(1)


def extract_frames(
    video_path: str,
    label: str,
    fps: float = 1.0,
    output_root: Path = Path(__file__).parent.parent / "data" / "training",
    max_frames: int = 2000,
) -> int:
    """
    Extrahuje framy z videa a uloží je do složky dle labelu.
    Vrátí počet uložených framů.
    """
    out_dir = output_root / label
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Nelze otevřít video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(video_fps / fps)))  # každý N-tý frame

    # Zjisti aktuální nejvyšší index v cílové složce
    existing = sorted(out_dir.glob("frame_*.jpg"))
    start_idx = len(existing)

    saved = 0
    frame_idx = 0

    while saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            filename = out_dir / f"frame_{start_idx + saved:05d}.jpg"
            cv2.imwrite(str(filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            saved += 1
        frame_idx += 1

    cap.release()
    print(f"[extract_frames] Label '{label}': uloženo {saved} framů → {out_dir}")
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extrakce framů z videa pro ExitNav trénink")
    parser.add_argument("--video",  required=True,  help="Cesta k video souboru")
    parser.add_argument("--label",  required=True,  help="Název místnosti / label třídy")
    parser.add_argument("--fps",    type=float, default=1.0, help="Počet framů za sekundu (default 1)")
    parser.add_argument("--max",    type=int, default=2000,  help="Max počet framů (default 2000)")
    args = parser.parse_args()

    n = extract_frames(args.video, args.label, args.fps, max_frames=args.max)
    print(f"Hotovo: {n} framů pro třídu '{args.label}'")
