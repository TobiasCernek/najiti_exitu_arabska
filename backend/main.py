"""
main.py – ExitNav FastAPI backend
"""

import hashlib
import json
import os
import re
import secrets
import shutil
import tempfile
import threading
import traceback
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from pydantic import BaseModel
from typing import Optional

import inference
import pathfinding
from training.storage import TRAINING_DIR

# --------------------------------------------------------------------------- #
ROOT           = Path(__file__).parent
GRAPH_PATH     = ROOT / "data" / "building_graph.json"
POSITIONS_PATH = ROOT / "data" / "node_positions.json"
FRONTEND       = ROOT.parent / "frontend"
AUTH_FILE      = ROOT / "data" / "auth.json"

app = FastAPI(title="ExitNav API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --------------------------------------------------------------------------- #
# Auth helpers
# --------------------------------------------------------------------------- #
DEFAULT_PASSWORD = "admin"
_sessions: set[str] = set()

def _load_password_hash() -> str:
    if AUTH_FILE.exists():
        with open(AUTH_FILE) as f:
            return json.load(f).get("password_hash", _hash(DEFAULT_PASSWORD))
    return _hash(DEFAULT_PASSWORD)

def _save_password_hash(h: str):
    AUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(AUTH_FILE, "w") as f:
        json.dump({"password_hash": h}, f)

def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def _require_admin(x_session_token: Optional[str] = Header(default=None)):
    if not x_session_token or x_session_token not in _sessions:
        raise HTTPException(401, "Neautorizováno")

def _require_admin_token_or_query(x_session_token: Optional[str] = Header(default=None), token: Optional[str] = None):
    """Like _require_admin, but also accepts ?token=... for <img> requests
    which cannot send custom headers."""
    candidate = x_session_token or token
    if not candidate or candidate not in _sessions:
        raise HTTPException(401, "Neautorizováno")

# --------------------------------------------------------------------------- #
# Training state
# --------------------------------------------------------------------------- #
_train_lock = threading.Lock()
_train_status = {
    "running": False,
    "log": [],
    "last_val_acc": None,
    "error": None,
    "epoch": 0,
    "epochs": 0,
    "progress": 0,
}


def _append_train_log(line: str):
    line = line.rstrip()
    if not line:
        return
    with _train_lock:
        _train_status["log"].append(line)
        _train_status["log"] = _train_status["log"][-500:]
        if line.startswith("Epoch "):
            parts = line.split()
            if len(parts) > 1 and "/" in parts[1]:
                current, total = parts[1].split("/", 1)
                if current.isdigit() and total.isdigit():
                    epoch = int(current)
                    epochs = int(total)
                    _train_status["epoch"] = epoch
                    _train_status["epochs"] = epochs
                    progress_epoch = epoch
                    if len(parts) > 3 and parts[2] == "batch" and "/" in parts[3]:
                        batch, batches = parts[3].split("/", 1)
                        if batch.isdigit() and batches.isdigit():
                            progress_epoch = (epoch - 1) + (int(batch) / max(1, int(batches)))
                    _train_status["progress"] = int((progress_epoch / max(1, epochs)) * 100)

# --------------------------------------------------------------------------- #
# Page routes  (must be declared BEFORE StaticFiles mount)
# --------------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
def serve_home():
    return FileResponse(str(FRONTEND / "home.html"))

@app.get("/camera", include_in_schema=False)
def serve_camera():
    return FileResponse(str(FRONTEND / "index.html"))

@app.get("/admin", include_in_schema=False)
def serve_admin():
    return FileResponse(str(FRONTEND / "admin.html"))

@app.get("/editor", include_in_schema=False)
def serve_editor():
    return FileResponse(str(FRONTEND / "editor.html"))

@app.get("/train", include_in_schema=False)
def serve_train():
    return FileResponse(str(FRONTEND / "train.html"))

# Static assets (CSS, JS, images inside frontend folder)
if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")

# ============================================================================ #
# Auth endpoints
# ============================================================================ #
class LoginRequest(BaseModel):
    password: str

@app.post("/api/auth/login")
def login(req: LoginRequest):
    if _hash(req.password) != _load_password_hash():
        raise HTTPException(401, "Špatné heslo")
    token = secrets.token_hex(32)
    _sessions.add(token)
    return {"token": token}

@app.post("/api/auth/logout")
def logout(x_session_token: Optional[str] = Header(default=None)):
    if x_session_token:
        _sessions.discard(x_session_token)
    return {"ok": True}

class ChangePasswordRequest(BaseModel):
    current: str
    new_password: str

@app.post("/api/auth/change-password")
def change_password(req: ChangePasswordRequest, _=Depends(_require_admin)):
    if _hash(req.current) != _load_password_hash():
        raise HTTPException(400, "Aktuální heslo není správné")
    if len(req.new_password) < 4:
        raise HTTPException(400, "Heslo musí mít alespoň 4 znaky")
    _save_password_hash(_hash(req.new_password))
    return {"ok": True}

@app.get("/api/auth/check")
def auth_check(_=Depends(_require_admin)):
    return {"ok": True}

# ============================================================================ #
# Graph
# ============================================================================ #
@app.get("/api/graph")
def get_graph():
    if not GRAPH_PATH.exists():
        raise HTTPException(404, "Graf nenalezen")
    with open(GRAPH_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if POSITIONS_PATH.exists():
        with open(POSITIONS_PATH, "r", encoding="utf-8") as f:
            data["positions"] = json.load(f)
    else:
        data["positions"] = {}
    return data

class GraphPayload(BaseModel):
    nodes: dict
    edges: list
    positions: dict = {}

@app.post("/api/graph")
def save_graph(payload: GraphPayload, _=Depends(_require_admin)):
    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    graph_data = {"nodes": payload.nodes, "edges": payload.edges}
    with open(GRAPH_PATH, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)
    with open(POSITIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload.positions, f, ensure_ascii=False, indent=2)
    return {"ok": True, "nodes": len(payload.nodes), "edges": len(payload.edges)}

# ============================================================================ #
# Model
# ============================================================================ #
@app.get("/api/model/status")
def model_status():
    available = inference.is_model_available()
    classes   = inference.get_classes() if available else None
    return {"available": available, "classes": classes}

@app.get("/api/training/classes")
def list_training_classes():
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    classes = sorted([d.name for d in TRAINING_DIR.iterdir() if d.is_dir()])
    counts  = {c: len(list((TRAINING_DIR / c).glob("*.jpg"))) for c in classes}
    return {"classes": classes, "counts": counts, "storage": str(TRAINING_DIR)}


def _safe_class_dir(label: str) -> Path:
    """Resolve a class folder path safely (no traversal outside TRAINING_DIR)."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip()).strip("._-")
    if not cleaned:
        raise HTTPException(400, "Neplatný název třídy.")
    class_dir = (TRAINING_DIR / cleaned).resolve()
    if class_dir.parent != TRAINING_DIR.resolve():
        raise HTTPException(400, "Neplatný název třídy.")
    return class_dir


@app.delete("/api/training/classes/{label}")
def delete_training_class(label: str, _=Depends(_require_admin)):
    """Delete an entire training class (all extracted frames for that label)."""
    class_dir = _safe_class_dir(label)
    if not class_dir.exists() or not class_dir.is_dir():
        raise HTTPException(404, "Třída nenalezena.")
    shutil.rmtree(class_dir)
    return {"ok": True, "label": label, "deleted": True}


@app.get("/api/training/classes/{label}/frames")
def list_training_frames(label: str, _=Depends(_require_admin)):
    """List extracted frame filenames for a given class."""
    class_dir = _safe_class_dir(label)
    if not class_dir.exists() or not class_dir.is_dir():
        raise HTTPException(404, "Třída nenalezena.")
    frames = sorted(p.name for p in class_dir.glob("*.jpg"))
    return {"label": label, "frames": frames, "count": len(frames)}


@app.get("/api/training/classes/{label}/frames/{filename}")
def get_training_frame(label: str, filename: str, _=Depends(_require_admin_token_or_query)):
    """Serve a single extracted frame image."""
    class_dir = _safe_class_dir(label)
    safe_name = Path(filename).name
    if not re.fullmatch(r"frame_\d{5}\.jpg", safe_name):
        raise HTTPException(400, "Neplatný název souboru.")
    frame_path = class_dir / safe_name
    if not frame_path.exists():
        raise HTTPException(404, "Frame nenalezen.")
    return FileResponse(str(frame_path), media_type="image/jpeg")


@app.delete("/api/training/classes/{label}/frames/{filename}")
def delete_training_frame(label: str, filename: str, _=Depends(_require_admin)):
    """Delete a single extracted frame."""
    class_dir = _safe_class_dir(label)
    safe_name = Path(filename).name
    if not re.fullmatch(r"frame_\d{5}\.jpg", safe_name):
        raise HTTPException(400, "Neplatný název souboru.")
    frame_path = class_dir / safe_name
    if not frame_path.exists():
        raise HTTPException(404, "Frame nenalezen.")
    frame_path.unlink()
    remaining = len(list(class_dir.glob("*.jpg")))
    return {"ok": True, "deleted": filename, "remaining": remaining}

# ============================================================================ #
# Inference
# ============================================================================ #
class InferRequest(BaseModel):
    image: str

@app.post("/api/infer")
def infer(req: InferRequest):
    if not inference.is_model_available():
        raise HTTPException(503, "Model není natrénován.")
    try:
        return inference.predict_from_base64(req.image)
    except Exception as e:
        raise HTTPException(500, str(e))

# ============================================================================ #
# Navigation
# ============================================================================ #
class NavRequest(BaseModel):
    node: str

@app.post("/api/navigate")
def navigate(req: NavRequest):
    try:
        return pathfinding.find_nearest_exit(req.node)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except RuntimeError as e:
        raise HTTPException(500, str(e))

class InferNavRequest(BaseModel):
    image: str

@app.post("/api/infer-and-navigate")
def infer_and_navigate(req: InferNavRequest):
    if not inference.is_model_available():
        raise HTTPException(503, "Model není natrénován.")
    try:
        infer_result = inference.predict_from_base64(req.image)
        label = infer_result["label"]
        try:
            nav_result = pathfinding.find_nearest_exit(label)
        except Exception as nav_err:
            nav_result = {"error": str(nav_err)}
        return {"inference": infer_result, "navigation": nav_result}
    except Exception as e:
        raise HTTPException(500, str(e))

# ============================================================================ #
# Training – extract
# ============================================================================ #
@app.post("/api/train/extract")
async def extract_frames(
    video: UploadFile = File(...),
    label: str = Form(...),
    fps: float = Form(1.0),
    _=Depends(_require_admin),
):
    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(prefix="exitnav_upload_", suffix=suffix, delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    try:
        content = await video.read()
        tmp_path.write_bytes(content)
        from training.extract_frames import extract_frames as do_extract
        result = do_extract(str(tmp_path), label, fps)
        return {
            "ok": True,
            "label": label,
            "frames_saved": result["saved"],
            "storage": str(TRAINING_DIR),
            "decoded_frames": result["decoded_frames"],
            "duration_sec": result["duration_sec"],
            "reported_fps": result["reported_fps"],
            "skipped_quality": result["skipped_quality"],
            "skipped_duplicate": result["skipped_duplicate"],
            "warning": result["warning"],
        }
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

# ============================================================================ #
# Training – start
# ============================================================================ #
class TrainRequest(BaseModel):
    epochs: int = 20
    batch:  int = 16
    lr:     float = 5e-4

@app.post("/api/train/start")
def start_training(req: TrainRequest, _=Depends(_require_admin)):
    global _train_status
    if _train_status["running"]:
        raise HTTPException(409, "Trénink již běží.")
    with _train_lock:
        _train_status = {
            "running": True,
            "log": ["Training job queued."],
            "last_val_acc": None,
            "error": None,
            "epoch": 0,
            "epochs": req.epochs,
            "progress": 0,
        }

    def _run():
        global _train_status
        try:
            from training.train_model import train
            _append_train_log("Training job started.")
            acc = train(epochs=req.epochs, batch_size=req.batch, lr=req.lr, log=_append_train_log)
            with _train_lock:
                _train_status["last_val_acc"] = acc
                _train_status["progress"] = 100
            inference.reset_model_cache()
        except Exception:
            err = traceback.format_exc()
            with _train_lock:
                _train_status["error"] = err
            _append_train_log("Training failed. See error details below.")
        finally:
            with _train_lock:
                _train_status["running"] = False

    threading.Thread(target=_run, daemon=True).start()
    return {"ok": True}

@app.get("/api/train/status")
def train_status(_=Depends(_require_admin)):
    with _train_lock:
        return dict(_train_status)

# ============================================================================ #
# WebSocket – live stream
# ============================================================================ #
@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    model_ready = inference.is_model_available()
    try:
        while True:
            data = await ws.receive_json()
            frame_b64 = data.get("frame", "")
            ts = data.get("ts", 0)
            if not model_ready:
                await ws.send_json({"ts": ts, "error": "model_not_ready"})
                continue
            try:
                infer_result = inference.predict_from_base64(frame_b64)
                label = infer_result["label"]
                try:
                    nav = pathfinding.find_nearest_exit(label)
                except Exception as e:
                    nav = {"error": str(e)}
                await ws.send_json({"ts": ts, "inference": infer_result, "navigation": nav})
            except Exception as e:
                await ws.send_json({"ts": ts, "error": str(e)})
    except WebSocketDisconnect:
        pass
