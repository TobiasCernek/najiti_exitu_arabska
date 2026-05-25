"""
main.py – ExitNav FastAPI backend

Endpointy:
    GET  /                          → health check
    GET  /api/graph                 → načte graf budovy
    POST /api/graph                 → uloží graf budovy
    GET  /api/model/status          → stav modelu (natrénován / ne)
    POST /api/infer                 → rozpozná místnost z base64 snímku
    POST /api/navigate              → vrátí cestu k exitu z dané místnosti
    POST /api/infer-and-navigate    → rozpozná místnost + vrátí cestu
    POST /api/train/extract         → spustí extrakci framů z videa (multipart)
    POST /api/train/start           → spustí trénink v pozadí
    GET  /api/train/status          → stav tréninku
    WS   /ws/stream                 → WebSocket: přijímá framy, vrací inferenci + navigaci
"""

import asyncio
import json
import subprocess
import sys
import traceback
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import inference
import pathfinding

# --------------------------------------------------------------------------- #
ROOT      = Path(__file__).parent
GRAPH_PATH = ROOT / "data" / "building_graph.json"
TRAINING_DIR = ROOT / "data" / "training"
FRONTEND  = ROOT.parent / "frontend"

app = FastAPI(title="ExitNav API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Statické soubory frontendu
if FRONTEND.exists():
    app.mount("/app", StaticFiles(directory=str(FRONTEND), html=True), name="frontend")

# --------------------------------------------------------------------------- #
# Stav tréninku (jednoduchý in-memory state)
# --------------------------------------------------------------------------- #
_train_status = {"running": False, "log": [], "last_val_acc": None, "error": None}


# ============================================================================ #
# Health
# ============================================================================ #
@app.get("/")
def root():
    return {"service": "ExitNav API", "version": "1.0.0"}


# ============================================================================ #
# Graf budovy
# ============================================================================ #
@app.get("/api/graph")
def get_graph():
    if not GRAPH_PATH.exists():
        raise HTTPException(404, "Graf budovy nenalezen")
    with open(GRAPH_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


class GraphPayload(BaseModel):
    nodes: dict
    edges: list


@app.post("/api/graph")
def save_graph(payload: GraphPayload):
    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GRAPH_PATH, "w", encoding="utf-8") as f:
        json.dump(payload.dict(), f, ensure_ascii=False, indent=2)
    return {"ok": True, "nodes": len(payload.nodes), "edges": len(payload.edges)}


# ============================================================================ #
# Model
# ============================================================================ #
@app.get("/api/model/status")
def model_status():
    available = inference.is_model_available()
    classes   = inference.get_classes() if available else None
    return {
        "available":  available,
        "classes":    classes,
        "model_path": str(inference.MODEL_FILE),
    }


# ============================================================================ #
# Inference
# ============================================================================ #
class InferRequest(BaseModel):
    image: str  # base64 JPEG


@app.post("/api/infer")
def infer(req: InferRequest):
    if not inference.is_model_available():
        raise HTTPException(503, "Model není natrénován. Spusťte trénink přes /api/train/start.")
    try:
        result = inference.predict_from_base64(req.image)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


# ============================================================================ #
# Navigace
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
    image: str  # base64 JPEG


@app.post("/api/infer-and-navigate")
def infer_and_navigate(req: InferNavRequest):
    if not inference.is_model_available():
        raise HTTPException(503, "Model není natrénován.")
    try:
        infer_result = inference.predict_from_base64(req.image)
        label = infer_result["label"]
        nav_result = pathfinding.find_nearest_exit(label)
        return {
            "inference":  infer_result,
            "navigation": nav_result,
        }
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


# ============================================================================ #
# Trénink – extrakce framů
# ============================================================================ #
@app.post("/api/train/extract")
async def extract_frames(
    video: UploadFile = File(...),
    label: str = Form(...),
    fps: float = Form(1.0),
):
    """Přijme video soubor, uloží ho dočasně a spustí extrakci framů."""
    tmp_path = ROOT / "data" / f"_tmp_{video.filename}"
    try:
        content = await video.read()
        tmp_path.write_bytes(content)

        from training.extract_frames import extract_frames as do_extract
        n = do_extract(str(tmp_path), label, fps)
        return {"ok": True, "label": label, "frames_saved": n}
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


# ============================================================================ #
# Trénink – spuštění
# ============================================================================ #
class TrainRequest(BaseModel):
    epochs: int = 20
    batch:  int = 16
    lr:     float = 5e-4


@app.post("/api/train/start")
def start_training(req: TrainRequest):
    global _train_status
    if _train_status["running"]:
        raise HTTPException(409, "Trénink již běží.")

    _train_status = {"running": True, "log": [], "last_val_acc": None, "error": None}

    def _run():
        global _train_status
        try:
            from training.train_model import train
            # Přesměrování výstupu do logu
            import io as _io, contextlib
            buf = _io.StringIO()
            with contextlib.redirect_stdout(buf):
                acc = train(epochs=req.epochs, batch_size=req.batch, lr=req.lr)
            _train_status["log"] = buf.getvalue().splitlines()
            _train_status["last_val_acc"] = acc
        except Exception as e:
            _train_status["error"] = traceback.format_exc()
        finally:
            _train_status["running"] = False

    import threading
    threading.Thread(target=_run, daemon=True).start()
    return {"ok": True, "message": "Trénink spuštěn na pozadí."}


@app.get("/api/train/status")
def train_status():
    return _train_status


# ============================================================================ #
# WebSocket – live stream framů
# ============================================================================ #
@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    model_ready = inference.is_model_available()
    try:
        while True:
            data = await ws.receive_json()
            frame_b64 = data.get("frame", "")
            ts         = data.get("ts", 0)

            if not model_ready:
                await ws.send_json({
                    "ts":     ts,
                    "error":  "model_not_ready",
                    "msg":    "Model není natrénován.",
                })
                continue

            try:
                infer_result = inference.predict_from_base64(frame_b64)
                label = infer_result["label"]

                try:
                    nav = pathfinding.find_nearest_exit(label)
                except Exception as nav_err:
                    nav = {"error": str(nav_err)}

                await ws.send_json({
                    "ts":         ts,
                    "inference":  infer_result,
                    "navigation": nav,
                })
            except Exception as e:
                await ws.send_json({"ts": ts, "error": str(e)})

    except WebSocketDisconnect:
        pass
