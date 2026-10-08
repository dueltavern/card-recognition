# uvicorn service:app --host 0.0.0.0 --port 8008

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, Query, Request
from starlette.concurrency import run_in_threadpool

from detector import CardDetector

detector: CardDetector | None = None

# Debugging: saves every frame (.jpg) with its detections (.json) to this folder.
SAVE_FRAMES_DIR = os.environ.get("DRAW_SAVE_FRAMES")


def _save_frame(image_bgr, detections):
    folder = Path(SAVE_FRAMES_DIR)
    folder.mkdir(parents=True, exist_ok=True)
    stem = str(folder / (time.strftime("%Y%m%d-%H%M%S-") + f"{int(time.time() * 1000) % 1000:03d}"))
    cv2.imwrite(f"{stem}.jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    with open(f"{stem}.json", "w") as f:
        json.dump(detections, f, indent=1)
    print(f"[service] saved {stem}.jpg ({len(detections)} detection(s))")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector
    detector = CardDetector()
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/detect")
async def detect(request: Request, allowedCardIds: list[str] | None = Query(None)):
    raw = await request.body()
    image_bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return {"detections": []}

    allowed = set(allowedCardIds) if allowedCardIds else None
    detections = await run_in_threadpool(detector.detect, image_bgr, allowed_card_ids=allowed)
    if SAVE_FRAMES_DIR:
        _save_frame(image_bgr, detections)
    return {"detections": detections}


@app.get("/health")
def health():
    return {"ready": detector is not None}
