# draw-service

The card detection and recognition service behind
[Duel Tavern](https://github.com/dueltavern), built on the models from
[draw2](https://github.com/HichTala/draw2) by HichTala. Given a photo of a
Yu-Gi-Oh! board, it returns every card it finds, with its position and,
when it can tell, which card it is.

## Setup

```bash
uv sync
uv run uvicorn service:app --host 0.0.0.0 --port 8008
```

Without uv: `pip install fastapi "uvicorn[standard]" pydantic ultralytics transformers accelerate huggingface_hub opencv-python-headless pillow torch numpy`.

The first start downloads the YOLO detector and the card classifier from
Hugging Face (several hundred MB) and caches them. They run on an NVIDIA
GPU (CUDA) if there is one, otherwise on Apple silicon's GPU (MPS),
otherwise on the CPU; set `DRAW_DEVICE` to `cuda`, `mps` or `cpu` to force
one. On an M3 Ultra a 9-card field takes ~90 ms on MPS against ~650 ms on
the CPU.

## API

`POST /detect` with the image (JPEG or PNG) as the raw request body:

```bash
curl --data-binary @frame.jpg -H "Content-Type: image/jpeg" "http://localhost:8008/detect?allowedCardIds=46986414"
```

`allowedCardIds` is optional, can be repeated, and limits recognition to
those cards.

```json
{
  "detections": [
    { "points": [[x, y], [x, y], [x, y], [x, y]], "cardId": "46986414", "cardName": "Dark Magician", "confidence": 0.94 },
    { "points": [[x, y], [x, y], [x, y], [x, y]], "cardId": null, "cardName": "Face-down card", "confidence": 0.0 }
  ]
}
```

`points` are the card's four corners in image pixels. A card that was found
but couldn't be identified (usually a face-down card) has `cardId: null`.

`GET /health` returns `{ "ready": true }` once the models are loaded.

### Debugging

- `LOG_DETECTIONS` at the top of `detector.py` logs a summary of every
  frame: boxes found, cards identified, and why the rest weren't.
- Setting `DRAW_SAVE_FRAMES=<folder>` saves every scanned frame with its
  detections, to inspect later.

## How it works

For each frame, `detector.py`:

1. Finds card-shaped regions with draw2's YOLO oriented-box detector.
2. Warps each region to a square-on 224×224 crop.
3. Works out which way up the card is from where its text box sits (draw2's
   rotation heuristic).
4. Identifies every card with draw2's image classifier, all in one batch. A
   card whose orientation couldn't be settled is tried in all four
   rotations, and the most confident reading is kept.
5. Keeps cards it can't identify as face-down entries, and finds plain card
   backs and light-colored sleeves, which the detector doesn't see, by color.

### Differences from draw2's `Draw` class

draw2's `Draw` is built for one continuous video source: it reloads both
models on every construction, and its `process()` stops at the first card it
can't orient (`break` instead of `continue`), dropping every other card in
the frame. It also returns only labels. This service loads the models once,
handles each frame independently, never drops a card, and returns positions
and confidence for each one.

## Licensing

Licensed under the **GNU Affero General Public License v3.0** (see
`LICENSE`), like draw2. `detector.py` reimplements and modifies draw2's
`Draw.process()` and `get_rotation`. This repository is the corresponding
source for the card recognition running in Duel Tavern. Credit to HichTala
for the pipeline design and for training and hosting the models.
