"""
Card detection and recognition with the YOLO detector and classifier from
HichTala/draw2 (https://github.com/HichTala/draw2, AGPL-3.0).

Modified from draw2's Draw.process() and utils.get_rotation by the Duel
Tavern developers, first on 2026-09-19.

draw2's Draw class is bound to a single video source and reloads both models
every time it's constructed, so this loads them once and exposes detect() for
one image at a time. Unlike Draw.process(), a card whose orientation can't be
worked out is classified in all four rotations instead of ending the loop
(draw2 breaks there and drops the rest of the frame).
"""

import json
import math
import os

import cv2
import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor, AutoModelForImageClassification
from ultralytics import YOLO

MODEL_REPO = "HichTala/draw2"
DEFAULT_CONFIDENCE_THRESHOLD = 5  # percent, same default as draw2
LOG_DETECTIONS = True
CLASSIFY_BATCH_SIZE = 32
WARP_CORNERS = np.float32([[224, 224], [224, 0], [0, 0], [0, 224]])
ALL_ROTATIONS = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]

# Unsleeved backs: max HSV value for the dark swirl. The swirl is inset from
# the card edge, so its box gets scaled up to cover the whole card.
CARD_BACK_MAX_BRIGHTNESS = 90
CARD_BACK_EDGE_SCALE = 1.1
# Light sleeves (white/cream/grey). A wooden table sits around saturation 120.
LIGHT_SLEEVE_MIN_BRIGHTNESS = 200
LIGHT_SLEEVE_MAX_SATURATION = 70


def _extract_contours(roi, d, sigma_color, sigma_space, thresh):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d, sigma_color, sigma_space)
    equalized = cv2.equalizeHist(gray)
    _, binary = cv2.threshold(equalized, thresh, 255, cv2.THRESH_BINARY)

    kernel = np.ones((7, 7), np.uint8)
    edged = cv2.erode(binary, kernel, iterations=3)
    edged = cv2.dilate(edged, kernel, iterations=3)

    contours = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0] if len(contours) == 2 else contours[1]


def _text_box(contour):
    return np.intp(cv2.boxPoints(cv2.minAreaRect(contour)))


def _face_down(pts, confidence=0.0):
    return {
        "points": pts.tolist(),
        "cardId": None,
        "cardName": "Face-down card",
        "confidence": confidence,
    }


def _long_side(pts):
    return max(float(np.linalg.norm(np.subtract(pts[(i + 1) % 4], pts[i]))) for i in range(4))


def _find_card_backs(image_bgr, face_up_points):
    """
    The YOLO model only knows card faces and misses backs and sleeves
    entirely, so those are found by color instead: very dark for unsleeved
    backs, bright and nearly colorless for light sleeves.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    masks = [
        (cv2.inRange(hsv, (0, 0, 0), (180, 255, CARD_BACK_MAX_BRIGHTNESS)), CARD_BACK_EDGE_SCALE),
        (cv2.inRange(hsv, (0, 0, LIGHT_SLEEVE_MIN_BRIGHTNESS), (180, LIGHT_SLEEVE_MAX_SATURATION, 255)), 1.0),
    ]
    found = []
    for mask, edge_scale in masks:
        found += _card_shaped_blobs(mask, edge_scale, image_bgr.shape[0], face_up_points, found)
    return found


def _card_shaped_blobs(mask, edge_scale, frame_h, face_up_points, already_found):
    typical = float(np.median([_long_side(p) for p in face_up_points])) if face_up_points else None
    taken = list(face_up_points) + list(already_found)
    k = max(3, int((typical or frame_h * 0.24) * 0.06)) | 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blobs = []
    for contour in contours:
        (cx, cy), (w, h), angle = cv2.minAreaRect(contour)
        if min(w, h) < 10:
            continue
        long_side, short_side = max(w, h), min(w, h)
        card_long = long_side * edge_scale
        # Without face-up cards to compare against, size it against the frame.
        size_ok = (0.55 < card_long / typical < 1.5) if typical else (0.1 < card_long / frame_h < 0.5)
        area = cv2.contourArea(contour)
        # A tilted camera turns cards into trapezoids, so fill is kept loose
        # and solidity is what rejects shadows and hands.
        fill = area / (w * h)
        solidity = area / max(cv2.contourArea(cv2.convexHull(contour)), 1)
        if not (1.0 <= long_side / short_side < 2.2 and fill > 0.75 and solidity > 0.9 and size_ok):
            continue
        if any(cv2.pointPolygonTest(np.float32(p), (cx, cy), False) >= 0 for p in taken):
            continue
        blobs.append(np.float32(cv2.boxPoints(((cx, cy), (w * edge_scale, h * edge_scale), angle))))
    return blobs


def _get_rotation(xywhr, text_box):
    # Port of draw2's utils.get_rotation. The angle has to stay (r % pi) / 2
    # like draw2 computes it, the branches were tuned against that.
    w, h, r = xywhr[2], xywhr[3], xywhr[4]
    angle = (r % math.pi) / 2

    if min(text_box[:, 0]) < 112:
        if max(text_box[:, 0]) < 112:
            if min(text_box[:, 1]) < 112 < max(text_box[:, 1]):
                if (h > w and angle > math.pi / 4) or (h < w and angle < math.pi / 4):
                    return cv2.ROTATE_90_COUNTERCLOCKWISE
                return None
            return None
        if min(text_box[:, 1]) < max(text_box[:, 1]) < 112:
            if (h > w and angle < math.pi / 4) or (h < w and angle > math.pi / 4):
                return cv2.ROTATE_180
            return None
        if 112 < min(text_box[:, 1]) < max(text_box[:, 1]):
            if (h > w and angle < math.pi / 4) or (h < w and angle > math.pi / 4):
                return 0
            return None
        return None
    if min(text_box[:, 1]) < 112 < max(text_box[:, 1]):
        if (h > w and angle > math.pi / 4) or (h < w and angle < math.pi / 4):
            return cv2.ROTATE_90_CLOCKWISE
        return None
    return None


def _pick_device():
    forced = os.environ.get("DRAW_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class CardDetector:
    def __init__(self, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
        self.device = _pick_device()
        self.confidence_threshold = confidence_threshold

        config_path = hf_hub_download(repo_id=MODEL_REPO, filename="draw_config.json")
        yolo_path = hf_hub_download(repo_id=MODEL_REPO, filename="ygo_yolo.pt")
        cardnames_path = hf_hub_download(repo_id=MODEL_REPO, filename="cardnames.json")

        with open(config_path) as f:
            self.config = json.load(f)
        with open(cardnames_path, encoding="utf-8") as f:
            self.cardnames = json.load(f)

        self.yolo = YOLO(yolo_path)

        self.image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", use_fast=True)
        self.classifier = AutoModelForImageClassification.from_pretrained(MODEL_REPO).to(self.device).eval()
        self.id2label = self.classifier.config.id2label

    def _classify(self, rois, allowed_card_ids):
        # Same softmax + top 15 as the transformers pipeline, but done on the
        # whole batch at once. The pipeline's per-image post-processing was
        # about 4x slower than the model itself.
        images = [Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)) for roi in rois]
        best = []
        for start in range(0, len(images), CLASSIFY_BATCH_SIZE):
            batch = images[start : start + CLASSIFY_BATCH_SIZE]
            inputs = self.image_processor(images=batch, return_tensors="pt").to(self.device)
            with torch.no_grad():
                top = self.classifier(**inputs).logits.softmax(-1).topk(15)
            for scores, label_ids in zip(top.values.cpu().tolist(), top.indices.cpu().tolist()):
                candidates = [{"label": self.id2label[i], "score": s} for s, i in zip(scores, label_ids)]
                if allowed_card_ids:
                    candidates = [c for c in candidates if c["label"].split("-")[-1] in allowed_card_ids]
                best.append(candidates[0] if candidates else None)
        return best

    def detect(self, image_bgr, allowed_card_ids=None):
        """
        Returns one { points, cardId, cardName, confidence } per card found,
        with points as the 4 corners in image pixels. Cards that can't be
        identified (usually face-down) are still returned with cardId None.
        """
        results = self.yolo.predict(
            source=image_bgr,
            show_labels=False,
            save=False,
            device=self.device,
            verbose=False,
        )
        if not results or results[0].obb is None:
            card_backs = [_face_down(pts) for pts in _find_card_backs(image_bgr, [])]
            if LOG_DETECTIONS:
                print(f"[detector] no YOLO boxes, card_backs={len(card_backs)}")
            return card_backs
        result = results[0]
        raw_box_count = len(result.obb.xyxyxyxyn)
        frame_h, frame_w = result.orig_img.shape[:2]

        detections = []
        pending = []  # (index in detections, pts, crops)
        no_contour = 0
        for i, box in enumerate(result.obb.xyxyxyxyn):
            pts = np.float32([[p[0] * frame_w, p[1] * frame_h] for p in box.cpu()])
            transform = cv2.getPerspectiveTransform(pts, WARP_CORNERS)
            roi = cv2.warpPerspective(image_bgr, transform, (224, 224), flags=cv2.INTER_LINEAR)

            contours = _extract_contours(
                roi,
                d=self.config["bilateral_filter_d"],
                sigma_color=self.config["bilateral_filter_sigma_color"],
                sigma_space=self.config["bilateral_filter_sigma_space"],
                thresh=self.config["txt_box_contour_threshold"],
            )
            if not contours:
                no_contour += 1
                detections.append(_face_down(pts))
                continue

            rotation = _get_rotation(result.obb.xywhr[i], _text_box(max(contours, key=cv2.contourArea)))
            if rotation is None:
                crops = [roi] + [cv2.rotate(roi, r) for r in ALL_ROTATIONS]
            else:
                crops = [roi if rotation == 0 else cv2.rotate(roi, rotation)]
            pending.append((len(detections), pts, crops))
            detections.append(None)

        # One classifier pass for every crop; ambiguous cards keep whichever rotation scores best.
        classified = iter(self._classify([crop for _, _, crops in pending for crop in crops], allowed_card_ids))
        low_confidence = 0
        for index, pts, crops in pending:
            candidates = [c for c in (next(classified) for _ in crops) if c is not None]
            best = max(candidates, key=lambda c: c["score"]) if candidates else None
            if best is None or best["score"] < self.confidence_threshold / 100:
                low_confidence += 1
                detections[index] = _face_down(pts, best["score"] if best else 0.0)
                continue
            fallback_name, _, card_id = best["label"].rpartition("-")
            detections[index] = {
                "points": pts.tolist(),
                "cardId": card_id,
                "cardName": self.cardnames.get(card_id, {}).get("EN") or fallback_name,
                "confidence": best["score"],
            }

        card_backs = _find_card_backs(image_bgr, [d["points"] for d in detections])
        detections.extend(_face_down(pts) for pts in card_backs)

        if LOG_DETECTIONS:
            names = ", ".join(d["cardName"] for d in detections) or "none"
            print(
                f"[detector] {raw_box_count} box(es) -> {len(detections)} detection(s): {names} "
                f"(no_contour={no_contour}, low_confidence={low_confidence}, card_backs={len(card_backs)})"
            )

        return detections
