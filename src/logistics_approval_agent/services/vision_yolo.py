from __future__ import annotations

import io
from pathlib import Path
from typing import Callable

from logistics_approval_agent.config import Settings, ensure_yolo_weights, get_settings

try:  # pragma: no cover - import depends on heavy optional runtime
    from ultralytics import YOLO
except Exception:  # pragma: no cover - import depends on heavy optional runtime
    YOLO = None


_YOLO_MODEL_CACHE = None
_YOLO_MODEL_PATH: Path | None = None
_YOLO_INITIALIZED = False


def load_yolo_model(
    settings: Settings | None = None,
    success_callback: Callable[[str], None] | None = None,
    warning_callback: Callable[[str], None] | None = None,
):
    global _YOLO_MODEL_CACHE, _YOLO_MODEL_PATH, _YOLO_INITIALIZED

    if YOLO is None:
        if warning_callback:
            warning_callback(
                "Object detection currently does not work and thus container damage "
                "detection features will be disabled."
            )
        return None

    app_settings = settings or get_settings()
    weights_path = ensure_yolo_weights(
        weights_path=app_settings.yolo_weights_path,
        download_url=app_settings.yolo_weights_url,
    )

    if _YOLO_MODEL_CACHE is None or _YOLO_MODEL_PATH != weights_path:
        _YOLO_MODEL_CACHE = YOLO(str(weights_path))
        _YOLO_MODEL_PATH = weights_path

    if not _YOLO_INITIALIZED:
        if success_callback:
            success_callback("Object and container damage detection initialized successfully.")
        _YOLO_INITIALIZED = True

    return _YOLO_MODEL_CACHE


def analyze_image_with_yolo(image_bytes, model) -> str:
    if model is None:
        return "YOLO model not loaded. Analysis skipped."
    if not image_bytes:
        return "No image data provided for analysis."

    try:
        from PIL import Image as PILImage

        pil_image = PILImage.open(io.BytesIO(image_bytes))
        results = model(pil_image)

        detections_summary = []
        if results and results[0].boxes:
            names = results[0].names
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = names[class_id]
                    confidence = float(box.conf[0])
                    detections_summary.append(
                        f"{class_name} (confidence: {confidence:.2f})"
                    )

        if not detections_summary:
            return "No objects detected by YOLOv8."
        return "Detected: " + ", ".join(detections_summary)
    except Exception as exc:
        return f"Error during YOLOv8 analysis: {exc}"
