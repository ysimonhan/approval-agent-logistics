from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


DEFAULT_COHERE_MODEL = "c4ai-aya-vision-32b"
DEFAULT_YOLO_WEIGHTS_FILENAME = "yolov8m.pt"
DEFAULT_YOLO_WEIGHTS_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt"
)


Downloader = Callable[[str, Path], None]


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_dir: Path
    models_dir: Path
    sample_sop_path: Path
    yolo_weights_path: Path
    yolo_weights_url: str
    cohere_api_key: str
    cohere_model: str
    mistral_api_key: str
    custom_model_api_key: str
    custom_model_endpoint: str


def get_project_root() -> Path:
    configured_root = os.getenv("PROJECT_ROOT")
    if configured_root:
        return Path(configured_root).resolve()
    return Path(__file__).resolve().parents[2]


def get_settings() -> Settings:
    project_root = get_project_root()
    data_dir = Path(os.getenv("DATA_DIR", project_root / "data"))
    models_dir = Path(os.getenv("MODEL_DIR", project_root / "models"))
    default_sample_sop_path = data_dir / "samples" / "temp_sop.pdf"
    legacy_sample_sop_path = project_root / "01-App" / "temp_sop.pdf"
    sample_sop_fallback = default_sample_sop_path
    if not default_sample_sop_path.exists() and legacy_sample_sop_path.exists():
        sample_sop_fallback = legacy_sample_sop_path
    sample_sop_path = Path(os.getenv("SAMPLE_SOP_PATH", sample_sop_fallback))
    yolo_weights_path = Path(
        os.getenv("YOLO_WEIGHTS_PATH", models_dir / DEFAULT_YOLO_WEIGHTS_FILENAME)
    )

    return Settings(
        project_root=project_root,
        data_dir=data_dir,
        models_dir=models_dir,
        sample_sop_path=sample_sop_path,
        yolo_weights_path=yolo_weights_path,
        yolo_weights_url=os.getenv("YOLO_WEIGHTS_URL", DEFAULT_YOLO_WEIGHTS_URL),
        cohere_api_key=os.getenv("COHERE_API_KEY", "YOUR_COHERE_API_KEY_HERE"),
        cohere_model=os.getenv("COHERE_MODEL", DEFAULT_COHERE_MODEL),
        mistral_api_key=os.getenv("MISTRAL_API_KEY", "YOUR_MISTRAL_API_KEY"),
        custom_model_api_key=os.getenv(
            "CUSTOM_MODEL_API_KEY", "YOUR_CUSTOM_MODEL_API_KEY"
        ),
        custom_model_endpoint=os.getenv(
            "CUSTOM_MODEL_ENDPOINT", "YOUR_CUSTOM_MODEL_ENDPOINT"
        ),
    )


def _default_downloader(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url) as response:
        destination.write_bytes(response.read())


def ensure_yolo_weights(
    weights_path: Path | str,
    download_url: str,
    downloader: Downloader | None = None,
) -> Path:
    resolved_path = Path(weights_path)
    if resolved_path.exists():
        return resolved_path

    if not download_url:
        raise ValueError("A YOLO weights download URL is required when the file is missing.")

    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    download = downloader or _default_downloader
    download(download_url, resolved_path)

    if not resolved_path.exists() or resolved_path.stat().st_size == 0:
        raise RuntimeError(f"YOLO weights download did not create a valid file at {resolved_path}.")

    return resolved_path
