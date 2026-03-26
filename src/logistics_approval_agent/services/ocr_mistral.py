from __future__ import annotations

import base64
from typing import Callable

from mistralai import Mistral


def create_mistral_client(api_key: str) -> Mistral | None:
    if not api_key or api_key == "YOUR_MISTRAL_API_KEY":
        return None
    return Mistral(api_key=api_key)


def process_sop_with_mistral_ocr(
    uploaded_file,
    api_key: str,
    error_callback: Callable[[str], None] | None = None,
):
    client = create_mistral_client(api_key)
    if not client:
        if error_callback:
            error_callback(
                "Mistral API key not configured. Please set MISTRAL_API_KEY environment variable."
            )
        return None

    try:
        base64_pdf = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
        return client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{base64_pdf}",
            },
            include_image_base64=True,
        )
    except Exception as exc:  # pragma: no cover - network/API dependent
        if error_callback:
            error_callback(f"Error processing SOP with Mistral OCR: {exc}")
        return None
