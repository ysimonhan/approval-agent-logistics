from __future__ import annotations

import json
import time
from typing import Callable

from logistics_approval_agent.config import DEFAULT_COHERE_MODEL, Settings, get_settings
from logistics_approval_agent.services.llm_cohere import create_cohere_client
from logistics_approval_agent.services.vision_yolo import load_yolo_model


Status = str


def find_missing_required_images(
    ticket_data: dict,
    repair_codes_needing_images: dict[str, str],
) -> list[str]:
    missing_codes: list[str] = []
    for repair in ticket_data.get("repairs", []):
        code = repair.get("code", "N/A")
        requires_image = code in repair_codes_needing_images
        image_provided_for_code = any(
            media_item.get("repair_code_association") == code
            for media_item in ticket_data.get("media", [])
        )
        if requires_image and not image_provided_for_code:
            missing_codes.append(code)
    return missing_codes


def determine_ticket_status(
    ai_result: dict,
    approval_confidence_threshold: float = 0.75,
) -> Status:
    if ai_result.get("missing_data_request"):
        return "Additional Data Requested"
    if ai_result.get("decision") == "APPROVE" and ai_result.get(
        "confidence_score", 0.0
    ) >= approval_confidence_threshold:
        return "AI Approved"
    if ai_result.get("decision") == "DISAPPROVE":
        return "AI Disapproved"
    return "Manual Review Required"


def call_custom_model(
    ticket_data: dict,
    model_config: dict,
    info_callback: Callable[[str], None] | None = None,
) -> dict:
    if info_callback:
        info_callback(
            f"Trying to call custom model with endpoint: {model_config.get('endpoint')}"
        )
    time.sleep(1)
    decision_options = ["APPROVE", "DISAPPROVE", "MANUAL_REVIEW"]
    decision = decision_options[hash(ticket_data["ticket_id"]) % 3]
    confidence = 0.65 + (hash(ticket_data["ticket_id"]) % 30) / 100.0
    reasoning = (
        f"Custom model mock decision: Processed ticket {ticket_data['ticket_id']}. "
        f"Decision: {decision}."
    )
    return {
        "decision": decision,
        "confidence_score": confidence,
        "reasoning": reasoning,
        "missing_data_request": None,
    }


def build_yolo_analysis_prompt_section(ticket_data: dict) -> str:
    yolo_summaries_for_prompt: list[str] = []
    for media_item in ticket_data.get("media", []):
        if media_item.get("type") == "image" and media_item.get("yolo_summary"):
            assoc_info = ""
            if media_item.get("repair_code_association"):
                assoc_info = (
                    f" (associated with repair code "
                    f"{media_item.get('repair_code_association')})"
                )
            yolo_summaries_for_prompt.append(
                f"- Image '{media_item.get('filename', 'N/A')}'{assoc_info}: "
                f"{media_item['yolo_summary']}"
            )
    if not yolo_summaries_for_prompt:
        return ""
    return "\n\nImage Analysis Summary (from YOLOv8 object detection):\n" + "\n".join(
        yolo_summaries_for_prompt
    )


def call_cohere_aya_yolo_model(
    ticket_data: dict,
    age_cost_thresholds: dict[str, int],
    repair_codes_needing_images: dict[str, str],
    settings: Settings | None = None,
    error_callback: Callable[[str], None] | None = None,
    success_callback: Callable[[str], None] | None = None,
    warning_callback: Callable[[str], None] | None = None,
) -> dict:
    app_settings = settings or get_settings()
    if (
        not app_settings.cohere_api_key
        or app_settings.cohere_api_key == "YOUR_COHERE_API_KEY_HERE"
    ):
        if error_callback:
            error_callback(
                "Cohere API Key not configured. Please set the COHERE_API_KEY "
                "environment variable."
            )
        return {
            "decision": "MANUAL_REVIEW",
            "confidence_score": 0.0,
            "reasoning": "Configuration error: Cohere API Key missing.",
            "missing_data_request": "Cohere API Key configuration.",
        }

    load_yolo_model(
        settings=app_settings,
        success_callback=success_callback,
        warning_callback=warning_callback,
    )
    cohere_client = create_cohere_client(app_settings.cohere_api_key)
    yolo_analysis_prompt_section = build_yolo_analysis_prompt_section(ticket_data)

    repair_details_prompt = []
    missing_images_for_codes = find_missing_required_images(
        ticket_data,
        repair_codes_needing_images,
    )

    for repair in ticket_data.get("repairs", []):
        code = repair.get("code", "N/A")
        description = repair.get("description", "No description")
        requires_image = code in repair_codes_needing_images
        image_provided_for_code = any(
            media_item.get("repair_code_association") == code
            for media_item in ticket_data.get("media", [])
        )
        repair_details_prompt.append(
            "  - Repair Code: "
            f"{code}, Description: {description} "
            f"(Requires Image: {requires_image}, "
            f"Image Provided for this code: {image_provided_for_code})"
        )

    if missing_images_for_codes:
        missing_data_details = (
            "Mandatory images missing for repair codes: "
            + ", ".join(missing_images_for_codes)
            + "."
        )
        return {
            "decision": "MANUAL_REVIEW",
            "confidence_score": 1.0,
            "reasoning": (
                "Cannot proceed with AI approval/disapproval. "
                f"{missing_data_details} Please upload them."
            ),
            "missing_data_request": missing_data_details,
        }

    media_summary = []
    for media_item in ticket_data.get("media", []):
        assoc = ""
        if media_item.get("repair_code_association"):
            assoc = f" (for code {media_item.get('repair_code_association')})"
        media_summary.append(f"{media_item.get('filename', 'Unknown file')}{assoc}")

    prompt = f"""You are an AI Repair Ticket Approval Agent for a container depot.
    Review the container repair ticket and decide: 'APPROVE', 'DISAPPROVE', or 'MANUAL_REVIEW'.
    Provide a confidence score (0.0-1.0) and reasoning.
    If data is missing (e.g., required images and/or videos), an AI decision cannot be made; instead, the decision should be 'MANUAL_REVIEW', and the reasoning should clearly state what specific data is missing. The 'missing_data_request' field in JSON should detail this.

    Approval Criteria:
    1. Cost vs. Container Age:
        {chr(10).join([f"    - {age_range}: Max approved cost ${threshold}" for age_range, threshold in age_cost_thresholds.items()])}
    2. Image Requirements:
        Repair codes needing images: {', '.join(repair_codes_needing_images.keys()) if repair_codes_needing_images else "None specified"}.
        For each suggested repair:
            - If its code requires an image AND an image for that specific repair code is NOT listed as provided, this is considered missing data.
    3. Image Content Analysis (from YOLOv8):
        Review the 'Image Analysis Summary' section below. Consider if detected damages align with suggested repairs and their severity.
        If YOLOv8 detects significant damages not listed in repairs, or if detected damages seem minor for high-cost repairs, flag for 'MANUAL_REVIEW'.

    Ticket Details:
    - Ticket ID: {ticket_data['ticket_id']}
    - Container ID: {ticket_data['container_id']}
    - Company: {ticket_data['company']}
    - Container Age (years): {ticket_data['container_age']}
    - Total Cost Estimate: ${ticket_data['total_cost_estimate']}
    - Suggested Repairs:
    {'\n'.join(repair_details_prompt)}
    - Other Notes: {ticket_data.get('other_notes', 'None')}
    - Media Provided: {len(ticket_data.get('media', []))} files. ({', '.join(media_summary)}){yolo_analysis_prompt_section}

    Based on all the above (and assuming all necessary data like images IS present if not flagged as missing), provide your response strictly in the following JSON format:
    {{
    "decision": "APPROVE" / "DISAPPROVE" / "MANUAL_REVIEW",
    "confidence_score": <float between 0.0 and 1.0>,
    "reasoning": "<concise explanation for the decision, including specific criteria met or failed. If 'MANUAL_REVIEW' due to low confidence or complex rules, explain why.>",
    "missing_data_request": null
    }}
    """

    try:
        response = cohere_client.chat(model=app_settings.cohere_model, message=prompt)
        if success_callback:
            success_callback("AI successfully processed the repair estimate.")
        ai_response_text = response.text
        json_start = ai_response_text.find("{")
        json_end = ai_response_text.rfind("}") + 1
        if json_start != -1 and json_end != -1:
            parsed_response = json.loads(ai_response_text[json_start:json_end])
            if not all(
                key in parsed_response
                for key in ["decision", "confidence_score", "reasoning"]
            ):
                raise ValueError("LLM response missing required JSON keys.")
            parsed_response.setdefault("missing_data_request", None)
            return parsed_response
        if error_callback:
            error_callback(f"Could not parse JSON from Cohere response: {ai_response_text}")
        return {
            "decision": "MANUAL_REVIEW",
            "confidence_score": 0.1,
            "reasoning": "Error: Could not parse AI response.",
            "missing_data_request": "AI response parsing error.",
        }
    except Exception as exc:  # pragma: no cover - network/API dependent
        if error_callback:
            error_callback(f"Error calling Cohere API: {exc}")
        return {
            "decision": "MANUAL_REVIEW",
            "confidence_score": 0.0,
            "reasoning": f"API error: {exc}",
            "missing_data_request": "API communication failure.",
        }


def process_ticket_with_ai(
    ticket_data: dict,
    age_cost_thresholds: dict[str, int],
    repair_codes_needing_images: dict[str, str],
    use_cohere_aya: bool = True,
    model_config: dict | None = None,
    settings: Settings | None = None,
    info_callback: Callable[[str], None] | None = None,
    error_callback: Callable[[str], None] | None = None,
    success_callback: Callable[[str], None] | None = None,
    warning_callback: Callable[[str], None] | None = None,
) -> dict:
    app_settings = settings or get_settings()
    custom_model_config = model_config or {
        "api_key": app_settings.custom_model_api_key,
        "endpoint": app_settings.custom_model_endpoint,
    }

    if use_cohere_aya:
        ai_result = call_cohere_aya_yolo_model(
            ticket_data=ticket_data,
            age_cost_thresholds=age_cost_thresholds,
            repair_codes_needing_images=repair_codes_needing_images,
            settings=app_settings,
            error_callback=error_callback,
            success_callback=success_callback,
            warning_callback=warning_callback,
        )
        ai_result["ai_agent_type"] = (
            f"Cohere Aya ({app_settings.cohere_model or DEFAULT_COHERE_MODEL})"
        )
    else:
        ai_result = call_custom_model(
            ticket_data=ticket_data,
            model_config=custom_model_config,
            info_callback=info_callback,
        )
        ai_result["ai_agent_type"] = "Custom Model (Placeholder)"
    return ai_result
