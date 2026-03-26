from __future__ import annotations

import json
from typing import Callable


def create_cohere_client(api_key: str):
    import cohere

    return cohere.Client(api_key)


def extract_json_object(text: str) -> dict:
    json_start = text.find("{")
    json_end = text.rfind("}") + 1
    if json_start == -1 or json_end == 0:
        return {}
    return json.loads(text[json_start:json_end])


def analyze_sop_with_cohere(
    ocr_result,
    api_key: str,
    model_name: str,
    error_callback: Callable[[str], None] | None = None,
) -> dict | None:
    if not api_key or api_key == "YOUR_COHERE_API_KEY_HERE":
        if error_callback:
            error_callback(
                "Cohere API key not configured. Please set COHERE_API_KEY environment variable."
            )
        return None

    try:
        client = create_cohere_client(api_key)
        full_text = "\n".join(page.markdown for page in ocr_result.pages)
        prompt = f"""You are an AI assistant analyzing a Standard Operating Procedure (SOP) document for container repairs.
        Your task is to identify repair codes that require images for documentation.

        Document content:
        {full_text}

        Please analyze the text and identify any repair codes that explicitly require images or visual documentation.
        For each identified code, provide:
        1. The repair code (e.g., DMG001)
        2. A brief description of why images are required

        Format your response as a JSON object with repair codes as keys and descriptions as values.
        Example format:
        {{
            "DMG001": "Severe damage assessment requiring visual documentation",
            "CRK003": "Crack inspection requiring photographic evidence"
        }}

        Only include codes that explicitly mention image requirements. If no codes are found, return an empty object {{}}.
        """
        response = client.chat(model=model_name, message=prompt)
        return extract_json_object(response.text)
    except json.JSONDecodeError:
        if error_callback:
            error_callback("Could not parse Cohere response as JSON")
        return {}
    except Exception as exc:  # pragma: no cover - network/API dependent
        if error_callback:
            error_callback(f"Error analyzing SOP with Cohere: {exc}")
        return None


def chat_with_ai_agent(
    ticket_data: dict,
    user_question: str,
    chat_history: list[dict],
    repair_codes_requiring_images: dict[str, str],
    api_key: str,
    model_name: str,
    error_callback: Callable[[str], None] | None = None,
) -> str | None:
    if not api_key or api_key == "YOUR_COHERE_API_KEY_HERE":
        if error_callback:
            error_callback(
                "Cohere API key not configured. Please set COHERE_API_KEY environment variable."
            )
        return None

    try:
        client = create_cohere_client(api_key)
        repairs_text = "\n".join(
            [
                f"- {repair.get('code', 'N/A')}: "
                f"{repair.get('description', 'No description')}"
                for repair in ticket_data.get("repairs", [])
            ]
        )
        media_text = "\n".join(
            [
                f"- {media.get('filename', 'Unknown file')} "
                f"({media.get('type', 'unknown')})"
                for media in ticket_data.get("media", [])
            ]
        )
        image_req_text = "\n".join(
            [f"- {code}: {desc}" for code, desc in repair_codes_requiring_images.items()]
        ) or "None currently configured"
        relevant_chat = [
            msg
            for msg in chat_history[-5:]
            if msg.get("sender") in ["AI Agent", "System"]
        ]
        chat_context = "\n".join(
            [f"{msg['sender']}: {msg['message']}" for msg in relevant_chat]
        )

        prompt = f"""You are the AI Repair Ticket Approval Agent that previously analyzed this container repair ticket.
        You can answer questions about your decision, reasoning, and provide clarifications about repair approvals.

        TICKET CONTEXT:
        - Ticket ID: {ticket_data['ticket_id']}
        - Container ID: {ticket_data['container_id']}
        - Company: {ticket_data['company']}
        - Container Age: {ticket_data['container_age']} years
        - Total Cost: ${ticket_data['total_cost_estimate']}
        - Repairs: {repairs_text if repairs_text else "None listed"}
        - Media Files: {media_text if media_text else "None provided"}
        - Other Notes: {ticket_data.get('other_notes', 'None')}

        CURRENT SYSTEM RULES - REPAIR CODES REQUIRING IMAGES:
        {image_req_text}

        YOUR PREVIOUS DECISION:
        - Decision: {ticket_data.get('ai_decision', 'Not yet decided')}
        - Confidence: {ticket_data.get('ai_confidence', 'N/A')}
        - Reasoning: {ticket_data.get('ai_reasoning', 'No reasoning provided')}
        - Missing Data Request: {ticket_data.get('ai_missing_data_request', 'None')}

        RECENT CONVERSATION:
        {chat_context if chat_context else "No previous AI conversation"}

        USER QUESTION: {user_question}

        Please respond as the AI agent that made the original decision. Be helpful, explain your reasoning clearly,
        and provide specific guidance. If the user asks about changing your decision, explain what would be needed.
        When discussing image requirements, refer to the current system rules above.
        Keep your response conversational and professional.
        """

        response = client.chat(model=model_name, message=prompt)
        return response.text
    except Exception as exc:  # pragma: no cover - network/API dependent
        if error_callback:
            error_callback(f"Error chatting with AI agent: {exc}")
        return f"Error: Unable to communicate with AI agent. {exc}"
