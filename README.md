# AI Repair Ticket Approval Platform

A human-in-the-loop approval workflow for shipping-container repair tickets. The prototype combines deterministic approval rules, OCR, object detection, and LLM-assisted reasoning to help operations teams triage repair requests without removing human accountability.

Live prototype: https://approval-agent-logistics-production.up.railway.app/

## What It Shows

This project is a practical AI workflow system rather than a chat-only demo. It shows how different AI capabilities can be composed around an operational decision process:

- rule-based approval thresholds for predictable governance
- OCR extraction from SOP documents
- object detection for image evidence checks
- LLM-assisted explanation and edge-case reasoning
- queue-based human review states
- configurable settings for repair-code and cost policies
- Streamlit deployment on Railway

## Problem

Approval teams handling container repair tickets need to review cost, asset age, repair codes, SOP requirements, and image evidence. A fully manual process can create large review backlogs, while a fully automated process would be risky because approvals affect real operational spend.

The design goal is therefore not autonomous approval. The goal is decision support: classify easy cases, surface missing evidence, explain the reasoning, and keep ambiguous cases in a human review queue.

## Workflow

```text
Repair ticket
  -> rule checks for cost, age, and repair code
  -> OCR-assisted SOP extraction
  -> image evidence check with object detection
  -> LLM-assisted reasoning for nuanced cases
  -> approval queue state
  -> human review or AI-supported recommendation
```

Ticket states include:

- `Manual Review`
- `Additional Data Requested`
- `AI Approved`
- `AI Disapproved`

## Architecture

```text
Streamlit UI
  -> logistics_approval_agent.web
  -> decision engine service
  -> OCR / vision / LLM service adapters
  -> domain ticket model
  -> queue and settings state
```

Repository shape:

```text
src/logistics_approval_agent/
  domain/        ticket model and domain concepts
  services/      decision engine, OCR, vision, and LLM adapters
  web/           Streamlit application

tests/           focused tests for config and decision logic
infrastructure/  Railway/Docker deployment assets
```

## Key Features

- **Decision engine:** Combines deterministic cost and age checks with repair-code policy settings.
- **Evidence review:** Uses image analysis to support cases where visual documentation is required.
- **SOP extraction:** Uses OCR to extract repair codes that require mandatory documentation.
- **Human-in-the-loop queues:** Separates approved, rejected, missing-data, and manual-review cases.
- **Configurable policy controls:** Lets users adjust thresholds and evidence requirements in the UI.
- **Collaboration surface:** Supports notes and clarification around each ticket.

## Tech Stack

- Python
- Streamlit
- Cohere Aya for LLM-assisted reasoning
- Mistral OCR for SOP extraction
- YOLOv8 for object detection
- Railway for deployment
- Pytest for focused logic tests

## Local Development

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the app:

```powershell
python -m streamlit run src/logistics_approval_agent/web/streamlit_app.py
```

Run tests:

```powershell
pytest
```

## Deployment

The repo includes Railway deployment configuration:

- [railway.toml](railway.toml)
- [infrastructure/Dockerfile](infrastructure/Dockerfile)

Railway starts the Streamlit app with:

```text
python -m streamlit run src/logistics_approval_agent/web/streamlit_app.py --server.address=0.0.0.0 --server.port=${PORT:-8501} --server.headless=true
```

## Privacy And Safety

- The public prototype is intended for demonstration and portfolio review.
- The workflow keeps humans in control of ambiguous or high-impact decisions.
- API keys and provider credentials should be supplied through environment variables, not committed to the repo.
- Real production use would require stronger access control, audit logging, data retention policies, and review of model/provider behavior.

## Limitations

- This is a prototype, not a production claims or procurement system.
- Streamlit is appropriate for demonstration speed; a production version would likely separate frontend, backend API, storage, and auth.
- Model-assisted reasoning should be treated as advisory, especially where operational spend or contractual liability is involved.
- Tests currently focus on configuration and decision logic; broader UI and integration tests would be needed before production use.
