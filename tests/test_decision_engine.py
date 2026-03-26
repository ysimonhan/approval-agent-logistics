import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from logistics_approval_agent.services.decision_engine import (
    determine_ticket_status,
    find_missing_required_images,
)


class DecisionEngineTests(unittest.TestCase):
    def test_find_missing_required_images_returns_only_missing_codes(self):
        ticket = {
            "repairs": [
                {"code": "DMG001", "description": "Structural damage"},
                {"code": "SCT002", "description": "Scratch"},
                {"code": "FLR001", "description": "Floor replacement"},
            ],
            "media": [
                {
                    "filename": "damage.jpg",
                    "type": "image",
                    "repair_code_association": "DMG001",
                }
            ],
        }
        required_codes = {
            "DMG001": "Structural damage",
            "FLR001": "Floor replacement",
        }

        self.assertEqual(find_missing_required_images(ticket, required_codes), ["FLR001"])

    def test_determine_ticket_status_prefers_missing_data_request(self):
        result = {
            "decision": "APPROVE",
            "confidence_score": 0.99,
            "missing_data_request": "Mandatory images missing for repair codes: FLR001.",
        }

        self.assertEqual(determine_ticket_status(result), "Additional Data Requested")

    def test_determine_ticket_status_uses_confident_approvals(self):
        result = {
            "decision": "APPROVE",
            "confidence_score": 0.8,
            "missing_data_request": None,
        }

        self.assertEqual(determine_ticket_status(result), "AI Approved")

    def test_determine_ticket_status_defaults_to_manual_review(self):
        result = {
            "decision": "MANUAL_REVIEW",
            "confidence_score": 0.2,
            "missing_data_request": None,
        }

        self.assertEqual(determine_ticket_status(result), "Manual Review Required")


if __name__ == "__main__":
    unittest.main()
