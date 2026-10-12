import os
import unittest
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test")

agent_module = import_module("gtm_agent.gtm_agent")
SYSTEM_PROMPT = agent_module.SYSTEM_PROMPT
send_prospect_email = agent_module.send_prospect_email


class SendProspectEmailTests(unittest.TestCase):
    def call_tool(self, prospect):
        return send_prospect_email.func(
            prospect=prospect,
            subject="Hello",
            body="A note",
            runtime=SimpleNamespace(config={"metadata": {}}),
            from_rep={"email": "rep@example.com", "name": "Rep"},
        )

    def test_disqualified_prospect_is_blocked_before_message_id(self):
        with patch("gtm_agent.gtm_agent.uuid.uuid4") as make_id:
            result = self.call_tool({"prospect_id": "LEAD-50001", "email": "model@example.com"})

        self.assertEqual(
            result,
            {"status": "blocked", "error": "Prospect is disqualified; email not sent."},
        )
        make_id.assert_not_called()

    def test_missing_prospect_record_is_blocked_before_message_id(self):
        with patch("gtm_agent.gtm_agent.uuid.uuid4") as make_id:
            result = self.call_tool({"prospect_id": "LEAD-DOES-NOT-EXIST"})

        self.assertEqual(
            result,
            {"status": "blocked", "error": "Prospect record was not found; email not sent."},
        )
        make_id.assert_not_called()

    def test_eligible_prospect_uses_server_side_contact_details(self):
        record = {"name": "Source Name", "email": "source@example.com", "disqualified": False}
        with patch("gtm_agent.gtm_agent.data_service.get_prospect_record", return_value=record), \
                patch("gtm_agent.gtm_agent.uuid.uuid4", return_value=SimpleNamespace(hex="123456789abc")):
            result = self.call_tool({
                "prospect_id": "LEAD-12853",
                "name": "Model Name",
                "email": "model@example.com",
            })

        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["to"], "source@example.com")
        self.assertEqual(result["to_name"], "Source Name")

    def test_blocked_send_instructions_prevent_retry_and_explain_reason(self):
        self.assertIn("do not retry", SYSTEM_PROMPT)
        self.assertIn("tell the rep the reason returned", SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
