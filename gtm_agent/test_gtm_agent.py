"""Tests for prospect email eligibility and blocked-send replies."""

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

HAS_MODEL_KEY = bool(os.environ.get("OPENAI_API_KEY"))
with patch.dict(os.environ, {
    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY") or "test-key",
    "LANGSMITH_TRACING": "false",
}):
    from gtm_agent import data_service
    from gtm_agent.gtm_agent import run_agent, send_prospect_email


class SendProspectEmailTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(config={"metadata": {"user_id": "rep_amills"}})

    def send(self, prospect, **kwargs):
        return send_prospect_email.func(
            prospect, "Demo invitation", "Please book a demo.", self.runtime, **kwargs
        )

    def test_disqualified_source_blocks_with_and_without_id(self):
        for prospect_id, record in data_service.PROSPECTS.items():
            if not record.get("disqualified"):
                continue
            for include_id in (True, False):
                with self.subTest(prospect_id=prospect_id, include_id=include_id):
                    prospect = {"name": record["name"], "email": record["email"]}
                    if include_id:
                        prospect.update(prospect_id=prospect_id, disqualified=False)
                    result = self.send(prospect)
                    self.assertEqual(result["status"], "blocked")
                    self.assertIn("disqualified", result["error"])
                    self.assertIn("No email was sent", result["error"])
                    self.assertNotIn("message_id", result)

    def test_email_lookup_ignores_case_and_surrounding_whitespace(self):
        record = data_service.get_prospect_record("LEAD-50003")
        result = self.send({"name": record["name"], "email": " " + record["email"].upper() + " "})
        self.assertEqual(result["status"], "blocked")

    def test_source_id_takes_precedence_over_supplied_details(self):
        result = self.send({"prospect_id": "LEAD-50003", "email": "other@example.com"})
        self.assertEqual(result["status"], "blocked")

    def test_truthy_source_disqualification_blocks(self):
        record = {"name": "Prospect", "email": "prospect@example.com", "disqualified": "yes"}
        with patch.dict(data_service.PROSPECTS, {"test-lead": record}):
            for prospect in ({"prospect_id": "test-lead", "email": record["email"]},
                             {"name": record["name"], "email": record["email"]}):
                with self.subTest(prospect=prospect):
                    self.assertEqual(self.send(prospect)["status"], "blocked")

    def test_eligible_and_unmatched_recipients_still_send(self):
        record = data_service.get_prospect_record("LEAD-12853")
        for prospect in (
            {"prospect_id": "LEAD-12853", **record},
            {"name": record["name"], "email": record["email"]},
            {"name": "New prospect", "email": "new@example.com"},
        ):
            with self.subTest(prospect=prospect):
                result = self.send(prospect)
                self.assertEqual(result["status"], "sent")
                self.assertTrue(result["message_id"].startswith("msg-"))
                self.assertEqual(result["to"], prospect["email"])
                self.assertEqual(result["to_name"], prospect["name"])
                self.assertEqual(result["from"], data_service.get_rep("rep_amills")["email"])
                self.assertEqual(result["subject"], "Demo invitation")
                self.assertEqual(result["body"], "Please book a demo.")

    def test_explicit_sender_is_preserved(self):
        result = self.send({"email": "new@example.com"}, from_rep={"email": "rep@example.com", "name": "Rep"})
        self.assertEqual(result["from"], "rep@example.com")
        self.assertEqual(result["from_name"], "Rep")

    def test_missing_email_validation_is_preserved(self):
        for prospect in ({}, {"prospect_id": "LEAD-50003"}, {"email": ""}):
            with self.subTest(prospect=prospect):
                self.assertEqual(self.send(prospect), {
                    "status": "failed", "error": "Prospect record has no email address."
                })


@unittest.skipUnless(HAS_MODEL_KEY, "Live agent examples require OPENAI_API_KEY")
class EmailReplyTests(unittest.TestCase):
    def test_run_examples_explain_disqualification_without_sending(self):
        from run import EXAMPLES

        with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}):
            for user_id, request in EXAMPLES:
                if "LEAD-5000" not in request:
                    continue
                with self.subTest(request=request):
                    result = run_agent(request, user_id=user_id)
                    reply = result["reply"].lower()
                    self.assertIn("disqualified", reply)
                    self.assertRegex(reply, r"no email|not (?:been )?sent|did not send|didn't send|haven't sent|have not sent")
                    for message in result["messages"]:
                        if getattr(message, "name", None) == "send_prospect_email":
                            self.assertNotEqual(json.loads(message.content)["status"], "sent")


if __name__ == "__main__":
    unittest.main()
