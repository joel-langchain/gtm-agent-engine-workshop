import importlib
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langsmith import tracing_context

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}), \
        patch("dotenv.load_dotenv"):
    agent = importlib.import_module("gtm_agent.gtm_agent")

from gtm_agent import data_service


class ProspectPrivacyTests(unittest.TestCase):
    def setUp(self):
        self.cache = patch.dict(data_service._PROFILES, {}, clear=True)
        self.cache.start()
        self.addCleanup(self.cache.stop)
        self.tracing = tracing_context(enabled=False)
        self.tracing.__enter__()
        self.addCleanup(self.tracing.__exit__, None, None, None)
        self.prospect_id = "LEAD-70001"
        self.record = data_service.get_prospect_record(self.prospect_id)

    def assert_redacted(self, value):
        serialized = json.dumps(value)
        self.assertNotIn("billing_qualification", serialized)
        for secret in self.record["billing_qualification"].values():
            self.assertNotIn(secret, serialized)

    def test_contact_lookup_and_email_eligibility(self):
        for prospect_id, record in data_service.PROSPECTS.items():
            with self.subTest(prospect_id=prospect_id):
                result = agent.get_prospect.invoke({"prospect_id": prospect_id})
                self.assertEqual(result, {"found": True, "prospect": {
                    "prospect_id": prospect_id,
                    **{key: record[key] for key in ("name", "email", "disqualified")},
                }})
                runtime = SimpleNamespace(config={"metadata": {"user_id": "rep_amills"}})
                sent = agent.send_prospect_email.func(
                    result["prospect"], "Hello", "Following up", runtime
                )
                self.assertEqual(sent["status"], "blocked" if record["disqualified"] else "sent")
                if not record["disqualified"]:
                    self.assertEqual(sent["to"], record["email"])
        self.assertEqual(agent.get_prospect.invoke({"prospect_id": "missing"}),
                         {"prospect": None, "found": False})

    def test_new_profile_and_scoring_arguments(self):
        result = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        self.assertTrue(result["found"])
        self.assert_redacted(result)
        self.assert_redacted(data_service._PROFILES[self.prospect_id])
        profile = result["prospect_profile"]
        for key in ("name", "email", "disqualified", "annual_revenue", "tech_stack"):
            self.assertEqual(profile[key], self.record[key])
        self.check_scoring(profile)

    def test_cached_profile_and_traced_lookup(self):
        data_service._PROFILES[self.prospect_id] = {"prospect_id": self.prospect_id, **self.record}
        runs = []
        with tracing_context(enabled="local"):
            cached = data_service.get_profile_from_db(
                self.prospect_id, langsmith_extra={"on_end": runs.append}
            )
        self.assertEqual(len(runs), 1)
        self.assert_redacted(runs[0].outputs)
        self.assert_redacted(cached)
        result = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        self.assertEqual(result["prospect_profile"], cached["prospect_profile"])
        self.assert_redacted(result)
        self.assertEqual(result["prospect_profile"]["email"], self.record["email"])

    def check_scoring(self, profile):
        expected = {"score": 75, "justification": "Suitable fit"}
        model = Mock()
        model.invoke.return_value.model_dump.return_value = expected
        offering = data_service.get_offering("OFFER-10001")
        with patch.object(agent, "_scoring_llm", model):
            result = agent.score_prospect.invoke({"prospect_profile": profile, "offering": offering})
        self.assertEqual(result, expected)
        messages = model.invoke.call_args.args[0]
        self.assert_redacted(messages)
        scored = json.loads(messages[1]["content"].split("Prospect profile:\n", 1)[1])
        self.assertEqual(scored["annual_revenue"], self.record["annual_revenue"])
        self.assertEqual(scored["tech_stack"], self.record["tech_stack"])

    def test_scoring_redacts_untrusted_profile(self):
        profile = {"prospect_id": self.prospect_id, **self.record}
        self.check_scoring(profile)
        self.assertIn("billing_qualification", profile)


if __name__ == "__main__":
    unittest.main()
