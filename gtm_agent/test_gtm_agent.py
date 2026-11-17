import json
import unittest
from unittest.mock import Mock, patch

from langsmith import trace, tracing_context

with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    from gtm_agent import data_service
    from gtm_agent import gtm_agent as agent


class ProspectPrivacyTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict("os.environ", {"LANGSMITH_TRACING": "false"}))
        self.prospect_id = "LEAD-12853"
        self.record = {
            "name": "Test Prospect",
            "email": "prospect@example.com",
            "disqualified": False,
            "billing_qualification": {"card_on_file": "sensitive-test-value"},
            "engagement_history": [{"channel": "demo"}],
            "account_details": [{"annual_revenue": 50000000}],
            "tech_stack": ["AWS"],
            "other_field": "preserved in profiles",
        }
        self.enterContext(patch.dict(data_service.PROSPECTS, {self.prospect_id: self.record}))
        self.enterContext(patch.object(data_service, "_PROFILES", {}))
        self.enterContext(tracing_context(enabled=False))

    def test_contact_allowlist_and_optional_fields(self):
        for fields in (("name", "email", "disqualified"), ("email",), ()):
            with self.subTest(fields=fields):
                record = {key: value for key, value in self.record.items()
                          if key not in ("name", "email", "disqualified") or key in fields}
                with patch.dict(data_service.PROSPECTS, {self.prospect_id: record}):
                    result = agent.get_prospect.invoke({"prospect_id": self.prospect_id})
                self.assertEqual(result, {
                    "prospect": {"prospect_id": self.prospect_id,
                                 **{key: self.record[key] for key in fields}},
                    "found": True,
                })

    def test_missing_prospect(self):
        self.assertEqual(agent.get_prospect.invoke({"prospect_id": "missing"}),
                         {"prospect": None, "found": False})
        self.assertEqual(agent.build_prospect_profile.invoke({"prospect_id": "missing"}),
                         {"prospect_profile": None, "found": False})
        self.assertEqual(data_service.get_profile_from_db("missing"),
                         {"prospect_profile": None})

    def test_new_profile_is_sanitized_before_cache_save_trace(self):
        with tracing_context(enabled="local"):
            with trace("build profile") as root:
                result = agent.build_prospect_profile.func(self.prospect_id)
        expected = {"prospect_id": self.prospect_id,
                    **{key: value for key, value in self.record.items()
                       if key != "billing_qualification"}}
        self.assertEqual(result, {"prospect_profile": expected, "found": True})
        self.assertEqual(data_service._PROFILES[self.prospect_id], expected)
        saved = [run for run in root.child_runs if run.name == "save_profile_to_db"]
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0].inputs["profile"], expected)

    def test_cached_profile_is_sanitized_at_trace_boundary(self):
        cached = {"prospect_id": self.prospect_id, **self.record}
        data_service._PROFILES[self.prospect_id] = cached
        runs = []
        with tracing_context(enabled="local"):
            result = data_service.get_profile_from_db(
                self.prospect_id, langsmith_extra={"on_end": runs.append},
            )
        expected = {key: value for key, value in cached.items()
                    if key != "billing_qualification"}
        self.assertEqual(result, {"prospect_profile": expected})
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].outputs, result)
        self.assertEqual(agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id}),
                         {"prospect_profile": expected, "found": True})
        self.assertIn("billing_qualification", cached)

    def test_scoring_sanitizes_legacy_profile_and_preserves_scoring(self):
        profile = {"prospect_id": self.prospect_id, **self.record}
        offering = agent.lookup_offering.invoke({"offering_id": "OFFER-10001"})["offering"]
        score = agent.ProspectScore(
            score=75, justification="Fits AWS requirements",
            rubric_breakdown=agent.RubricBreakdown(
                revenue_fit=80, tech_stack_match=70, segment_fit=75),
        )
        llm = Mock()
        llm.invoke.return_value = score
        with patch.object(agent, "_scoring_llm", llm):
            result = agent.score_prospect.invoke({"prospect_profile": profile, "offering": offering})
        self.assertEqual(result, score.model_dump())
        messages = llm.invoke.call_args.args[0]
        self.assertNotIn("billing_qualification", json.dumps(messages))
        self.assertNotIn("sensitive-test-value", json.dumps(messages))
        sent_profile = json.loads(messages[1]["content"].split("Prospect profile:\n", 1)[1])
        self.assertEqual(sent_profile, {key: value for key, value in profile.items()
                                       if key != "billing_qualification"})
        self.assertIn("billing_qualification", profile)

    def test_sanitized_contact_supports_email(self):
        contact = agent.get_prospect.invoke({"prospect_id": self.prospect_id})["prospect"]
        result = agent.send_prospect_email.func(
            contact, "Demo follow-up", "Thanks for your interest.", runtime=None,
            from_rep={"name": "Test Rep", "email": "rep@example.com"},
        )
        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["to"], self.record["email"])
        self.assertEqual(result["to_name"], self.record["name"])
        self.assertNotIn("billing_qualification", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
