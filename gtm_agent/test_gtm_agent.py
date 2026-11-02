import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

with patch.dict(os.environ, OPENAI_API_KEY="test-key", LANGSMITH_TRACING="false"):
    with patch("dotenv.load_dotenv"):
        from gtm_agent import data_service
        from gtm_agent import gtm_agent as agent


class ProspectDataTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, LANGSMITH_TRACING="false"))
        self.enterContext(patch.dict(data_service._PROFILES, {}, clear=True))
        self.prospect_id = "LEAD-12853"
        self.record = data_service.get_prospect_record(self.prospect_id)
        self.contact = {
            "prospect_id": self.prospect_id,
            **{key: self.record[key] for key in (
                "name", "email", "disqualified", "annual_revenue", "enrichment_source",
            )},
        }
        self.profile = {
            **self.contact,
            **{key: self.record[key] for key in (
                "engagement_history", "account_details", "tech_stack",
            )},
        }

    def assert_no_billing(self, value):
        serialized = json.dumps(value)
        self.assertNotIn("billing_qualification", serialized)
        for key, value in self.record["billing_qualification"].items():
            self.assertNotIn(key, serialized)
            self.assertNotIn(value, serialized)

    def test_contact_retains_email_fields_without_billing_or_unknown_fields(self):
        with patch.dict(self.record, {"unrelated_private_field": "private"}):
            result = agent.get_prospect.invoke({"prospect_id": self.prospect_id})
        self.assertEqual(result, {"prospect": self.contact, "found": True})
        self.assert_no_billing(result)
        sent = agent.send_prospect_email.func(
            result["prospect"], "Introduction", "Hello", runtime=None,
            from_rep={"name": "Rep", "email": "rep@example.com"},
        )
        self.assertEqual(sent["status"], "sent")
        self.assertEqual(sent["to"], self.contact["email"])
        self.assertEqual(sent["to_name"], self.contact["name"])
        self.assert_no_billing(sent)

    def test_profile_returns_and_caches_only_allowed_fields(self):
        with patch.dict(self.record, {"unrelated_private_field": "private"}):
            for _ in range(2):
                result = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
                self.assertEqual(result, {"prospect_profile": self.profile, "found": True})
                cached = data_service.get_profile_from_db(self.prospect_id)
                self.assertEqual(cached["prospect_profile"], self.profile)
                self.assert_no_billing(result)
                self.assert_no_billing(cached)

    def test_existing_cached_profile_is_sanitized(self):
        data_service.save_profile_to_db(self.prospect_id, {
            **self.profile,
            "billing_qualification": self.record["billing_qualification"],
            "unrelated_private_field": "private",
        })
        result = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        self.assertEqual(result["prospect_profile"], self.profile)
        self.assertEqual(
            data_service.get_profile_from_db(self.prospect_id)["prospect_profile"],
            self.profile,
        )
        self.assert_no_billing(result)

    def test_unknown_prospect_is_not_found(self):
        for tool, key in (
            (agent.get_prospect, "prospect"),
            (agent.build_prospect_profile, "prospect_profile"),
        ):
            self.assertEqual(tool.invoke({"prospect_id": "unknown"}), {key: None, "found": False})

    def test_scoring_uses_revenue_and_tech_stack_without_billing(self):
        profile = agent.build_prospect_profile.invoke(
            {"prospect_id": self.prospect_id},
        )["prospect_profile"]
        offering = agent.lookup_offering.invoke({"offering_id": "OFFER-10003"})["offering"]
        score = agent.ProspectScore(
            score=85, justification="Revenue and technology fit",
            rubric_breakdown=agent.RubricBreakdown(
                revenue_fit=90, tech_stack_match=80, segment_fit=85,
            ),
        )

        def score_messages(messages):
            self.assert_no_billing(messages)
            prompt = messages[-1]["content"]
            serialized_profile = prompt.split("\n\nProspect profile:\n", 1)[1]
            scored_profile = json.loads(serialized_profile)
            self.assertEqual(scored_profile, self.profile)
            self.assertEqual(scored_profile["annual_revenue"], self.record["annual_revenue"])
            self.assertEqual(scored_profile["tech_stack"], self.record["tech_stack"])
            return score

        with patch.object(agent, "_scoring_llm", SimpleNamespace(invoke=score_messages)):
            for injected in (False, True):
                with self.subTest(injected=injected):
                    supplied = dict(profile)
                    if injected:
                        supplied["billing_qualification"] = self.record["billing_qualification"]
                    result = agent.score_prospect.invoke({
                        "prospect_profile": supplied, "offering": offering,
                    })
                    self.assertEqual(result, score.model_dump())
                    self.assertEqual("billing_qualification" in supplied, injected)


if __name__ == "__main__":
    unittest.main()
