import json
import os
import unittest
from copy import deepcopy
from unittest.mock import Mock, patch

from langsmith import tracing_context

from gtm_agent import data_service

with patch.dict(os.environ, {"OPENAI_API_KEY": "test", "LANGSMITH_TRACING": "false"}):
    from gtm_agent import gtm_agent as agent


class UpdateProspectInfoTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(tracing_context(enabled=False))
        self.prospect_id = "LEAD-71001"
        self.record = deepcopy(data_service.get_prospect_record(self.prospect_id))
        self.enterContext(patch.dict(data_service.PROSPECTS, {self.prospect_id: self.record}))
        self.enterContext(patch.dict(data_service._PROFILES, clear=True))

    def test_update_persists_and_builds_profile(self):
        expected = self.record["tech_stack"] + ["Kafka"]

        result = data_service.update_prospect_info(self.prospect_id, "Kafka")

        self.assertEqual(result, {"updated": True, "found": True, "tech_stack": expected})
        self.assertEqual(data_service.get_prospect_record(self.prospect_id)["tech_stack"], expected)
        self.assertEqual(data_service.fetch_tech_stack(self.prospect_id), expected)
        profile = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        self.assertEqual(profile["prospect_profile"]["tech_stack"], expected)

    def test_repeated_updates_refresh_cached_profile(self):
        agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        expected = list(self.record["tech_stack"])

        for technology in ("Kafka", "Redis", "Kafka"):
            with self.subTest(technology=technology):
                if technology not in expected:
                    expected.append(technology)
                result = data_service.update_prospect_info(self.prospect_id, technology)
                profile = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})

                self.assertEqual(result["tech_stack"], expected)
                self.assertEqual(data_service.fetch_tech_stack(self.prospect_id), expected)
                self.assertEqual(profile["prospect_profile"]["tech_stack"], expected)

    def test_scoring_uses_updated_record_even_with_stale_profile(self):
        stale_profile = deepcopy(agent.build_prospect_profile.invoke(
            {"prospect_id": self.prospect_id}
        )["prospect_profile"])
        result = agent.update_prospect_info.invoke(
            {"prospect_id": self.prospect_id, "technology": "Kafka"}
        )
        scoring_llm = Mock()
        scoring_llm.invoke.return_value = agent.ProspectScore(
            score=90,
            justification="Technology profile matches.",
            rubric_breakdown=agent.RubricBreakdown(
                revenue_fit=90, tech_stack_match=100, segment_fit=90
            ),
        )

        with patch.object(agent, "_scoring_llm", scoring_llm):
            agent.score_prospect.invoke({
                "prospect_profile": stale_profile,
                "offering": data_service.get_offering("OFFER-10005"),
            })

        content = scoring_llm.invoke.call_args.args[0][1]["content"]
        scored_profile = json.loads(content.split("\n\nProspect profile:\n", 1)[1])
        self.assertNotIn("Kafka", stale_profile["tech_stack"])
        self.assertIn("Kafka", scored_profile["tech_stack"])
        self.assertEqual(scored_profile["tech_stack"], result["tech_stack"])
        self.assertEqual(scored_profile["tech_stack"], data_service.fetch_tech_stack(self.prospect_id))

    def test_unknown_prospect_is_not_updated(self):
        self.assertEqual(
            data_service.update_prospect_info("unknown", "Kafka"),
            {"updated": False, "found": False},
        )
        self.assertIsNone(data_service.get_prospect_record("unknown"))


if __name__ == "__main__":
    unittest.main()
