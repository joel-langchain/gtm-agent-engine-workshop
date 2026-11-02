import copy
import json
import os
import unittest
from unittest.mock import patch

with patch.dict(os.environ, {"OPENAI_API_KEY": "test", "LANGSMITH_TRACING": "false"}), \
        patch("dotenv.load_dotenv"):
    from gtm_agent import data_service
    from gtm_agent import gtm_agent as agent


class UpdateProspectInfoTests(unittest.TestCase):
    def setUp(self):
        self.prospect_id = "LEAD-71001"
        record = copy.deepcopy(data_service.get_prospect_record(self.prospect_id))
        self.records = patch.dict(data_service.PROSPECTS, {self.prospect_id: record})
        self.records.start()
        self.addCleanup(self.records.stop)
        self.profiles = patch.dict(data_service._PROFILES, {}, clear=True)
        self.profiles.start()
        self.addCleanup(self.profiles.stop)
        self.tracing = patch.dict(os.environ, {"LANGSMITH_TRACING": "false"})
        self.tracing.start()
        self.addCleanup(self.tracing.stop)

    def test_update_persists_and_returns_stored_tech_stack(self):
        original = list(data_service.fetch_tech_stack(self.prospect_id))

        result = data_service.update_prospect_info(self.prospect_id, "Kafka")

        expected = original + ["Kafka"]
        self.assertEqual(result, {"updated": True, "found": True, "tech_stack": expected})
        self.assertEqual(data_service.get_prospect_record(self.prospect_id)["tech_stack"], expected)
        self.assertEqual(data_service.fetch_tech_stack(self.prospect_id), expected)

    def test_update_refreshes_cached_profile_and_preserves_other_fields(self):
        cached = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})[
            "prospect_profile"
        ]
        cached = {**cached, "custom_field": "preserved"}
        data_service.save_profile_to_db(self.prospect_id, cached)
        expected = {**cached, "tech_stack": cached["tech_stack"] + ["Kafka"]}

        data_service.update_prospect_info(self.prospect_id, "Kafka")
        profile = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})[
            "prospect_profile"
        ]

        self.assertEqual(profile, expected)
        self.assertNotIn("Kafka", cached["tech_stack"])

    def test_repeated_update_does_not_duplicate_technology(self):
        data_service.update_prospect_info(self.prospect_id, "Kafka")
        result = data_service.update_prospect_info(self.prospect_id, "Kafka")

        self.assertEqual(data_service.fetch_tech_stack(self.prospect_id).count("Kafka"), 1)
        self.assertEqual(result["tech_stack"], data_service.fetch_tech_stack(self.prospect_id))

    def test_unknown_prospect_is_not_updated(self):
        self.assertEqual(
            data_service.update_prospect_info("unknown", "Kafka"),
            {"updated": False, "found": False},
        )

    def test_score_after_update_receives_added_technology_from_record(self):
        profile = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})[
            "prospect_profile"
        ]
        offering = data_service.get_offering("OFFER-10005")
        agent.update_prospect_info.invoke({"prospect_id": self.prospect_id, "technology": "Kafka"})

        with patch.object(agent, "_scoring_llm") as scoring_llm:
            agent.score_prospect.invoke({"prospect_profile": profile, "offering": offering})

        messages = scoring_llm.invoke.call_args.args[0]
        scored_profile = json.loads(messages[1]["content"].split("\n\nProspect profile:\n", 1)[1])
        self.assertNotIn("Kafka", profile["tech_stack"])
        self.assertIn("Kafka", scored_profile["tech_stack"])
        self.assertTrue(set(offering["required_tech_stack"]).issubset(scored_profile["tech_stack"]))


if __name__ == "__main__":
    unittest.main()
