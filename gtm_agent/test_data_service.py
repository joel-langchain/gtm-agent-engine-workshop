import importlib
import json
import os
import unittest
from copy import deepcopy
from unittest.mock import patch

from . import data_service

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    with patch("dotenv.load_dotenv"):
        agent = importlib.import_module("gtm_agent.gtm_agent")


class UpdateProspectInfoTests(unittest.TestCase):
    def setUp(self):
        self.prospects = patch.object(data_service, "PROSPECTS", deepcopy(data_service.PROSPECTS))
        self.prospects.start()
        self.addCleanup(self.prospects.stop)
        self.profiles = patch.object(data_service, "_PROFILES", {})
        self.profiles.start()
        self.addCleanup(self.profiles.stop)
        self.tracing = patch.dict(os.environ, {"LANGSMITH_TRACING": "false"})
        self.tracing.start()
        self.addCleanup(self.tracing.stop)

    def test_update_persists_and_builds_profile(self):
        result = agent.update_prospect_info.invoke({
            "prospect_id": "LEAD-39002", "technology": "Terraform",
        })
        self.assertTrue(result["updated"])
        self.assertTrue(result["found"])
        self.assertIn("Terraform", data_service.fetch_tech_stack("LEAD-39002"))
        self.assertEqual(result["tech_stack"], data_service.fetch_tech_stack("LEAD-39002"))
        profile = agent.build_prospect_profile.invoke({"prospect_id": "LEAD-39002"})
        self.assertIn("Terraform", profile["prospect_profile"]["tech_stack"])

    def test_update_refreshes_cached_profile(self):
        before = agent.build_prospect_profile.invoke({"prospect_id": "LEAD-39002"})
        self.assertNotIn("Terraform", before["prospect_profile"]["tech_stack"])
        data_service.update_prospect_info("LEAD-39002", "Terraform")
        after = agent.build_prospect_profile.invoke({"prospect_id": "LEAD-39002"})
        self.assertIn("Terraform", after["prospect_profile"]["tech_stack"])

    def test_score_uses_persisted_stack_even_with_stale_profile(self):
        profile = agent.build_prospect_profile.invoke({"prospect_id": "LEAD-39002"})
        data_service.update_prospect_info("LEAD-39002", "Terraform")
        offering = data_service.get_offering("OFFER-10004")
        with patch.object(agent, "_scoring_llm") as scorer:
            agent.score_prospect.invoke({
                "prospect_profile": profile["prospect_profile"], "offering": offering,
            })
        content = scorer.invoke.call_args.args[0][1]["content"]
        scored_profile = json.loads(content.split("\n\nProspect profile:\n", 1)[1])
        self.assertIn("Terraform", scored_profile["tech_stack"])
        self.assertTrue(set(offering["required_tech_stack"]).issubset(scored_profile["tech_stack"]))

    def test_duplicate_update_adds_technology_once(self):
        data_service.update_prospect_info("LEAD-39002", "Terraform")
        result = data_service.update_prospect_info("LEAD-39002", "Terraform")
        self.assertTrue(result["updated"])
        self.assertEqual(data_service.fetch_tech_stack("LEAD-39002").count("Terraform"), 1)

    def test_unknown_prospect_is_not_updated(self):
        self.assertEqual(
            data_service.update_prospect_info("unknown", "Terraform"),
            {"updated": False, "found": False},
        )
        self.assertIsNone(data_service.get_prospect_record("unknown"))


if __name__ == "__main__":
    unittest.main()
