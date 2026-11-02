import importlib
import json
import os
import unittest
from unittest.mock import patch


class ScoreProspectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tracing = patch.dict(os.environ, {"LANGSMITH_TRACING": "false"})
        tracing.start()
        cls.addClassCleanup(tracing.stop)
        with patch("langchain_openai.ChatOpenAI") as model, \
                patch("deepagents.create_deep_agent"):
            cls.agent = importlib.import_module("gtm_agent.gtm_agent")
            cls.output_schema = model.return_value.with_structured_output.call_args.args[0]

    def setUp(self):
        self.data = self.agent.data_service
        self.cache = patch.object(self.data, "_PROFILES", {})
        self.cache.start()
        self.addCleanup(self.cache.stop)
        self.prospect_id = next(iter(self.data.PROSPECTS))
        self.offering_id = next(iter(self.data.OFFERINGS))
        self.args = {"prospect_id": self.prospect_id, "offering_id": self.offering_id}
        self.score = self.agent.ProspectScore(
            score=75,
            justification="Good fit with some missing technology.",
            rubric_breakdown=self.agent.RubricBreakdown(
                revenue_fit=80, tech_stack_match=60, segment_fit=85,
            ),
        )
        self.scorer = patch.object(self.agent._scoring_llm, "invoke", return_value=self.score)
        self.invoke = self.scorer.start()
        self.addCleanup(self.scorer.stop)

    def assert_scoring_inputs(self, profile):
        self.assertIs(self.output_schema, self.agent.ProspectScore)
        self.assertEqual(self.invoke.call_args.args[0], [
            {"role": "system", "content": self.agent.SCORING_PROMPT},
            {"role": "user", "content": (
                "Offering:\n" + json.dumps(self.data.get_offering(self.offering_id), indent=2)
                + "\n\nProspect profile:\n" + json.dumps(profile, indent=2)
            )},
        ])

    def test_score_with_only_ids_builds_and_stores_profile(self):
        self.assertEqual(self.agent.score_prospect.invoke(self.args), self.score.model_dump())
        profile = self.data.get_profile_from_db(self.prospect_id)["prospect_profile"]
        self.assertEqual(profile, {"prospect_id": self.prospect_id,
                                   **self.data.get_prospect_record(self.prospect_id)})
        self.assert_scoring_inputs(profile)
        self.assertEqual(self.agent.build_prospect_profile.invoke(
            {"prospect_id": self.prospect_id}), {"prospect_profile": profile, "found": True})
        schema = self.agent.score_prospect.args_schema.model_json_schema()
        self.assertEqual(set(schema["properties"]), {"prospect_id", "offering_id"})
        self.assertEqual(set(schema["required"]), {"prospect_id", "offering_id"})
        self.assertIn("the tool loads both records itself", self.agent.score_prospect.description)

    def test_build_profile_then_score_reuses_stored_profile(self):
        profile = self.agent.build_prospect_profile.invoke(
            {"prospect_id": self.prospect_id})["prospect_profile"]
        profile["name"] = "Stored profile name"
        profile["tech_stack"] = ["Stale technology"]
        with patch.object(self.data, "get_prospect_record", side_effect=AssertionError("Rebuilt profile")), \
                patch.object(self.data, "fetch_tech_stack", return_value=["Current technology"]):
            self.assertEqual(self.agent.build_prospect_profile.invoke(
                {"prospect_id": self.prospect_id}), {"prospect_profile": profile, "found": True})
            self.assertEqual(self.agent.score_prospect.invoke(self.args), self.score.model_dump())
        self.assert_scoring_inputs({**profile, "tech_stack": ["Current technology"]})
        self.assertEqual(profile["tech_stack"], ["Stale technology"])

    def test_tool_call_with_only_ids(self):
        from langchain_core.messages import AIMessage
        from langgraph.graph import END, START, MessagesState, StateGraph
        from langgraph.prebuilt import ToolNode

        request = AIMessage(content="", tool_calls=[{
            "name": "score_prospect", "args": self.args, "id": "score-request",
        }])
        graph = StateGraph(MessagesState)
        graph.add_node("tools", ToolNode([self.agent.score_prospect]))
        graph.add_edge(START, "tools")
        graph.add_edge("tools", END)
        response = graph.compile().invoke({"messages": [request]})
        result = response["messages"][-1]
        self.assertEqual(result.tool_call_id, "score-request")
        self.assertEqual(json.loads(result.content), self.score.model_dump())

    def test_missing_offering(self):
        self.assertEqual(self.agent.score_prospect.invoke(
            {**self.args, "offering_id": "unknown"}),
            {"score": None, "error": "Cannot score without a valid offering."})
        self.invoke.assert_not_called()

    def test_invalid_offering(self):
        offering = self.data.get_offering(self.offering_id)
        for field in ("required_tech_stack", "min_annual_revenue", "description"):
            with self.subTest(field=field), patch.object(
                    self.data, "get_offering", return_value={**offering, field: None}):
                self.assertEqual(self.agent.score_prospect.invoke(self.args),
                                 {"score": None, "error": "Cannot score without a valid offering."})
        self.invoke.assert_not_called()

    def test_unknown_prospect(self):
        self.assertEqual(self.agent.build_prospect_profile.invoke({"prospect_id": "unknown"}),
                         {"prospect_profile": None, "found": False})
        self.assertEqual(self.agent.score_prospect.invoke({**self.args, "prospect_id": "unknown"}),
                         {"score": None, "error": "Unknown prospect."})
        self.invoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()
