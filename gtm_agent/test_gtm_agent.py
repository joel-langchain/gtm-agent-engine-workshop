import copy
import importlib
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

with patch.dict(os.environ, {"OPENAI_API_KEY": "test", "LANGSMITH_TRACING": "false"}):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"LANGSMITH_TRACING": "false"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.records = patch.object(
            agent.data_service, "PROSPECTS", copy.deepcopy(agent.data_service.PROSPECTS)
        )
        self.records.start()
        self.addCleanup(self.records.stop)
        self.prospect_id = "LEAD-90001"
        self.offering_id = "OFFER-10001"
        self.score = agent.ProspectScore(
            score=85, justification="Good fit",
            rubric_breakdown=agent.RubricBreakdown(
                revenue_fit=90, tech_stack_match=80, segment_fit=85
            ),
        )
        self.scorer = patch.object(agent, "_scoring_llm", Mock())
        self.llm = self.scorer.start()
        self.addCleanup(self.scorer.stop)
        self.llm.invoke.return_value = self.score

    def test_score_loads_records_by_id(self):
        with patch.object(agent.data_service, "fetch_engagement_history", return_value=["recent"]), \
             patch.object(agent.data_service, "fetch_account_details", return_value=["account"]), \
             patch.object(agent.data_service, "fetch_tech_stack", return_value=["current"]):
            result = agent.score_prospect.invoke({
                "prospect_id": self.prospect_id, "offering_id": self.offering_id,
            })
        self.assertEqual(result, self.score.model_dump())
        messages = self.llm.invoke.call_args.args[0]
        self.assertEqual(messages[0]["content"], agent.SCORING_PROMPT)
        offering, profile = messages[1]["content"].removeprefix("Offering:\n").split(
            "\n\nProspect profile:\n"
        )
        self.assertEqual(json.loads(offering), agent.data_service.get_offering(self.offering_id))
        profile = json.loads(profile)
        self.assertEqual(profile["prospect_id"], self.prospect_id)
        self.assertEqual(profile["annual_revenue"], 80000000)
        self.assertEqual(profile["engagement_history"], ["recent"])
        self.assertEqual(profile["account_details"], ["account"])
        self.assertEqual(profile["tech_stack"], ["current"])
        self.assertNotIn("billing_qualification", profile)

    def test_missing_records(self):
        for prospect_id, offering_id, error in (
            ("missing", self.offering_id, "Unknown prospect: missing."),
            (self.prospect_id, "missing", "Unknown offering: missing."),
        ):
            with self.subTest(error=error):
                self.assertEqual(agent.score_prospect.invoke({
                    "prospect_id": prospect_id, "offering_id": offering_id,
                }), {"score": None, "error": error})
        self.llm.invoke.assert_not_called()

    def test_invalid_offering(self):
        for field in ("required_tech_stack", "min_annual_revenue", "description"):
            offering = dict(agent.data_service.get_offering(self.offering_id))
            del offering[field]
            with self.subTest(field=field), patch.object(
                agent.data_service, "get_offering", return_value=offering
            ):
                result = agent.score_prospect.invoke({
                    "prospect_id": self.prospect_id, "offering_id": self.offering_id,
                })
                self.assertIsNone(result["score"])
                self.assertIn("valid offering", result["error"])
        self.llm.invoke.assert_not_called()

    def test_email_rechecks_disqualification(self):
        prospect = {"prospect_id": self.prospect_id, "disqualified": False}
        agent.data_service.PROSPECTS[self.prospect_id]["disqualified"] = True
        result = agent.send_prospect_email.func(
            prospect, "Hello", "Body", SimpleNamespace(config={})
        )
        self.assertEqual(result["status"], "blocked")
        self.assertIn("disqualified", result["error"])

    def run_script(self, request, respond):
        def generate(messages, **kwargs):
            return ChatResult(generations=[ChatGeneration(message=respond(messages, kwargs))])

        with patch.object(agent.ChatOpenAI, "_generate", side_effect=generate):
            return agent.run_agent(request, user_id="rep_amills")

    def test_email_uses_rep_context_without_lookup(self):
        def respond(messages, kwargs):
            context = "\n".join(str(message.content) for message in messages)
            self.assertIn("Ava Mills", context)
            self.assertIn("ava.mills@northpoint.com", context)
            self.assertIn("get_current_rep", [tool["function"]["name"] for tool in kwargs["tools"]])
            if isinstance(messages[-1], ToolMessage):
                sent = json.loads(messages[-1].content)
                self.assertEqual(sent["from_name"], "Ava Mills")
                self.assertEqual(sent["from"], "ava.mills@northpoint.com")
                return AIMessage(content="Sent")
            return AIMessage(content="", tool_calls=[{
                "name": "send_prospect_email", "id": "email", "args": {
                    "prospect": {"prospect_id": self.prospect_id},
                    "subject": "Hello", "body": "Regards, Ava Mills",
                },
            }])

        result = self.run_script("Email the prospect", respond)
        self.assertEqual(result["reply"], "Sent")
        self.assertEqual([message.name for message in result["messages"]
                          if isinstance(message, ToolMessage)], ["send_prospect_email"])

    def test_agent_scores_with_only_ids(self):
        def respond(messages, kwargs):
            if isinstance(messages[-1], ToolMessage):
                self.assertEqual(json.loads(messages[-1].content), self.score.model_dump())
                return AIMessage(content="Score: 85")
            return AIMessage(content="", tool_calls=[{
                "name": "score_prospect", "id": "score", "args": {
                    "prospect_id": self.prospect_id, "offering_id": self.offering_id,
                },
            }])

        result = self.run_script("Score the prospect against the offering", respond)
        self.assertEqual(result["reply"], "Score: 85")
        self.assertEqual([message.name for message in result["messages"]
                          if isinstance(message, ToolMessage)], ["score_prospect"])

    def test_agent_decides_update_then_score_and_sees_errors(self):
        observed = []

        def respond(messages, kwargs):
            if isinstance(messages[-1], ToolMessage):
                observed.append(json.loads(messages[-1].content))
            steps = [
                ("update_prospect_info", {"prospect_id": "missing", "technology": "Kafka"}),
                ("score_prospect", {"prospect_id": "missing", "offering_id": self.offering_id}),
            ]
            if len(observed) == len(steps):
                return AIMessage(content="Unknown prospect")
            name, args = steps[len(observed)]
            return AIMessage(content="", tool_calls=[{
                "name": name, "args": args, "id": str(len(observed)),
            }])

        result = self.run_script("Update missing with Kafka then score it", respond)
        self.assertEqual([message.name for message in result["messages"]
                          if isinstance(message, ToolMessage)],
                         ["update_prospect_info", "score_prospect"])
        self.assertFalse(observed[0]["found"])
        self.assertEqual(observed[1], {"score": None, "error": "Unknown prospect: missing."})
        self.llm.invoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()
