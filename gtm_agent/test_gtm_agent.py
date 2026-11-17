"""Tests for signed-in sender context and tool availability."""

import importlib
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

with patch.dict(os.environ, {"OPENAI_API_KEY": "test", "LANGSMITH_TRACING": "false"}):
    module = importlib.import_module("gtm_agent.gtm_agent")


class SenderContextTests(unittest.TestCase):
    def test_email_graph_uses_sender_context_without_lookup_turn(self):
        from langchain_core.messages import AIMessage, ToolMessage
        from langchain_core.outputs import ChatGeneration, ChatResult

        for rep in module.REP_IDS[:2]:
            calls = []

            def respond(messages, **kwargs):
                tool_results = [m for m in messages if isinstance(m, ToolMessage)]
                if not tool_results:
                    context = next(m.content for m in messages if "Signed-in sender identity" in m.content)
                    identity = json.loads(context.split(": ", 1)[1])
                    self.assertEqual(identity, rep)
                    name, args = "get_prospect", {"prospect_id": "LEAD-12853"}
                elif len(tool_results) == 1:
                    prospect = json.loads(tool_results[0].content)["prospect"]
                    name, args = "send_prospect_email", {
                        "prospect": prospect, "subject": "Demo",
                        "body": "Let's schedule a demo.\n\n" + rep["name"],
                    }
                else:
                    return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Sent"))])
                calls.append(name)
                return ChatResult(generations=[ChatGeneration(message=AIMessage(
                    content="", tool_calls=[{"name": name, "args": args, "id": str(len(calls))}],
                ))])

            with self.subTest(rep=rep["rep_id"]):
                with patch.object(module.ChatOpenAI, "_generate", side_effect=respond):
                    result = module.run_agent("Email LEAD-12853 to schedule a demo.", user_id=rep["rep_id"])
                self.assertEqual(calls, ["get_prospect", "send_prospect_email"])
                sent = json.loads(next(m.content for m in result["messages"] if isinstance(m, ToolMessage) and m.name == "send_prospect_email"))
                self.assertEqual(sent["from"], rep["email"])
                self.assertTrue(sent["body"].endswith(rep["name"]))

    def test_requests_receive_signed_in_identity(self):
        requests = (
            "Email LEAD-12853 to schedule a demo.",
            "Score LEAD-71001 against OFFER-10005.",
            "Add Kafka to LEAD-71001 then score against OFFER-10005.",
        )
        for rep in module.REP_IDS[:2]:
            for request in requests:
                with self.subTest(rep=rep["rep_id"], request=request):
                    with patch.object(module.gtm_agent, "invoke") as invoke:
                        invoke.return_value = {"messages": [SimpleNamespace(content="Done")]}
                        result = module.run_agent(request, user_id=rep["rep_id"])
                    payload = invoke.call_args.args[0]
                    context, user = payload["messages"]
                    self.assertEqual(context["role"], "system")
                    identity = json.loads(context["content"].split(": ", 1)[1])
                    self.assertEqual(identity, rep)
                    self.assertEqual(user, {"role": "user", "content": request})
                    self.assertEqual(invoke.call_args.kwargs["config"]["metadata"]["user_id"], rep["rep_id"])
                    self.assertEqual(result["reply"], "Done")

    def test_unknown_rep_proceeds_without_identity(self):
        with patch.object(module.gtm_agent, "invoke") as invoke:
            invoke.return_value = {"messages": [SimpleNamespace(content="Done")]}
            module.run_agent("Score a prospect", user_id="unknown")
        self.assertEqual(invoke.call_args.args[0]["messages"], [
            {"role": "user", "content": "Score a prospect"},
        ])
        runtime = SimpleNamespace(config={"metadata": {"user_id": "unknown"}})
        self.assertEqual(module.get_current_rep.func(runtime), {"rep": None, "found": False})

    def test_lookup_remains_registered_and_callable(self):
        tools = module.gtm_agent.nodes["tools"].bound.tools_by_name
        self.assertIn("get_current_rep", tools)
        rep = module.REP_IDS[0]
        runtime = SimpleNamespace(config={"metadata": {"user_id": rep["rep_id"]}})
        self.assertEqual(tools["get_current_rep"].func(runtime), {"rep": rep, "found": True})

    def test_email_sender_defaults_to_metadata(self):
        for rep in module.REP_IDS[:2]:
            with self.subTest(rep=rep["rep_id"]):
                runtime = SimpleNamespace(config={"metadata": {"user_id": rep["rep_id"]}})
                result = module.send_prospect_email.func(
                    prospect={"name": "Prospect", "email": "prospect@example.com"},
                    subject="Demo", body="Let's schedule a demo.", runtime=runtime,
                )
                self.assertEqual(result["status"], "sent")
                self.assertEqual(result["from"], rep["email"])
                self.assertEqual(result["from_name"], rep["name"])


if __name__ == "__main__":
    unittest.main()
