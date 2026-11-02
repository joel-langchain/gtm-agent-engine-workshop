import importlib
import json
import os
import unittest
import uuid
from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"},
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.agent = importlib.import_module("gtm_agent.gtm_agent")

    def test_crm_tools_and_runtime_injection(self):
        rep = self.agent.data_service.REP_IDS[0]
        prospect_id = next(
            pid for pid, record in self.agent.data_service.PROSPECTS.items()
            if not record.get("disqualified")
        )
        responses = iter([
            AIMessage(content="", tool_calls=[
                {"name": "get_current_rep", "args": {}, "id": "rep"},
            ]),
            AIMessage(content="", tool_calls=[{
                "name": "send_prospect_email",
                "args": {
                    "prospect": {"prospect_id": prospect_id},
                    "subject": "Follow up",
                    "body": "Thanks for your time.",
                },
                "id": "email",
            }]),
            AIMessage(content="Email sent."),
        ])
        model_inputs = []
        bound_tools = []
        original_bind_tools = ChatOpenAI.bind_tools

        def bind_tools(model, tools, **kwargs):
            bound_tools.append(tools)
            return original_bind_tools(model, tools, **kwargs)

        def generate(model, messages, **kwargs):
            model_inputs.append(messages)
            return ChatResult(generations=[ChatGeneration(message=next(responses))])

        with patch.object(ChatOpenAI, "bind_tools", bind_tools), patch.object(
            ChatOpenAI, "_generate", generate
        ):
            result = self.agent.run_agent("Send a follow up email", user_id=rep["rep_id"])

        expected_tools = {
            "lookup_offering", "build_prospect_profile", "get_prospect",
            "send_prospect_email", "score_prospect", "update_prospect_info",
            "get_current_rep",
        }
        self.assertTrue(bound_tools)
        for tools in bound_tools:
            self.assertEqual({tool.name for tool in tools}, expected_tools)
            for tool in tools:
                self.assertNotIn("runtime", tool.tool_call_schema.model_json_schema()["properties"])
        for messages in model_inputs:
            self.assertEqual(
                [message.content for message in messages if message.type == "system"],
                [self.agent.SYSTEM_PROMPT],
            )
        tool_results = {
            message.name: json.loads(message.content)
            for message in result["messages"] if isinstance(message, ToolMessage)
        }
        self.assertEqual(tool_results["get_current_rep"], {"rep": rep, "found": True})
        email = tool_results["send_prospect_email"]
        self.assertEqual(email["status"], "sent")
        self.assertEqual(email["from"], rep["email"])
        self.assertEqual(email["from_name"], rep["name"])
        self.assertEqual(
            email["to"], self.agent.data_service.PROSPECTS[prospect_id]["email"]
        )
        self.assertEqual(result["reply"], "Email sent.")

    def test_run_agent_invoke_configuration(self):
        run_id = uuid.uuid4()
        with patch.object(self.agent.uuid, "uuid4", return_value=run_id), patch.object(
            self.agent.gtm_agent, "invoke",
            return_value={"messages": [AIMessage(content="Done.")]},
        ) as invoke:
            result = self.agent.run_agent(
                "Look up offering OFFER-10001",
                user_id="rep_amills",
                environment="test",
                thread_id="thread-123",
            )

        invoke.assert_called_once_with(
            {"messages": [{"role": "user", "content": "Look up offering OFFER-10001"}]},
            config={
                "run_name": "GTM Assistant",
                "run_id": run_id,
                "metadata": {
                    "thread_id": "thread-123",
                    "user_id": "rep_amills",
                    "environment": "test",
                    "request_intent": "lookup_offering",
                },
            },
        )
        self.assertEqual(result["run_id"], str(run_id))
        self.assertEqual(result["reply"], "Done.")


if __name__ == "__main__":
    unittest.main()
