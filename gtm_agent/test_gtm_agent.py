import importlib
import json
import os
import unittest
import uuid
from unittest.mock import Mock, patch

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableLambda


class AgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test", "LANGSMITH_TRACING": "false"}):
            with patch("langchain.agents.create_agent", wraps=create_agent) as factory:
                cls.agent = importlib.import_module("gtm_agent.gtm_agent")
            cls.factory_args = factory.call_args

    def setUp(self):
        self.tracing = patch.dict(os.environ, {"LANGSMITH_TRACING": "false"})
        self.tracing.start()
        self.addCleanup(self.tracing.stop)

    def test_agent_configuration(self):
        self.assertIs(self.factory_args.kwargs["model"], self.agent.agent_model)
        self.assertEqual(self.factory_args.kwargs["tools"], [
            self.agent.lookup_offering,
            self.agent.build_prospect_profile,
            self.agent.get_prospect,
            self.agent.send_prospect_email,
            self.agent.score_prospect,
            self.agent.update_prospect_info,
            self.agent.get_current_rep,
        ])
        self.assertEqual(self.factory_args.kwargs["system_prompt"], self.agent.SYSTEM_PROMPT)

    def test_graph_uses_only_domain_context_and_passes_rep_to_tool(self):
        model = FakeMessagesListChatModel(responses=[
            AIMessage(content="", tool_calls=[{
                "name": "get_current_rep", "args": {}, "id": "rep-call",
            }]),
            AIMessage(content="Your rep is Ava Mills."),
        ])
        model_call = Mock(side_effect=model.invoke)
        with patch.object(type(self.agent.agent_model), "bind_tools",
                          return_value=RunnableLambda(model_call)) as bind_tools:
            result = self.agent.run_agent("Who am I?", user_id="rep_amills")
        self.assertEqual(result["reply"], "Your rep is Ava Mills.")
        self.assertEqual(model_call.call_count, 2)
        first_messages = model_call.call_args_list[0].args[0]
        self.assertEqual(len(first_messages), 2)
        self.assertEqual(first_messages[0], SystemMessage(content=self.agent.SYSTEM_PROMPT))
        self.assertEqual(first_messages[1].content, "Who am I?")
        self.assertEqual(bind_tools.call_args.args[0], self.factory_args.kwargs["tools"])
        tool_result = next(m for m in result["messages"] if isinstance(m, ToolMessage))
        self.assertEqual(json.loads(tool_result.content), {
            "rep": self.agent.data_service.get_rep("rep_amills"), "found": True,
        })

    def test_run_agent_preserves_invoke_config_and_result(self):
        run_id = uuid.UUID("00000000-0000-0000-0000-000000000001")
        messages = [AIMessage(content="Done")]
        with patch.object(self.agent, "gtm_agent") as graph:
            graph.invoke.return_value = {"messages": messages}
            with patch.object(self.agent.uuid, "uuid4", return_value=run_id):
                result = self.agent.run_agent(
                    "Score LEAD-71001 against OFFER-10005.",
                    user_id="rep_jchen", environment="test", thread_id="thread-1",
                )
        graph.invoke.assert_called_once_with(
            {"messages": [{"role": "user", "content": "Score LEAD-71001 against OFFER-10005."}]},
            config={
                "run_name": "GTM Assistant",
                "run_id": run_id,
                "metadata": {
                    "thread_id": "thread-1", "user_id": "rep_jchen",
                    "environment": "test", "request_intent": "score_prospect",
                },
            },
        )
        self.assertEqual(result, {"reply": "Done", "messages": messages, "run_id": str(run_id)})


if __name__ == "__main__":
    unittest.main()
