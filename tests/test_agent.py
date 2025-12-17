import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent import (
    create_deep_agent,
    DeepAgentConfig,
    calculator_tool,
    search_tool,
    get_current_time_tool,
)


class TestTools:
    def test_calculator_tool_addition(self):
        result = calculator_tool.invoke({"expression": "2 + 2"})
        assert "4" in result

    def test_calculator_tool_complex_expression(self):
        result = calculator_tool.invoke({"expression": "10 * 5 + 3"})
        assert "53" in result

    def test_calculator_tool_invalid_expression(self):
        result = calculator_tool.invoke({"expression": "invalid"})
        assert "Error" in result

    def test_get_current_time_tool(self):
        result = get_current_time_tool.invoke({})
        assert "Current time:" in result

    def test_search_tool(self):
        result = search_tool.invoke({"query": "test query"})
        assert "test query" in result


class TestDeepAgentConfig:
    def test_default_config(self):
        config = DeepAgentConfig()
        assert config.model_name == "openai:gpt-4o-mini"
        assert config.system_prompt is not None
        assert config.max_iterations == 10

    def test_custom_config(self):
        config = DeepAgentConfig(
            model_name="anthropic:claude-sonnet-4-20250514",
            system_prompt="Custom prompt",
            max_iterations=5,
        )
        assert config.model_name == "anthropic:claude-sonnet-4-20250514"
        assert config.system_prompt == "Custom prompt"
        assert config.max_iterations == 5


class TestCreateDeepAgent:
    @patch("agent.init_chat_model")
    def test_create_agent_returns_compiled_graph(self, mock_init_model):
        mock_model = MagicMock()
        mock_init_model.return_value = mock_model

        config = DeepAgentConfig()
        agent = create_deep_agent(config)

        assert agent is not None
        mock_init_model.assert_called_once()

    @patch("agent.init_chat_model")
    def test_create_agent_with_custom_tools(self, mock_init_model):
        mock_model = MagicMock()
        mock_init_model.return_value = mock_model

        def custom_tool(x: str) -> str:
            return f"Custom: {x}"

        config = DeepAgentConfig()
        agent = create_deep_agent(config, additional_tools=[custom_tool])

        assert agent is not None

    @patch("agent.init_chat_model")
    def test_create_agent_without_default_tools(self, mock_init_model):
        mock_model = MagicMock()
        mock_init_model.return_value = mock_model

        config = DeepAgentConfig()
        agent = create_deep_agent(config, include_default_tools=False)

        assert agent is not None
