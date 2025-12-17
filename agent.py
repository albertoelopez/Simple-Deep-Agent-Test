from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Sequence

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import Annotated, TypedDict

from langchain.chat_models import init_chat_model


def setup_langsmith(
    project_name: str = "langchain-deep-agent",
    tracing_enabled: bool = True,
) -> None:
    if tracing_enabled:
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", project_name)


@tool
def calculator_tool(expression: str) -> str:
    """Evaluate a mathematical expression. Use this for any calculations."""
    try:
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression: {e}"


@tool
def search_tool(query: str) -> str:
    """Search for information on a topic. Use this when you need to look up facts."""
    return f"Search results for '{query}': This is a simulated search result. In production, connect to a real search API."


@tool
def get_current_time_tool() -> str:
    """Get the current date and time."""
    now = datetime.now()
    return f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}"


DEFAULT_TOOLS = [calculator_tool, search_tool, get_current_time_tool]

DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant with access to various tools.

You can:
- Perform calculations using the calculator tool
- Search for information using the search tool
- Get the current time using the get_current_time tool

Think step by step when solving complex problems. Use tools when needed to provide accurate answers.
Always explain your reasoning and provide clear, helpful responses."""


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    iteration_count: int


@dataclass
class DeepAgentConfig:
    model_name: str = "openai:gpt-4o-mini"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_iterations: int = 10
    temperature: float = 0.0
    langsmith_project: str = "langchain-deep-agent"
    langsmith_tracing: bool = True


def create_deep_agent(
    config: DeepAgentConfig,
    additional_tools: Sequence[BaseTool | Callable] | None = None,
    include_default_tools: bool = True,
) -> StateGraph:
    setup_langsmith(
        project_name=config.langsmith_project,
        tracing_enabled=config.langsmith_tracing,
    )

    model = init_chat_model(config.model_name, temperature=config.temperature)

    tools: list[BaseTool | Callable] = []
    if include_default_tools:
        tools.extend(DEFAULT_TOOLS)
    if additional_tools:
        tools.extend(additional_tools)

    if tools:
        model_with_tools = model.bind_tools(tools)
    else:
        model_with_tools = model

    tool_node = ToolNode(tools) if tools else None

    def should_continue(state: AgentState) -> str:
        messages = state["messages"]
        iteration_count = state.get("iteration_count", 0)

        if iteration_count >= config.max_iterations:
            return "end"

        last_message = messages[-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return "end"

    def call_model(state: AgentState) -> dict[str, Any]:
        messages = state["messages"]
        iteration_count = state.get("iteration_count", 0)

        system_message = SystemMessage(content=config.system_prompt)
        response = model_with_tools.invoke([system_message] + messages)

        return {
            "messages": [response],
            "iteration_count": iteration_count + 1,
        }

    graph = StateGraph(AgentState)

    graph.add_node("agent", call_model)

    if tool_node:
        graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")

    if tool_node:
        graph.add_conditional_edges(
            "agent",
            should_continue,
            {"tools": "tools", "end": END},
        )
        graph.add_edge("tools", "agent")
    else:
        graph.add_edge("agent", END)

    return graph.compile()


def run_agent(agent, user_input: str, stream: bool = False) -> str | None:
    initial_state = {
        "messages": [{"role": "user", "content": user_input}],
        "iteration_count": 0,
    }

    if stream:
        for chunk in agent.stream(initial_state, stream_mode="updates"):
            print(chunk)
            print("---")
        return None
    else:
        result = agent.invoke(initial_state)
        final_message = result["messages"][-1]
        return final_message.content if hasattr(final_message, "content") else str(final_message)
