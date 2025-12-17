#!/usr/bin/env python3
import os
from dotenv import load_dotenv

load_dotenv()

from agent import (
    create_deep_agent,
    DeepAgentConfig,
    run_agent,
)
from langchain_core.tools import tool


@tool
def weather_tool(location: str) -> str:
    """Get the current weather for a location."""
    return f"The weather in {location} is sunny with a temperature of 72°F."


def main():
    config = DeepAgentConfig(
        model_name="ollama:gpt-oss",
        system_prompt="""You are a helpful AI assistant.
        Use tools when needed to provide accurate answers.
        Think step by step when solving problems.""",
        max_iterations=5,
        temperature=0.0,
    )

    agent = create_deep_agent(
        config,
        additional_tools=[weather_tool],
        include_default_tools=True,
    )

    print("=" * 60)
    print("LangChain Deep Agent Example")
    print("=" * 60)

    queries = [
        "What is 15 * 23 + 42?",
        "What time is it right now?",
        "What's the weather like in San Francisco?",
        "Calculate the compound interest on $1000 at 5% for 3 years. The formula is P * (1 + r)^t",
    ]

    for query in queries:
        print(f"\nUser: {query}")
        print("-" * 40)
        response = run_agent(agent, query)
        print(f"Agent: {response}")
        print("=" * 60)


def interactive_mode():
    config = DeepAgentConfig(
        model_name="openai:gpt-4o-mini",
        max_iterations=10,
    )

    agent = create_deep_agent(config)

    print("LangChain Deep Agent - Interactive Mode")
    print("Type 'quit' to exit")
    print("-" * 40)

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if not user_input:
            continue

        response = run_agent(agent, user_input)
        print(f"\nAgent: {response}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        main()
