#!/usr/bin/env python3
from dotenv import load_dotenv

load_dotenv()

import chainlit as cl
from agent import DeepAgentConfig, create_deep_agent


@cl.on_chat_start
async def start():
    config = DeepAgentConfig(
        model_name="ollama:gpt-oss",
        system_prompt="""You are a helpful AI assistant with access to various tools.

You can:
- Perform calculations using the calculator tool
- Search for information using the search tool
- Get the current time using the get_current_time tool

Think step by step when solving complex problems. Use tools when needed to provide accurate answers.
Always explain your reasoning and provide clear, helpful responses.""",
        max_iterations=10,
        temperature=0.0,
    )

    agent = create_deep_agent(config)
    cl.user_session.set("agent", agent)

    await cl.Message(
        content="Hello! I'm your AI assistant. I can help you with calculations, search for information, and tell you the current time. What would you like to know?"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")

    msg = cl.Message(content="")
    await msg.send()

    initial_state = {
        "messages": [{"role": "user", "content": message.content}],
        "iteration_count": 0,
    }

    tool_calls_made = []

    for chunk in agent.stream(initial_state, stream_mode="updates"):
        if "agent" in chunk:
            agent_output = chunk["agent"]
            if "messages" in agent_output:
                for m in agent_output["messages"]:
                    if hasattr(m, "tool_calls") and m.tool_calls:
                        for tc in m.tool_calls:
                            tool_calls_made.append(tc["name"])
                    elif hasattr(m, "content") and m.content:
                        msg.content = m.content
                        await msg.update()

        if "tools" in chunk:
            tools_output = chunk["tools"]
            if "messages" in tools_output:
                for tool_msg in tools_output["messages"]:
                    if hasattr(tool_msg, "content"):
                        step_name = getattr(tool_msg, "name", "tool")
                        async with cl.Step(name=step_name) as step:
                            step.output = tool_msg.content

    if tool_calls_made:
        tools_used = ", ".join(set(tool_calls_made))
        await cl.Message(
            content=f"Tools used: {tools_used}",
            author="system",
        ).send()
