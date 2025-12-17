#!/usr/bin/env python3
from dotenv import load_dotenv

load_dotenv()

import chainlit as cl
from deep_agent import DeepAgentConfig, create_deep_agent, DeepAgentState
from langchain_core.messages import HumanMessage, AIMessage


@cl.on_chat_start
async def start():
    config = DeepAgentConfig(
        model_name="ollama:gpt-oss",
        supervisor_model="ollama:gpt-oss",
        planner_model="ollama:gpt-oss",
        researcher_model="ollama:gpt-oss",
        executor_model="ollama:gpt-oss",
        reflector_model="ollama:gpt-oss",
        max_iterations=15,
        temperature=0.0,
    )

    agent = create_deep_agent(config)
    cl.user_session.set("agent", agent)

    await cl.Message(
        content="Hello! I'm a **Deep Agent** with hierarchical planning, specialized sub-agents, and self-reflection capabilities.\n\nI can:\n- Break down complex tasks (Planner)\n- Research information (Researcher)\n- Execute calculations (Executor)\n- Evaluate and improve my work (Reflector)\n\nWhat would you like me to help with?"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")

    msg = cl.Message(content="")
    await msg.send()

    initial_state: DeepAgentState = {
        "messages": [HumanMessage(content=message.content)],
        "task": message.content,
        "plan": [],
        "task_hierarchy": [],
        "current_step": 0,
        "completed_tasks": [],
        "sub_agent_results": {},
        "reflection": "",
        "memory": [],
        "iteration": 0,
        "next_agent": None,
        "final_answer": None,
        "errors": [],
        "retry_count": 0,
    }

    plan_shown = False

    for chunk in agent.stream(initial_state, stream_mode="updates"):
        for node_name, node_output in chunk.items():
            if node_name == "supervisor":
                next_agent = node_output.get("next_agent", "")
                if next_agent and next_agent != "END":
                    async with cl.Step(name=f"Routing to {next_agent}") as step:
                        step.output = f"Supervisor delegating to {next_agent} agent"

            elif node_name == "planner":
                plan = node_output.get("plan", [])
                if plan and not plan_shown:
                    plan_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
                    async with cl.Step(name="Planner") as step:
                        step.output = f"**Task Breakdown:**\n{plan_text}"
                    plan_shown = True

            elif node_name == "researcher":
                results = node_output.get("sub_agent_results", {})
                findings = results.get("research_findings", {})
                if findings:
                    async with cl.Step(name="Researcher") as step:
                        step.output = "\n".join(f"**{k}:**\n{v[:300]}..." for k, v in findings.items())

            elif node_name == "executor":
                results = node_output.get("sub_agent_results", {})
                exec_results = results.get("execution_results", {})
                if exec_results:
                    async with cl.Step(name="Executor") as step:
                        step.output = "\n".join(f"**{k}:**\n{v[:300]}..." for k, v in exec_results.items())

            elif node_name == "reflector":
                reflection = node_output.get("reflection", "")
                final_answer = node_output.get("final_answer")

                if reflection:
                    async with cl.Step(name="Reflector") as step:
                        step.output = reflection[:500]

                if final_answer:
                    msg.content = final_answer
                    await msg.update()

            elif node_name == "tools":
                messages = node_output.get("messages", [])
                for tool_msg in messages:
                    if hasattr(tool_msg, "name") and hasattr(tool_msg, "content"):
                        async with cl.Step(name=f"Tool: {tool_msg.name}") as step:
                            step.output = str(tool_msg.content)[:500]

    if not msg.content:
        msg.content = "Task completed. Check the steps above for details."
        await msg.update()
