from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import Annotated, TypedDict

from langchain.chat_models import init_chat_model
from agent import calculator_tool, search_tool, get_current_time_tool, setup_langsmith


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_REFLECTION = "needs_reflection"


class AgentRole(Enum):
    SUPERVISOR = "supervisor"
    PLANNER = "planner"
    RESEARCHER = "researcher"
    EXECUTOR = "executor"
    REFLECTOR = "reflector"


@dataclass
class Task:
    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: AgentRole | None = None
    dependencies: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    result: str | None = None
    error: str | None = None


@dataclass
class MemoryEntry:
    agent_role: AgentRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    tags: list[str] = field(default_factory=list)


class DeepAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    task: str
    plan: list[str]
    task_hierarchy: list[Task]
    current_step: int
    completed_tasks: list[str]
    sub_agent_results: dict[str, Any]
    reflection: str
    memory: list[MemoryEntry]
    iteration: int
    next_agent: str | None
    final_answer: str | None


@dataclass
class DeepAgentConfig:
    model_name: str = "ollama:gpt-oss"
    supervisor_model: str = "ollama:gpt-oss"
    planner_model: str = "ollama:gpt-oss"
    researcher_model: str = "ollama:gpt-oss"
    executor_model: str = "ollama:gpt-oss"
    reflector_model: str = "ollama:gpt-oss"
    system_prompt: str = "You are a helpful AI assistant with deep reasoning capabilities."
    max_iterations: int = 20
    temperature: float = 0.0
    langsmith_project: str = "langchain-deep-agent"
    langsmith_tracing: bool = True


def supervisor_node(state: DeepAgentState, config: DeepAgentConfig) -> dict[str, Any]:
    messages = state.get("messages", [])
    iteration = state.get("iteration", 0)
    plan = state.get("plan", [])
    sub_agent_results = state.get("sub_agent_results", {})
    reflection = state.get("reflection", "")
    final_answer = state.get("final_answer")

    if iteration >= config.max_iterations:
        return {
            "iteration": iteration,
            "next_agent": "END",
            "messages": [AIMessage(content="Max iterations reached. Providing best answer.")],
        }

    if final_answer:
        return {
            "iteration": iteration,
            "next_agent": "END",
        }

    model = init_chat_model(config.supervisor_model, temperature=config.temperature)

    execution_results = sub_agent_results.get("execution_results", {})
    research_findings = sub_agent_results.get("research_findings", {})

    supervisor_prompt = f"""You are the supervisor of a multi-agent system. Analyze the current state and decide which agent should act next.

Current State:
- Iteration: {iteration}/{config.max_iterations}
- Plan exists: {bool(plan)}
- Plan steps: {plan if plan else "None"}
- Research findings: {list(research_findings.keys()) if research_findings else "None"}
- Execution results: {list(execution_results.keys()) if execution_results else "None"}
- Latest Reflection: {reflection if reflection else "None"}

Available Agents:
- PLANNER: Creates step-by-step plans for complex tasks
- RESEARCHER: Gathers information using search tools
- EXECUTOR: Executes actions using calculator and other tools
- REFLECTOR: Evaluates progress and synthesizes final answer
- END: Task is complete

Routing Rules:
1. If no plan exists, route to PLANNER
2. If plan exists but needs research, route to RESEARCHER
3. If research is done and actions needed, route to EXECUTOR
4. If execution done, route to REFLECTOR for evaluation
5. If reflection shows task complete, route to END

Respond with ONLY the agent name: PLANNER, RESEARCHER, EXECUTOR, REFLECTOR, or END"""

    response = model.invoke([
        SystemMessage(content=supervisor_prompt),
        HumanMessage(content=f"Original task: {state.get('task', 'Unknown')}")
    ])

    decision = response.content.strip().upper()
    valid_agents = ["PLANNER", "RESEARCHER", "EXECUTOR", "REFLECTOR", "END"]

    for agent in valid_agents:
        if agent in decision:
            decision = agent
            break
    else:
        if not plan:
            decision = "PLANNER"
        elif not research_findings and not execution_results:
            decision = "RESEARCHER"
        elif not reflection:
            decision = "EXECUTOR"
        else:
            decision = "REFLECTOR"

    return {
        "iteration": iteration + 1,
        "next_agent": decision,
        "messages": [AIMessage(content=f"[Supervisor] Routing to {decision}")],
    }


def planner_node(state: DeepAgentState, config: DeepAgentConfig) -> dict[str, Any]:
    task = state.get("task", "")

    model = init_chat_model(config.planner_model, temperature=0.2)

    system_prompt = """You are a task planning expert. Break down the user's task into clear, actionable steps.

Output a numbered list of steps. Each step should be:
- Specific and actionable
- Either a RESEARCH step (gathering info) or EXECUTE step (performing action)

Example format:
1. RESEARCH: Find the formula for compound interest
2. EXECUTE: Calculate the final amount using the formula
3. EXECUTE: Calculate the interest earned
4. RESEARCH: Verify the calculation method"""

    response = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Break down this task:\n{task}")
    ])

    plan_text = response.content
    lines = plan_text.strip().split("\n")
    plan = []
    task_hierarchy = []

    for i, line in enumerate(lines):
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith("-")):
            clean_line = line.lstrip("0123456789.-) ").strip()
            if clean_line:
                plan.append(clean_line)

                assigned_to = AgentRole.RESEARCHER if "RESEARCH" in line.upper() else AgentRole.EXECUTOR
                task_obj = Task(
                    id=f"task-{i+1}",
                    description=clean_line,
                    status=TaskStatus.PENDING,
                    assigned_to=assigned_to,
                )
                task_hierarchy.append(task_obj)

    if not plan:
        plan = [task]
        task_hierarchy = [Task(
            id="task-1",
            description=task,
            status=TaskStatus.PENDING,
            assigned_to=AgentRole.EXECUTOR,
        )]

    return {
        "plan": plan,
        "task_hierarchy": task_hierarchy,
        "current_step": 0,
        "messages": [AIMessage(content=f"[Planner] Created plan with {len(plan)} steps:\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(plan)))],
    }


def researcher_node(state: DeepAgentState, config: DeepAgentConfig) -> dict[str, Any]:
    messages = state.get("messages", [])
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    sub_agent_results = state.get("sub_agent_results", {})
    task = state.get("task", "")

    research_findings = sub_agent_results.get("research_findings", {})

    research_tasks = [
        (i, step) for i, step in enumerate(plan)
        if "RESEARCH" in step.upper() or "search" in step.lower() or "find" in step.lower()
    ]

    pending_research = [
        (i, step) for i, step in research_tasks
        if step not in research_findings
    ]

    if not pending_research:
        return {
            "messages": [AIMessage(content="[Researcher] No research tasks pending.")],
            "sub_agent_results": {**sub_agent_results, "research_findings": research_findings},
        }

    idx, current_task = pending_research[0]

    model = init_chat_model(config.researcher_model, temperature=0.0)
    tools = [search_tool, get_current_time_tool]
    model_with_tools = model.bind_tools(tools)

    research_prompt = f"""You are a research agent. Your current task: {current_task}

Original user query: {task}

Use the search_tool to gather relevant information. Provide clear, factual findings."""

    response = model_with_tools.invoke([
        SystemMessage(content=research_prompt),
        *messages[-5:]
    ])

    if response.tool_calls:
        return {"messages": [response]}

    research_findings[current_task] = response.content

    return {
        "messages": [AIMessage(content=f"[Researcher] Completed: {current_task}\nFindings: {response.content[:200]}...")],
        "sub_agent_results": {
            **sub_agent_results,
            "research_findings": research_findings,
        },
    }


def executor_node(state: DeepAgentState, config: DeepAgentConfig) -> dict[str, Any]:
    messages = state.get("messages", [])
    plan = state.get("plan", [])
    sub_agent_results = state.get("sub_agent_results", {})
    task = state.get("task", "")

    execution_results = sub_agent_results.get("execution_results", {})
    research_findings = sub_agent_results.get("research_findings", {})

    execute_tasks = [
        (i, step) for i, step in enumerate(plan)
        if "EXECUTE" in step.upper() or "calculate" in step.lower() or "compute" in step.lower()
    ]

    pending_execution = [
        (i, step) for i, step in execute_tasks
        if step not in execution_results
    ]

    if not pending_execution:
        all_tasks = [(i, step) for i, step in enumerate(plan) if step not in execution_results and step not in research_findings]
        if all_tasks:
            pending_execution = [all_tasks[0]]
        else:
            return {
                "messages": [AIMessage(content="[Executor] No execution tasks pending.")],
                "sub_agent_results": {**sub_agent_results, "execution_results": execution_results},
            }

    idx, current_task = pending_execution[0]

    model = init_chat_model(config.executor_model, temperature=0.0)
    tools = [calculator_tool, get_current_time_tool]
    model_with_tools = model.bind_tools(tools)

    context = ""
    if research_findings:
        context = "\n\nResearch findings:\n" + "\n".join(f"- {k}: {v}" for k, v in research_findings.items())

    execute_prompt = f"""You are an execution agent. Your current task: {current_task}

Original user query: {task}
{context}

Use calculator_tool for any mathematical computations. Be precise and show your work."""

    response = model_with_tools.invoke([
        SystemMessage(content=execute_prompt),
        *messages[-5:]
    ])

    if response.tool_calls:
        return {"messages": [response]}

    execution_results[current_task] = response.content

    return {
        "messages": [AIMessage(content=f"[Executor] Completed: {current_task}\nResult: {response.content[:200]}...")],
        "sub_agent_results": {
            **sub_agent_results,
            "execution_results": execution_results,
        },
    }


def reflector_node(state: DeepAgentState, config: DeepAgentConfig) -> dict[str, Any]:
    messages = state.get("messages", [])
    plan = state.get("plan", [])
    sub_agent_results = state.get("sub_agent_results", {})
    task = state.get("task", "")
    memory = state.get("memory", [])

    execution_results = sub_agent_results.get("execution_results", {})
    research_findings = sub_agent_results.get("research_findings", {})

    model = init_chat_model(config.reflector_model, temperature=0.3)

    all_results = {**research_findings, **execution_results}

    reflect_prompt = f"""You are a reflection agent. Evaluate the work done and synthesize a final answer.

Original Task: {task}

Plan:
{chr(10).join(f'{i+1}. {s}' for i, s in enumerate(plan))}

Results:
{chr(10).join(f'- {k}: {v}' for k, v in all_results.items())}

Evaluate:
1. Was the task fully addressed?
2. Are the results accurate and complete?
3. What is the final answer to give the user?

If the task is complete, provide a clear FINAL ANSWER.
If more work is needed, explain what's missing."""

    response = model.invoke([
        SystemMessage(content=reflect_prompt),
        *messages[-3:]
    ])

    reflection_content = response.content

    is_complete = "FINAL ANSWER" in reflection_content.upper() or "complete" in reflection_content.lower()

    final_answer = None
    if is_complete:
        if "FINAL ANSWER" in reflection_content.upper():
            parts = reflection_content.upper().split("FINAL ANSWER")
            if len(parts) > 1:
                final_answer = reflection_content[reflection_content.upper().find("FINAL ANSWER"):]
        else:
            final_answer = reflection_content

        memory_entry = MemoryEntry(
            agent_role=AgentRole.REFLECTOR,
            content=f"Task: {task} | Answer: {final_answer[:100]}",
            tags=["completed", "success"],
        )
        memory = memory + [memory_entry]

    return {
        "reflection": reflection_content,
        "final_answer": final_answer,
        "memory": memory,
        "messages": [AIMessage(content=f"[Reflector] {reflection_content[:300]}...")],
    }


def route_from_supervisor(state: DeepAgentState) -> Literal["planner", "researcher", "executor", "reflector", "__end__"]:
    next_agent = state.get("next_agent", "END")

    routing_map = {
        "PLANNER": "planner",
        "RESEARCHER": "researcher",
        "EXECUTOR": "executor",
        "REFLECTOR": "reflector",
        "END": "__end__",
    }

    return routing_map.get(next_agent, "__end__")


def route_after_tools(state: DeepAgentState) -> Literal["researcher", "executor", "supervisor"]:
    messages = state.get("messages", [])

    for msg in reversed(messages[-5:]):
        if isinstance(msg, AIMessage):
            content = str(msg.content) if msg.content else ""
            if "[Researcher]" in content:
                return "researcher"
            if "[Executor]" in content:
                return "executor"

    return "supervisor"


def create_deep_agent(config: DeepAgentConfig | None = None) -> StateGraph:
    if config is None:
        config = DeepAgentConfig()

    setup_langsmith(
        project_name=config.langsmith_project,
        tracing_enabled=config.langsmith_tracing,
    )

    graph = StateGraph(DeepAgentState)

    graph.add_node("supervisor", lambda state: supervisor_node(state, config))
    graph.add_node("planner", lambda state: planner_node(state, config))
    graph.add_node("researcher", lambda state: researcher_node(state, config))
    graph.add_node("executor", lambda state: executor_node(state, config))
    graph.add_node("reflector", lambda state: reflector_node(state, config))

    tools = [calculator_tool, search_tool, get_current_time_tool]
    tool_node = ToolNode(tools)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "planner": "planner",
            "researcher": "researcher",
            "executor": "executor",
            "reflector": "reflector",
            "__end__": END,
        }
    )

    graph.add_edge("planner", "supervisor")

    def researcher_route(state):
        messages = state.get("messages", [])
        if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
            return "tools"
        return "supervisor"

    graph.add_conditional_edges(
        "researcher",
        researcher_route,
        {"tools": "tools", "supervisor": "supervisor"}
    )

    def executor_route(state):
        messages = state.get("messages", [])
        if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
            return "tools"
        return "supervisor"

    graph.add_conditional_edges(
        "executor",
        executor_route,
        {"tools": "tools", "supervisor": "supervisor"}
    )

    graph.add_edge("reflector", "supervisor")

    graph.add_conditional_edges(
        "tools",
        route_after_tools,
        {"researcher": "researcher", "executor": "executor", "supervisor": "supervisor"}
    )

    return graph.compile()


def run_deep_agent(query: str, config: DeepAgentConfig | None = None, stream: bool = False) -> dict[str, Any]:
    if config is None:
        config = DeepAgentConfig()

    agent = create_deep_agent(config)

    initial_state: DeepAgentState = {
        "messages": [HumanMessage(content=query)],
        "task": query,
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
    }

    if stream:
        for chunk in agent.stream(initial_state, stream_mode="updates"):
            print(chunk)
            print("---")
        return {}

    final_state = agent.invoke(initial_state)

    return {
        "answer": final_state.get("final_answer") or final_state.get("reflection"),
        "plan": final_state.get("plan"),
        "iterations": final_state.get("iteration"),
        "results": final_state.get("sub_agent_results"),
        "memory": final_state.get("memory"),
    }


if __name__ == "__main__":
    config = DeepAgentConfig(
        model_name="ollama:gpt-oss",
        max_iterations=15,
    )

    result = run_deep_agent(
        "Calculate the compound interest on $1000 at 5% for 3 years. The formula is P * (1 + r)^t",
        config=config,
    )

    print("=" * 60)
    print("FINAL ANSWER:")
    print("=" * 60)
    print(result.get("answer"))
    print()
    print(f"Iterations: {result.get('iterations')}")
    print(f"Plan: {result.get('plan')}")
