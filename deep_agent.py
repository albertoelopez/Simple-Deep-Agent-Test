from __future__ import annotations

import json
import logging
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("deep_agent")


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
    errors: list[str]
    retry_count: int
    tool_attempts: dict[str, int]


@dataclass
class DeepAgentConfig:
    model_name: str = "ollama:gpt-oss"
    supervisor_model: str = "ollama:gpt-oss"
    planner_model: str = "ollama:gpt-oss"
    researcher_model: str = "ollama:gpt-oss"
    executor_model: str = "ollama:gpt-oss"
    reflector_model: str = "ollama:gpt-oss"
    system_prompt: str = "You are a helpful AI assistant with deep reasoning capabilities."
    max_iterations: int = 15
    max_retries: int = 2
    max_tool_attempts: int = 3
    recursion_limit: int = 50
    timeout_seconds: int = 60
    temperature: float = 0.0
    langsmith_project: str = "langchain-deep-agent"
    langsmith_tracing: bool = True
    fallback_response: str = "I apologize, but I encountered an issue processing your request. Please try rephrasing your question or breaking it into smaller parts."


def supervisor_node(state: DeepAgentState, config: DeepAgentConfig) -> dict[str, Any]:
    messages = state.get("messages", [])
    iteration = state.get("iteration", 0)
    plan = state.get("plan", [])
    sub_agent_results = state.get("sub_agent_results", {})
    reflection = state.get("reflection", "")
    final_answer = state.get("final_answer")
    errors = state.get("errors", [])
    retry_count = state.get("retry_count", 0)

    logger.info(f"Supervisor: iteration={iteration}, plan_steps={len(plan)}, errors={len(errors)}")

    if errors and retry_count >= config.max_retries:
        logger.warning(f"Max retries ({config.max_retries}) reached with errors: {errors[-1]}")
        return {
            "iteration": iteration,
            "next_agent": "END",
            "final_answer": f"{config.fallback_response}\n\nError details: {errors[-1]}",
            "messages": [AIMessage(content=f"[Supervisor] Max retries reached. Providing fallback response.")],
        }

    if iteration >= config.max_iterations:
        logger.warning(f"Max iterations ({config.max_iterations}) reached")
        return {
            "iteration": iteration,
            "next_agent": "END",
            "messages": [AIMessage(content="Max iterations reached. Providing best answer.")],
        }

    if final_answer:
        logger.info("Task complete - final answer available")
        return {
            "iteration": iteration,
            "next_agent": "END",
        }

    try:
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
- Recent Errors: {errors[-3:] if errors else "None"}

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
6. If there are errors, consider routing to REFLECTOR for graceful completion

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

        logger.info(f"Supervisor decision: {decision}")

        return {
            "iteration": iteration + 1,
            "next_agent": decision,
            "messages": [AIMessage(content=f"[Supervisor] Routing to {decision}")],
        }

    except Exception as e:
        error_msg = f"Supervisor error: {str(e)}"
        logger.error(error_msg)
        errors = errors + [error_msg]

        if retry_count < config.max_retries:
            return {
                "iteration": iteration + 1,
                "retry_count": retry_count + 1,
                "errors": errors,
                "next_agent": "REFLECTOR",
                "messages": [AIMessage(content=f"[Supervisor] Error occurred, routing to reflector for graceful completion.")],
            }

        return {
            "iteration": iteration,
            "next_agent": "END",
            "errors": errors,
            "final_answer": config.fallback_response,
            "messages": [AIMessage(content=f"[Supervisor] Critical error. {config.fallback_response}")],
        }


def planner_node(state: DeepAgentState, config: DeepAgentConfig) -> dict[str, Any]:
    task = state.get("task", "")
    errors = state.get("errors", [])

    logger.info(f"Planner: processing task '{task[:50]}...'")

    try:
        model = init_chat_model(config.planner_model, temperature=0.2)

        system_prompt = """You are a task planning expert. Break down the user's task into clear, actionable steps.

Output a numbered list of steps. Each step should be:
- Specific and actionable
- Either a RESEARCH step (gathering info) or EXECUTE step (performing action)
- Keep steps simple and achievable

Example format:
1. RESEARCH: Find the formula for compound interest
2. EXECUTE: Calculate the final amount using the formula
3. EXECUTE: Calculate the interest earned
4. RESEARCH: Verify the calculation method

IMPORTANT: Keep the plan concise (3-5 steps max). Simple questions may only need 1-2 steps."""

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
            plan = [f"EXECUTE: {task}"]
            task_hierarchy = [Task(
                id="task-1",
                description=task,
                status=TaskStatus.PENDING,
                assigned_to=AgentRole.EXECUTOR,
            )]

        logger.info(f"Planner: created {len(plan)} steps")

        return {
            "plan": plan,
            "task_hierarchy": task_hierarchy,
            "current_step": 0,
            "messages": [AIMessage(content=f"[Planner] Created plan with {len(plan)} steps:\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(plan)))],
        }

    except Exception as e:
        error_msg = f"Planner error: {str(e)}"
        logger.error(error_msg)

        fallback_plan = [f"EXECUTE: {task}"]
        return {
            "plan": fallback_plan,
            "task_hierarchy": [Task(
                id="task-1",
                description=task,
                status=TaskStatus.PENDING,
                assigned_to=AgentRole.EXECUTOR,
            )],
            "current_step": 0,
            "errors": errors + [error_msg],
            "messages": [AIMessage(content=f"[Planner] Using simplified plan due to error. Will attempt direct execution.")],
        }


def researcher_node(state: DeepAgentState, config: DeepAgentConfig) -> dict[str, Any]:
    messages = state.get("messages", [])
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    sub_agent_results = state.get("sub_agent_results", {})
    task = state.get("task", "")
    errors = state.get("errors", [])
    tool_attempts = state.get("tool_attempts", {})

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
        logger.info("Researcher: no pending research tasks")
        return {
            "messages": [AIMessage(content="[Researcher] No research tasks pending.")],
            "sub_agent_results": {**sub_agent_results, "research_findings": research_findings},
        }

    idx, current_task = pending_research[0]
    task_key = f"research:{current_task[:50]}"
    attempts = tool_attempts.get(task_key, 0)

    if attempts >= config.max_tool_attempts:
        logger.warning(f"Researcher: max tool attempts ({config.max_tool_attempts}) reached for task")
        last_tool_result = None
        for msg in reversed(messages[-10:]):
            if hasattr(msg, 'content') and msg.content and 'Search results' in str(msg.content):
                last_tool_result = str(msg.content)
                break

        research_findings[current_task] = last_tool_result or f"Research attempted but no definitive results found for: {current_task}. Based on available information, I cannot provide specific details about this query."

        return {
            "messages": [AIMessage(content=f"[Researcher] Completed research (max attempts): {current_task[:50]}...")],
            "sub_agent_results": {
                **sub_agent_results,
                "research_findings": research_findings,
            },
            "tool_attempts": {**tool_attempts, task_key: 0},
        }

    logger.info(f"Researcher: working on '{current_task[:50]}...' (attempt {attempts + 1}/{config.max_tool_attempts})")

    try:
        model = init_chat_model(config.researcher_model, temperature=0.0)
        tools = [search_tool, get_current_time_tool]
        model_with_tools = model.bind_tools(tools)

        research_prompt = f"""You are a research agent. Your current task: {current_task}

Original user query: {task}

IMPORTANT: You have made {attempts} previous tool attempts. If search results are insufficient, provide your best answer based on your knowledge rather than searching again.

Use the search_tool to gather relevant information. If search_tool is unavailable or fails, provide the best answer you can from your knowledge.

Provide clear, factual findings. If you cannot find the information, say so explicitly and provide what you do know."""

        response = model_with_tools.invoke([
            SystemMessage(content=research_prompt),
            *messages[-5:]
        ])

        if response.tool_calls:
            logger.info(f"Researcher: invoking tools {[tc['name'] for tc in response.tool_calls]} (attempt {attempts + 1})")
            return {
                "messages": [response],
                "tool_attempts": {**tool_attempts, task_key: attempts + 1},
            }

        research_findings[current_task] = response.content
        logger.info(f"Researcher: completed task without tools")

        return {
            "messages": [AIMessage(content=f"[Researcher] Completed: {current_task}\nFindings: {response.content[:200]}...")],
            "sub_agent_results": {
                **sub_agent_results,
                "research_findings": research_findings,
            },
            "tool_attempts": {**tool_attempts, task_key: 0},
        }

    except Exception as e:
        error_msg = f"Researcher error on '{current_task[:30]}': {str(e)}"
        logger.error(error_msg)

        research_findings[current_task] = f"Research unavailable: {str(e)[:100]}. Proceeding with available information."

        return {
            "messages": [AIMessage(content=f"[Researcher] Could not complete research for: {current_task}. Continuing with available data.")],
            "sub_agent_results": {
                **sub_agent_results,
                "research_findings": research_findings,
            },
            "errors": errors + [error_msg],
            "tool_attempts": {**tool_attempts, task_key: 0},
        }


def executor_node(state: DeepAgentState, config: DeepAgentConfig) -> dict[str, Any]:
    messages = state.get("messages", [])
    plan = state.get("plan", [])
    sub_agent_results = state.get("sub_agent_results", {})
    task = state.get("task", "")
    errors = state.get("errors", [])

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
            logger.info("Executor: no pending execution tasks")
            return {
                "messages": [AIMessage(content="[Executor] No execution tasks pending.")],
                "sub_agent_results": {**sub_agent_results, "execution_results": execution_results},
            }

    idx, current_task = pending_execution[0]
    logger.info(f"Executor: working on '{current_task[:50]}...'")

    try:
        model = init_chat_model(config.executor_model, temperature=0.0)
        tools = [calculator_tool, get_current_time_tool]
        model_with_tools = model.bind_tools(tools)

        context = ""
        if research_findings:
            context = "\n\nResearch findings:\n" + "\n".join(f"- {k}: {v[:200]}" for k, v in research_findings.items())

        execute_prompt = f"""You are an execution agent. Your current task: {current_task}

Original user query: {task}
{context}

Use calculator_tool for any mathematical computations. If tools are unavailable, provide the best answer you can.
Be precise and show your work."""

        response = model_with_tools.invoke([
            SystemMessage(content=execute_prompt),
            *messages[-5:]
        ])

        if response.tool_calls:
            logger.info(f"Executor: invoking tools {[tc['name'] for tc in response.tool_calls]}")
            return {"messages": [response]}

        execution_results[current_task] = response.content
        logger.info(f"Executor: completed task without tools")

        return {
            "messages": [AIMessage(content=f"[Executor] Completed: {current_task}\nResult: {response.content[:200]}...")],
            "sub_agent_results": {
                **sub_agent_results,
                "execution_results": execution_results,
            },
        }

    except Exception as e:
        error_msg = f"Executor error on '{current_task[:30]}': {str(e)}"
        logger.error(error_msg)

        execution_results[current_task] = f"Execution failed: {str(e)[:100]}. Manual review recommended."

        return {
            "messages": [AIMessage(content=f"[Executor] Could not complete execution for: {current_task}. Continuing with available results.")],
            "sub_agent_results": {
                **sub_agent_results,
                "execution_results": execution_results,
            },
            "errors": errors + [error_msg],
        }


def reflector_node(state: DeepAgentState, config: DeepAgentConfig) -> dict[str, Any]:
    messages = state.get("messages", [])
    plan = state.get("plan", [])
    sub_agent_results = state.get("sub_agent_results", {})
    task = state.get("task", "")
    memory = state.get("memory", [])
    errors = state.get("errors", [])

    execution_results = sub_agent_results.get("execution_results", {})
    research_findings = sub_agent_results.get("research_findings", {})

    logger.info(f"Reflector: evaluating task with {len(research_findings)} research findings, {len(execution_results)} execution results, {len(errors)} errors")

    try:
        model = init_chat_model(config.reflector_model, temperature=0.3)

        all_results = {**research_findings, **execution_results}

        error_context = ""
        if errors:
            error_context = f"\n\nErrors encountered during processing:\n" + "\n".join(f"- {e}" for e in errors[-3:])

        reflect_prompt = f"""You are a reflection agent. Evaluate the work done and synthesize a final answer.

Original Task: {task}

Plan:
{chr(10).join(f'{i+1}. {s}' for i, s in enumerate(plan)) if plan else "No plan created"}

Results:
{chr(10).join(f'- {k}: {str(v)[:300]}' for k, v in all_results.items()) if all_results else "No results gathered"}
{error_context}

IMPORTANT: You MUST provide a response to the user. Even if results are incomplete, synthesize the best answer possible.

Evaluate:
1. Was the task fully addressed?
2. Are the results accurate and complete?
3. What is the final answer to give the user?

ALWAYS provide a FINAL ANSWER section, even if incomplete. Format:
FINAL ANSWER: [Your synthesized response to the user]"""

        response = model.invoke([
            SystemMessage(content=reflect_prompt),
            *messages[-3:]
        ])

        reflection_content = response.content

        final_answer = None
        if "FINAL ANSWER" in reflection_content.upper():
            idx = reflection_content.upper().find("FINAL ANSWER")
            final_answer = reflection_content[idx:].replace("FINAL ANSWER:", "").replace("FINAL ANSWER", "").strip()
        elif all_results or errors:
            final_answer = reflection_content
        else:
            final_answer = f"I processed your request: '{task}'. {reflection_content}"

        memory_entry = MemoryEntry(
            agent_role=AgentRole.REFLECTOR,
            content=f"Task: {task[:50]} | Answer: {str(final_answer)[:100]}",
            tags=["completed", "success" if not errors else "with_errors"],
        )
        memory = memory + [memory_entry]

        logger.info(f"Reflector: completed with final_answer={'yes' if final_answer else 'no'}")

        return {
            "reflection": reflection_content,
            "final_answer": final_answer,
            "memory": memory,
            "messages": [AIMessage(content=f"[Reflector] {reflection_content[:300]}...")],
        }

    except Exception as e:
        error_msg = f"Reflector error: {str(e)}"
        logger.error(error_msg)

        fallback_answer = f"I attempted to process your request: '{task}'. "
        if research_findings:
            fallback_answer += f"I found some information: {list(research_findings.values())[0][:200]}..."
        elif execution_results:
            fallback_answer += f"Here are the results: {list(execution_results.values())[0][:200]}..."
        else:
            fallback_answer += config.fallback_response

        return {
            "reflection": f"Error during reflection: {str(e)}",
            "final_answer": fallback_answer,
            "memory": memory,
            "errors": errors + [error_msg],
            "messages": [AIMessage(content=f"[Reflector] Providing best available answer due to processing issue.")],
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

    return graph.compile(recursion_limit=config.recursion_limit)


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
        "errors": [],
        "retry_count": 0,
        "tool_attempts": {},
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
