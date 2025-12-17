---
name: deep-agent-dev
description: Orchestrate parallel development of the LangChain deep agent. Coordinates langchain-architect, tool-builder, graph-builder, test-writer, and ui-integrator agents to work simultaneously on different aspects.
---

# Deep Agent Development Skill

## Purpose
Coordinate parallel development of the LangChain deep agent using specialized sub-agents.

## Project Structure
```
langchain_deep_agent/
├── agent.py           # Main agent (upgrade to deep agent)
├── deep_agent.py      # New deep agent implementation
├── chainlit_app.py    # UI integration
├── example.py         # Usage examples
├── tests/
│   └── test_agent.py  # Tests
└── .claude/
    └── agents/        # Project-specific agents
```

## Available Agents

| Agent | Purpose | Model |
|-------|---------|-------|
| `langchain-architect` | Design architecture | sonnet |
| `graph-builder` | Implement StateGraphs | sonnet |
| `tool-builder` | Create/improve tools | haiku |
| `test-writer` | Write tests | haiku |
| `ui-integrator` | Chainlit UI | haiku |

## Parallel Development Pattern

### Phase 1: Architecture (Sequential)
```
langchain-architect → produces architecture spec
```

### Phase 2: Implementation (Parallel)
```
graph-builder ──┐
tool-builder ───┼→ all working simultaneously
test-writer ────┘
```

### Phase 3: Integration (Sequential)
```
ui-integrator → integrates everything
```

## How to Use

### Launch Parallel Tasks
```
Use Task tool with multiple agents:
- Task(agent="graph-builder", prompt="Implement supervisor node")
- Task(agent="tool-builder", prompt="Add planning tool")
- Task(agent="test-writer", prompt="Write supervisor tests")
```

### Coordinate Results
1. Collect outputs from parallel agents
2. Resolve any conflicts
3. Merge into codebase
4. Run tests to verify

## Target Architecture

```
                    ┌─────────────┐
                    │  Supervisor │
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        ↓                  ↓                  ↓
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│   Planner     │  │  Researcher   │  │   Executor    │
│  Sub-Agent    │  │   Sub-Agent   │  │   Sub-Agent   │
└───────────────┘  └───────────────┘  └───────────────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ↓
                    ┌─────────────┐
                    │  Reflector  │
                    └─────────────┘
```

## State Schema
```python
class DeepAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    task: str
    plan: list[str]
    current_step: int
    sub_agent_results: dict
    reflection: str
    iteration: int
```
