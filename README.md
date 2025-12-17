# LangChain Deep Agent

A simple LangChain deep agent with tool support, LangSmith tracing, and debugging integration.

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
python example.py
```

## Features

- Configurable LLM (OpenAI, Anthropic, etc.)
- Built-in tools: calculator, search, get_current_time
- Custom tool support
- LangSmith tracing integration
- LangSmith Fetch CLI for debugging
- Polly AI assistant support

## Usage

```python
from agent import create_deep_agent, DeepAgentConfig, run_agent

config = DeepAgentConfig(
    model_name="openai:gpt-4o-mini",
    max_iterations=10,
    langsmith_tracing=True,
    langsmith_project="my-agent-project",
)

agent = create_deep_agent(config)
response = run_agent(agent, "What is 15 * 23?")
print(response)
```

## LangSmith Integration

### Tracing

Traces are automatically sent to LangSmith when `langsmith_tracing=True`. View them at [smith.langchain.com](https://smith.langchain.com).

### LangSmith Fetch (Terminal Debugging)

Fetch traces directly to your terminal or IDE:

```bash
# Fetch recent traces
python debug.py traces 10

# Fetch traces from last 30 minutes
python debug.py traces 5 --minutes 30

# Fetch specific trace
python debug.py trace <trace-id>

# Or use CLI directly
langsmith-fetch traces ./traces --limit 10 --include-metadata
```

### Polly (AI Agent Debugger)

Polly is LangSmith's AI assistant for debugging agents. Access it in the LangSmith UI:

1. Open a trace in LangSmith
2. Click the Polly icon
3. Ask questions like:
   - "Why did this agent fail?"
   - "What happened at step 5?"
   - "How can I improve this prompt?"

Polly can analyze:
- Single trace executions
- Entire conversation threads
- Suggest prompt improvements

## Environment Variables

```bash
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
LANGSMITH_API_KEY=your-langsmith-key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=langchain-deep-agent
```

## Project Structure

```
langchain_deep_agent/
├── agent.py          # Main agent implementation
├── example.py        # Usage examples
├── debug.py          # LangSmith Fetch helper
├── tests/            # Unit tests
└── traces/           # Fetched traces (gitignored)
```

## Running Tests

```bash
pytest tests/
```

## Interactive Mode

```bash
python example.py --interactive
```
