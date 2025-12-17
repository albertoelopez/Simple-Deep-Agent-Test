#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from langchain_ollama import ChatOllama

TRACES_DIR = Path(__file__).parent / "traces"


def load_trace(trace_id: str) -> dict:
    trace_file = TRACES_DIR / f"{trace_id}.json"
    if not trace_file.exists():
        files = list(TRACES_DIR.glob("*.json"))
        if trace_id.isdigit() and int(trace_id) <= len(files):
            trace_file = sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)[int(trace_id) - 1]
        else:
            raise FileNotFoundError(f"Trace {trace_id} not found")

    with open(trace_file) as f:
        return json.load(f)


def analyze_trace(trace: dict, question: str = None) -> str:
    model = ChatOllama(model="gpt-oss", temperature=0)

    trace_summary = json.dumps(trace, indent=2)

    if question:
        prompt = f"""Analyze this LangChain agent trace and answer: {question}

Trace:
{trace_summary}"""
    else:
        prompt = f"""Analyze this LangChain agent trace. Identify:
1. What the user asked
2. What tools were used
3. Any errors or retries
4. How the agent arrived at the answer
5. Suggestions for improvement

Trace:
{trace_summary}"""

    response = model.invoke(prompt)
    return response.content


def main():
    if not sys.stdin.isatty():
        stdin_data = sys.stdin.read().strip()
        if stdin_data:
            print("Analyzing piped trace with Ollama gpt-oss...\n")
            print("-" * 50)
            trace = json.loads(stdin_data)
            if isinstance(trace, list):
                trace = {"messages": trace}
            question = sys.argv[1] if len(sys.argv) > 1 else None
            analysis = analyze_trace(trace, question)
            print(analysis)
            return

    if len(sys.argv) < 2:
        print("Local Trace Analyzer (Polly Alternative)")
        print("=" * 45)
        print("\nUsage:")
        print("  python analyze.py <trace-id>              - Analyze a trace")
        print("  python analyze.py <trace-id> '<question>' - Ask about a trace")
        print("  python analyze.py 1                       - Analyze most recent trace")
        print("  python analyze.py list                    - List available traces")
        print("  langsmith-fetch trace <id> | python analyze.py  - Pipe from fetch")
        print("\nExamples:")
        print("  python analyze.py 1")
        print("  python analyze.py 55b8499a-a65c-485d-976e-8937bb548977")
        print("  python analyze.py 1 'Why did the tool fail?'")
        print("  langsmith-fetch trace <id> --format raw | python analyze.py")
        return

    if sys.argv[1] == "list":
        files = sorted(TRACES_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
        print(f"Available traces in {TRACES_DIR}/:\n")
        for i, f in enumerate(files, 1):
            with open(f) as fp:
                data = json.load(fp)
            status = data.get("metadata", {}).get("status", "unknown")
            msgs = len(data.get("messages", []))
            print(f"  {i}. {f.stem} ({msgs} msgs, {status})")
        return

    trace_id = sys.argv[1]
    question = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Loading trace: {trace_id}")
    trace = load_trace(trace_id)

    print(f"Analyzing with Ollama gpt-oss...\n")
    print("-" * 50)

    analysis = analyze_trace(trace, question)
    print(analysis)


if __name__ == "__main__":
    main()
