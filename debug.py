#!/usr/bin/env python3
import subprocess
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

TRACES_DIR = Path(__file__).parent / "traces"


def check_langsmith_fetch_installed():
    try:
        subprocess.run(
            ["langsmith-fetch", "--help"],
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def install_langsmith_fetch():
    print("Installing langsmith-fetch...")
    subprocess.run([sys.executable, "-m", "pip", "install", "langsmith-fetch"], check=True)


def fetch_recent_traces(limit: int = 5, last_n_minutes: int | None = None, project: str = "langchain-deep-agent"):
    TRACES_DIR.mkdir(exist_ok=True)

    cmd = ["langsmith-fetch", "traces", str(TRACES_DIR), "--limit", str(limit)]

    if last_n_minutes:
        cmd.extend(["--last-n-minutes", str(last_n_minutes)])

    if project:
        cmd.extend(["--project", project])

    cmd.extend(["--include-metadata", "--include-feedback"])

    print(f"Fetching {limit} recent traces from project '{project}'...")
    subprocess.run(cmd, check=True, env=os.environ)
    print(f"Traces saved to {TRACES_DIR}/")


def fetch_trace_by_id(trace_id: str, output_format: str = "pretty"):
    cmd = ["langsmith-fetch", "trace", trace_id, "--format", output_format]
    subprocess.run(cmd, check=True, env=os.environ)


def fetch_thread_by_id(thread_id: str, output_format: str = "pretty"):
    cmd = ["langsmith-fetch", "thread", thread_id, "--format", output_format]
    subprocess.run(cmd, check=True, env=os.environ)


def show_config():
    subprocess.run(["langsmith-fetch", "config"], check=True, env=os.environ)


def analyze_trace(trace_id: str):
    cmd = ["langsmith-fetch", "trace", trace_id, "--format", "raw"]
    result = subprocess.run(cmd, capture_output=True, text=True, env=os.environ)
    if result.returncode != 0:
        print(f"Error fetching trace: {result.stderr}")
        return

    analyze_cmd = [sys.executable, "analyze.py"]
    subprocess.run(analyze_cmd, input=result.stdout, text=True)


def fetch_and_analyze(limit: int = 1, project: str = "langchain-deep-agent"):
    fetch_recent_traces(limit=limit, project=project)

    traces = sorted(TRACES_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    if traces:
        print(f"\nAnalyzing most recent trace: {traces[0].name}")
        subprocess.run([sys.executable, "analyze.py", "1"])


def main():
    if not check_langsmith_fetch_installed():
        install_langsmith_fetch()

    if len(sys.argv) < 2:
        print("LangSmith Fetch Debug Helper for Deep Agent")
        print("=" * 45)
        print("\nUsage:")
        print("  python debug.py traces [limit]               - Fetch recent traces")
        print("  python debug.py trace <trace-id>             - Fetch specific trace")
        print("  python debug.py analyze <trace-id>           - Fetch and analyze with Ollama")
        print("  python debug.py full                         - Fetch + analyze latest trace")
        print("  python debug.py config                       - Show config")
        print("\nExamples:")
        print("  python debug.py traces 10")
        print("  python debug.py full")
        print("  python debug.py analyze 3b0b15fe-1e3a-4aef-afa8-48df15879cfe")
        print("\nProject: langchain-deep-agent")
        return

    command = sys.argv[1]

    if command == "traces":
        limit = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 5
        last_n_minutes = None
        if "--minutes" in sys.argv:
            idx = sys.argv.index("--minutes")
            if idx + 1 < len(sys.argv):
                last_n_minutes = int(sys.argv[idx + 1])
        fetch_recent_traces(limit, last_n_minutes)

    elif command == "trace" and len(sys.argv) > 2:
        fetch_trace_by_id(sys.argv[2])

    elif command == "thread" and len(sys.argv) > 2:
        fetch_thread_by_id(sys.argv[2])

    elif command == "config":
        show_config()

    elif command == "analyze" and len(sys.argv) > 2:
        analyze_trace(sys.argv[2])

    elif command == "full":
        fetch_and_analyze()

    else:
        print(f"Unknown command: {command}")
        print("Run 'python debug.py' for help")


if __name__ == "__main__":
    main()
