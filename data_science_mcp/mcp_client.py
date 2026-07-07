"""
MCP Client for the Advanced Visualization Engine.
Free-form chat with a router agent that decides when to call the
visualization/stats/coder MAS vs. answer directly.
"""

import asyncio
import json
import os
import re
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import ollama
from viz_agent import run_analysis

OLLAMA_MODEL = "qwen3.5:9b"
MCP_SERVER_SCRIPT = os.path.join(current_dir, "mcp_server.py")

_TOOLS_DESCRIPTION = """You are a data-analysis and software-development assistant backed by an MCP server.

You have access to the following tools via the server:

VISUALIZATION (Plotly HTML - interactive):
  histogram, scatter, box, line, bar chart, scatter matrix, correlation heatmap, custom plotly

STATIC CHARTS (Matplotlib/Seaborn PNG):
  histogram, scatter, box, line, bar chart, pair plot, correlation heatmap, word cloud, custom static

STATISTICS:
  Pearson/Spearman correlation, T-test / ANOVA, OLS linear regression, feature correlation ranking

SYSTEM / CODE:
  run_shell_command  - execute any bash command
  read_file          - read any file from disk
  write_file         - write/create any file on disk
  list_directory     - list directory contents

INTERNET ACCESS:
  search_web         - DuckDuckGo web search (no API key)
  fetch_webpage      - fetch a URL and extract title, headings, CSS colors/fonts, page text
  screenshot_webpage - take a headless Chromium screenshot of any URL, saved as PNG

You can build complete projects, create Streamlit apps, install packages, write scripts, etc.
You can also look things up on the web, research libraries, and inspect sites before cloning them.
Never say you cannot do something that involves files, commands, or the internet - you can.
If no dataset path has been provided yet for analysis tasks, ask for it.
"""

_ROUTER_SYSTEM = """You are a routing assistant. Analyse the FULL conversation and decide for the LATEST user message:

1. Does it require the MCP server tools? Answer YES if ANY of the following apply:
   - Wants a plot, chart, graph, heatmap, or visualization
   - Wants statistics, correlation, regression, t-test, ANOVA
   - Wants to create files, write code, build a project, make an app
   - Wants to run a command, install packages, set up something on disk
   - Is following up on a previous analysis or coding task (e.g. "now add X", "fix that", "same data")
   Answer NO only for pure chat: greetings, general knowledge questions, meta questions about capabilities.

2. What is the most recently mentioned absolute file path for a DATASET in the conversation?
   Return null if no dataset path has been mentioned. This is only for data files (CSV, Excel, JSON), not code files.

Reply with ONLY raw JSON, no markdown, no explanation:
{"needs_tools": true_or_false, "data_file_path": "path_or_null"}
"""


def _log(log_type: str, msg: str) -> None:
    icons = {"info": "  ...", "warning": "  !", "error": "  x"}
    print(icons.get(log_type, "  *") + " " + msg)


def _route(conversation: list[dict]) -> tuple[bool, str | None]:
    try:
        resp = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "system", "content": _ROUTER_SYSTEM}] + conversation,
            options={"temperature": 0},
        )
        raw = resp["message"]["content"].strip()
        raw = re.sub(r"```[a-z]*\n?|```", "", raw).strip()
        data = json.loads(raw)
        path = data.get("data_file_path")
        if not path or path in ("null", "path_or_null", ""):
            path = None
        return bool(data.get("needs_tools")), path
    except Exception:
        return False, None


def _print_plots(plots: list[dict]) -> None:
    if not plots:
        return
    print(f"\n  {'─' * 52}")
    print(f"  {len(plots)} plot(s) saved:")
    for p in plots:
        path = p["path"]
        ext = os.path.splitext(path)[1].lower()
        kind = "Interactive HTML" if ext == ".html" else "Static PNG" if ext == ".png" else ext.upper().lstrip(".")
        label = p.get("tool_name", "plot").replace("_", " ").title()
        print(f"\n    [{kind}]  {label}")
        print(f"    {path}")
    print(f"\n  {'─' * 52}\n")


def _print_stats(stats: list[dict]) -> None:
    if not stats:
        return
    print("\n  [Statistical Results]")
    for s in stats:
        title = s["title"]
        print(f"\n  {title}\n  {'─' * len(title)}")
        for line in s["result"].splitlines():
            print(f"  {line}")
    print()


async def main() -> None:
    print(f"[Chat] model: {OLLAMA_MODEL}  |  Type 'quit' to exit.")
    print("[Chat] Chat freely. Ask for plots, stats, or to build/run anything on disk.\n")

    conversation: list[dict] = []
    pending_data_path: str | None = None

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit"):
                break

            conversation.append({"role": "user", "content": user_input})
            needs_tools, extracted_path = _route(conversation)

            if extracted_path and os.path.exists(extracted_path):
                pending_data_path = extracted_path

            if needs_tools:
                print("\n[Running...]\n")
                result = await run_analysis(
                    messages=conversation,
                    data_file_path=pending_data_path or "",
                    model_name=OLLAMA_MODEL,
                    mcp_server_script=MCP_SERVER_SCRIPT,
                    log_callback=_log,
                )
                print(f"\nAssistant: {result['summary']}")
                _print_plots(result["plots"])
                _print_stats(result["stats"])
                conversation.append({"role": "assistant", "content": result["summary"]})

            else:
                resp = ollama.chat(
                    model=OLLAMA_MODEL,
                    messages=[{"role": "system", "content": _TOOLS_DESCRIPTION}] + conversation,
                )
                reply = resp["message"]["content"]
                print(f"\nAssistant: {reply}\n")
                conversation.append({"role": "assistant", "content": reply})

        except KeyboardInterrupt:
            print("\n[Chat] Bye!")
            break
        except Exception as e:
            print(f"\n[Error] {e}\n")


if __name__ == "__main__":
    asyncio.run(main())