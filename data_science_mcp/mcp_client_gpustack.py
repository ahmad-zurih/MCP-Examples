"""
GPUStack MCP Client for the Advanced Visualization + System Tools Server.
Uses the OpenAI-compatible GPUStack API with native tool calling.
All tools (viz, stats, coder) are available to a single powerful model in one agent loop.

Usage:
  python mcp_client_gpustack.py

Configuration:
  Set api_key and api_base_url in the .env file inside data_science_mcp/.
  Change MODEL below to switch models.
"""

import asyncio
import base64
import json
import os
import re
import sys
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Load .env from project root (one level up from advanced_mcp/)
try:
    from dotenv import load_dotenv
    # Look for .env in current_dir first, then parent
    _env_path = os.path.join(current_dir, ".env")
    if not os.path.exists(_env_path):
        _env_path = os.path.join(current_dir, "..", ".env")
    load_dotenv(_env_path)
except ImportError:
    pass  # Fall back to environment variables already set

from openai import OpenAI
from openai import APIStatusError, APIConnectionError
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ── Configuration — edit here ──────────────────────────────────────────────
MODEL               = "qwen3-coder-30b-a3b-instruct"   # change model here
MAX_TOOL_ITERATIONS = 250
# ───────────────────────────────────────────────────────────────────────────

MCP_SERVER_SCRIPT = os.path.join(current_dir, "mcp_server.py")
_openai_singleton: "OpenAI | None" = None

_SYSTEM_PROMPT = """You are a powerful data analysis and software engineering assistant.
You have access to a full suite of tools via an MCP server. Use them proactively.

VISUALIZATION (saves as interactive HTML):
  plot_interactive_histogram, plot_interactive_scatterplot, plot_interactive_boxplot,
  plot_interactive_lineplot, plot_interactive_barchart, plot_interactive_scatter_matrix,
  plot_interactive_correlation_heatmap, generate_custom_plotly

STATIC CHARTS (saves as PNG):
  plot_static_histogram, plot_static_scatterplot, plot_static_boxplot,
  plot_static_lineplot, plot_static_barchart, plot_static_pairplot,
  plot_static_correlation_heatmap, plot_static_wordcloud, generate_custom_static_plot

DATA EXPLORATION:
  get_all_columns_summary, get_column_summary

STATISTICS:
  run_correlation, run_group_comparison, run_linear_regression, rank_target_correlations

SYSTEM / CODE (build anything on disk):
  run_shell_command    — execute any bash command
  read_file            — read a file from disk
  write_file           — write a complete new file (creates dirs)
  patch_file           — surgically edit one snippet inside an existing file (PREFER over write_file)
  list_directory       — list directory contents
  find_in_files        — grep/search across a codebase
  run_background_process  — start a server or daemon (tracked by label)
  stop_background_process — stop a tracked process
  list_background_processes — show running processes
  http_request         — test an HTTP endpoint

INTERNET ACCESS:
  search_web           — DuckDuckGo search, no API key (use before every coding task)
  fetch_webpage        — fetch URL: title, headings, CSS colors/fonts, page text (ideal for cloning sites)
  screenshot_webpage   — headless Chromium screenshot of any URL, saved as PNG

RULES:
1. For datasets: always call get_all_columns_summary first, then plot or analyse.
2. Always pass the absolute data_file_path when calling viz/stats tools.
3. For coding tasks follow: PLAN → BUILD (write_file/patch_file) → VALIDATE (py_compile) → TEST → FIX.
4. Prefer patch_file over write_file when editing existing files.
5. After starting a server with run_background_process, use http_request to verify it responds.
6. Never say you cannot do something that your tools support. Just do it.
"""


# Names of all system/coder tools — used to decide routing
_CODER_TOOL_NAMES = {
    "run_shell_command", "read_file", "write_file", "patch_file",
    "list_directory", "find_in_files", "run_background_process",
    "stop_background_process", "list_background_processes", "http_request",
    "fetch_webpage", "search_web", "screenshot_webpage",
}

_CODER_SUPERVISOR_SYSTEM = """You are a Coding Project Supervisor. You coordinate software projects
by breaking them into atomic subtasks and delegating them ONE AT A TIME to a Coding Worker
via the delegate_coding_task tool. You do NOT write code or use any other tools.

WORKFLOW:
1. PLAN first: think through the full project — directories, files, dependencies, test strategy.
2. Delegate ONE atomic subtask at a time (one file, one command, one install, one test).
3. Include exact file paths and all context the worker needs (it has no memory of previous subtasks).
4. After each worker report, update your mental log and decide the next step.
5. After everything is built, delegate a verification step.
6. When done: write a final summary with file tree and exact run commands.
"""

_CODER_WORKER_SYSTEM = """You are a focused Coding Worker. You receive ONE specific atomic task
and complete it using your tools. Be precise and efficient.
Do NOT plan ahead. Just execute the given task and report what you did.

Tools: run_shell_command, read_file, write_file, patch_file, list_directory,
       find_in_files, run_background_process, stop_background_process, list_background_processes, http_request,
       search_web, fetch_webpage, screenshot_webpage

Rules:
- Prefer patch_file over write_file for editing existing files.
- After write_file, run python3 -m py_compile <file> to validate Python syntax.
- If a command fails, read the error and fix it before reporting completion.
- Report concisely: what you did, what command you ran, what the output was.
"""

_DELEGATE_CODING_TASK_TOOL = {
    "type": "function",
    "function": {
        "name": "delegate_coding_task",
        "description": "Delegate one atomic coding subtask to a worker agent.",
        "parameters": {
            "type": "object",
            "properties": {
                "subtask": {
                    "type": "string",
                    "description": "The exact task for the worker. Include file paths and specific goals.",
                },
                "context": {
                    "type": "string",
                    "description": "Project context: directory layout, already-created files, env paths, progress so far.",
                },
            },
            "required": ["subtask", "context"],
        },
    },
}

_ROUTER_SYSTEM = """Analyse the conversation and classify the LATEST user message.
Reply with ONLY raw JSON (no markdown):
{"needs_tools": true/false, "is_coding": true/false, "data_file_path": "path_or_null"}

needs_tools=true if: user wants a plot, chart, statistics, data analysis, OR wants to
  create/edit files, build a project, run commands, install packages, or is following up
  on a previous analysis or coding task.
is_coding=true if: the request involves file creation, writing code, building an app,
  running shell commands, installing packages, or any system operations.
  is_coding=false for pure data visualization or statistics on a dataset.
data_file_path: the most recently mentioned absolute path to a DATA file (csv/excel/json),
  or null if none.
"""


def _get_openai_client() -> tuple["OpenAI", str]:
    """Read api_key and api_base_url from .env and return (client, base_url)."""
    api_key = (
        os.environ.get("API_KEY") or os.environ.get("api_key") or ""
    ).strip()
    base_url = (
        os.environ.get("API_BASE_URL") or os.environ.get("api_base_url") or
        "https://api.openai.com/v1"
    ).strip()
    if not api_key:
        print("[Error] api_key not set. Add a .env file in the project root")
        print("  Example .env:")
        print("    api_base_url=https://api.openai.com/v1")
        print("    api_key=sk-...")
        sys.exit(1)
    return OpenAI(base_url=base_url, api_key=api_key), base_url


def _route(conversation: list[dict]) -> tuple[bool, bool, str | None]:
    """Return (needs_tools, is_coding, data_file_path)."""
    try:
        # _openai_singleton set in main() before any calls
        resp = _openai_singleton.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": _ROUTER_SYSTEM}] + conversation[-6:],
            temperature=0,
            max_tokens=120,
        )
        raw = resp.choices[0].message.content or ""
        raw = re.sub(r"```[a-z]*\n?|```", "", raw).strip()
        data = json.loads(raw)
        path = data.get("data_file_path")
        if not path or path in ("null", "path_or_null", ""):
            path = None
        return bool(data.get("needs_tools")), bool(data.get("is_coding")), path
    except Exception:
        return True, False, None  # default: run normal agent_loop



def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks that some reasoning models emit."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _msg_to_dict(msg) -> dict:
    """Convert an OpenAI Message object to a plain dict suitable for the messages list."""
    d: dict = {"role": msg.role, "content": msg.content or ""}
    if getattr(msg, "tool_calls", None):
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
    return d


def _truncate(text: str, limit: int = 8000) -> str:
    if len(text) > limit:
        return text[:limit] + f"\n... [truncated — {len(text):,} chars total]"
    return text


def _print_plots(plots: list[dict]) -> None:
    if not plots:
        return
    print(f"\n  {'─' * 54}")
    print(f"  {len(plots)} plot(s) saved:")
    for p in plots:
        path = p["path"]
        ext = os.path.splitext(path)[1].lower()
        kind = "Interactive HTML" if ext == ".html" else "Static PNG" if ext == ".png" else ext.lstrip(".").upper()
        label = p.get("tool_name", "plot").replace("_", " ").title()
        print(f"\n    [{kind}]  {label}")
        print(f"    {path}")
    print(f"\n  {'─' * 54}\n")



def _format_tool_call(name: str, args: dict) -> str:
    def _s(v: str, n: int = 55) -> str:
        v = str(v).replace("\n", " ").strip()
        return v if len(v) <= n else v[:n] + "..."

    if name == "run_shell_command":
        return _s(args.get("command", ""), 70)
    if name == "write_file":
        path = args.get("file_path", "")
        size = len(args.get("content", ""))
        return f"{path}  ({size:,} chars)"
    if name == "patch_file":
        path = args.get("file_path", "")
        old = _s(args.get("old_str", "").lstrip(), 35)
        return f"{path}  <- {repr(old)}"
    if name in ("read_file",):
        return args.get("file_path", "")
    if name == "list_directory":
        return args.get("dir_path", "")
    if name == "find_in_files":
        pat = args.get("pattern", "")
        d = os.path.basename(args.get("directory", ""))
        glob = args.get("file_glob", "*.py")
        return f'"{pat}"  in {d}/  [{glob}]'
    if name == "run_background_process":
        label = args.get("label", "")
        cmd = _s(args.get("command", ""), 45)
        return f"[{label}]  {cmd}"
    if name == "stop_background_process":
        return f"[{args.get('label', '')}]"
    if name == "list_background_processes":
        return ""
    if name == "http_request":
        return f"{args.get('method','GET').upper()}  {args.get('url','')}"
    if name == "search_web":
        return _s(args.get("query", ""), 65)
    if name == "fetch_webpage":
        return args.get("url", "")
    if name == "screenshot_webpage":
        return args.get("url", "")
    # Viz / stats tools — show filename + key column args
    if "data_file_path" in args:
        fname = os.path.basename(args["data_file_path"])
        col_args = [v for k, v in args.items()
                    if k not in ("data_file_path", "python_code", "plot_filename_keyword")
                    and isinstance(v, str) and v]
        extra = "  ".join(col_args[:3])
        return f"{fname}" + (f"  |  {_s(extra, 45)}" if extra else "")
    return ""

def _trim_tools(tools: list[dict]) -> list[dict]:
    """
    Shorten tool descriptions to reduce token usage.
    Keeps only the first sentence of each description, and removes
    verbose 'additionalProperties' / 'title' noise from JSON schemas.
    """
    trimmed = []
    for t in tools:
        fn = dict(t["function"])
        desc = fn.get("description", "")
        # Keep only first sentence
        short = desc.split("\n")[0].split(". ")[0].rstrip(".") + "."
        fn["description"] = short
        # Clean schema noise
        params = dict(fn.get("parameters") or {})
        props = {}
        for pname, pval in params.get("properties", {}).items():
            p = dict(pval)
            p.pop("title", None)
            if isinstance(p.get("description"), str):
                p["description"] = p["description"].split("\n")[0][:120]
            props[pname] = p
        if props:
            params = dict(params)
            params["properties"] = props
            params.pop("additionalProperties", None)
            fn["parameters"] = params
        trimmed.append({"type": "function", "function": fn})
    return trimmed


async def _run_coding_worker_gpu(
    subtask: str,
    context: str,
    session: "ClientSession",
    coder_tools: list[dict],
    openai_client: "OpenAI",
    log_fn: "Callable",
) -> str:
    """Fresh-context worker for one atomic coding subtask."""
    messages = [
        {"role": "system", "content": _CODER_WORKER_SYSTEM},
        {"role": "user", "content": (
            f"Context from supervisor:\n{context}\n\nYour task:\n{subtask}" if context else subtask
        )},
    ]
    trimmed = _trim_tools(coder_tools)

    for _i in range(8):
        for _attempt in range(3):
            try:
                resp = openai_client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=trimmed,
                    tool_choice="auto",
                    temperature=0.3,
                    max_tokens=8000,
                )
                break
            except (APIStatusError, APIConnectionError) as _exc:
                _code = getattr(_exc, "status_code", 0)
                if _code in (502, 503, 504) or isinstance(_exc, APIConnectionError):
                    if _attempt < 2:
                        _w = 4 * (_attempt + 1)
                        log_fn(f"    [retry {_attempt+1}/2 after {_w}s]")
                        time.sleep(_w)
                        continue
                raise

        msg = resp.choices[0].message
        messages.append(_msg_to_dict(msg))

        if not msg.tool_calls:
            return _strip_thinking(msg.content or "Task completed.")

        results = []
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            detail = _format_tool_call(name, args)
            log_fn(f"    > {name}" + (f"  |  {detail}" if detail else ""))
            try:
                mcp_result = await session.call_tool(name, arguments=args)
                raw = "\n".join(c.text for c in mcp_result.content if c.type == "text")
            except Exception as e:
                raw = f"Error: {e}"
            results.append({"role": "tool", "tool_call_id": tc.id, "content": _truncate(raw, 3000)})
        messages.extend(results)

    return "Worker reached iteration limit."


async def _coder_supervisor_loop(
    task: str,
    context_summary: str,
    session: "ClientSession",
    all_tools: list[dict],
    openai_client: "OpenAI",
    log_fn: "Callable",
) -> str:
    """
    Supervisor that plans a coding project and delegates atomic subtasks to
    fresh-context workers. Avoids the 502 timeout by keeping each LLM call small.
    """
    coder_tools = [t for t in all_tools if t["function"]["name"] in _CODER_TOOL_NAMES]

    supervisor_messages = [
        {"role": "system", "content": _CODER_SUPERVISOR_SYSTEM},
        {"role": "user", "content": (
            f"Conversation context:\n{context_summary}\n\nProject task:\n{task}"
        )},
    ]
    completed: list[str] = []

    log_fn("  [Coder supervisor] Planning project breakdown...")

    for round_num in range(20):
        for _attempt in range(3):
            try:
                resp = openai_client.chat.completions.create(
                    model=MODEL,
                    messages=supervisor_messages,
                    tools=[_DELEGATE_CODING_TASK_TOOL],
                    tool_choice="auto",
                    temperature=0.4,
                    max_tokens=4000,
                )
                break
            except (APIStatusError, APIConnectionError) as _exc:
                _code = getattr(_exc, "status_code", 0)
                if _code in (502, 503, 504) or isinstance(_exc, APIConnectionError):
                    if _attempt < 2:
                        _w = 4 * (_attempt + 1)
                        log_fn(f"  [retry {_attempt+1}/2 after {_w}s]")
                        time.sleep(_w)
                        continue
                raise

        msg = resp.choices[0].message
        supervisor_messages.append(_msg_to_dict(msg))

        if not msg.tool_calls:
            log_fn(f"  [Coder supervisor] Done ({len(completed)} subtask(s) completed).")
            return _strip_thinking(msg.content or "Project completed.")

        for tc in msg.tool_calls:
            if tc.function.name != "delegate_coding_task":
                continue
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            subtask = args.get("subtask", "")
            context = args.get("context", "")

            log_fn(f"\n  [subtask {round_num + 1}] {subtask[:85]}")

            worker_result = await _run_coding_worker_gpu(
                subtask=subtask,
                context=context,
                session=session,
                coder_tools=coder_tools,
                openai_client=openai_client,
                log_fn=log_fn,
            )

            completed.append(subtask[:65])
            progress = "\n".join(f"  v {t}" for t in completed[-10:])

            supervisor_messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": (
                    f"Worker report:\n{worker_result}\n\n"
                    f"Completed ({len(completed)} subtask(s)):\n{progress}"
                ),
            })

    return "Coder supervisor reached max rounds. Project may be incomplete."



def _inject_vision_if_screenshot(
    conversation: list[dict],
    tool_calls: list,
    tool_results: list[dict],
) -> None:
    """
    For VL models: when a screenshot_webpage tool returns a file path, inject
    the PNG as a base64 image_url message so the model can actually see the page.
    Only activates when MODEL contains 'vl' or 'vision' (case-insensitive).
    """
    model_lower = MODEL.lower()
    if "vl" not in model_lower and "vision" not in model_lower:
        return
    for tc, result in zip(tool_calls, tool_results):
        if tc.function.name != "screenshot_webpage":
            continue
        result_text = result.get("content", "")
        if not result_text.startswith("Screenshot saved to: "):
            continue
        png_path = result_text.replace("Screenshot saved to: ", "").strip()
        if not os.path.exists(png_path):
            continue
        try:
            with open(png_path, "rb") as fh:
                img_b64 = base64.b64encode(fh.read()).decode()
            conversation.append({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            "Above is the screenshot of the webpage you just captured. "
                            "Analyse its visual design, layout, color scheme, typography, "
                            "and structure so you can clone or reference it accurately."
                        ),
                    },
                ],
            })
        except Exception:
            pass  # non-fatal if image inject fails


async def agent_loop(
    user_message: str,
    conversation: list[dict],
    session: ClientSession,
    tools: list[dict],
    openai_client: OpenAI,
) -> tuple[str, list[dict]]:
    """
    Multi-turn tool-calling loop for one user turn.
    Keeps calling tools until the model stops requesting them.
    Returns (final_reply_text, plots_generated_this_turn).
    """
    conversation.append({"role": "user", "content": user_message})
    plots: list[dict] = []
    printed_newline = False

    for iteration in range(MAX_TOOL_ITERATIONS):
        for _attempt in range(3):
            try:
                resp = openai_client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "system", "content": _SYSTEM_PROMPT}] + conversation,
                    tools=_trim_tools(tools),
                    tool_choice="auto",
                    temperature=0.6,
                    max_tokens=16000,
                )
                break  # success
            except (APIStatusError, APIConnectionError) as _exc:
                _code = getattr(_exc, "status_code", 0)
                if _code in (502, 503, 504) or isinstance(_exc, APIConnectionError):
                    if _attempt < 2:
                        _wait = 4 * (_attempt + 1)
                        print(f"  [retry {_attempt+1}/2 after {_wait}s — server error: {_code}]")
                        time.sleep(_wait)
                        continue
                raise

        msg = resp.choices[0].message
        conversation.append(_msg_to_dict(msg))

        if not msg.tool_calls:
            return _strip_thinking(msg.content or ""), plots

        if not printed_newline:
            print()
            printed_newline = True

        tool_results = []
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            _detail = _format_tool_call(name, args)
            print(f"  > {name}" + (f"  |  {_detail}" if _detail else ""))

            try:
                mcp_result = await session.call_tool(name, arguments=args)
                raw = "\n".join(c.text for c in mcp_result.content if c.type == "text")
            except Exception as e:
                raw = f"Error calling tool: {str(e)}"

            # Intercept plot artifacts — extract path, hide internal path from model
            if "|||" in raw and not raw.startswith("Error"):
                path_part, _ = raw.split("|||", 1)
                plots.append({"path": path_part.strip(), "tool_name": name})
                raw = "Plot generated and saved successfully."

            tool_results.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": _truncate(raw),
            })

        _inject_vision_if_screenshot(conversation, msg.tool_calls, tool_results)
        conversation.extend(tool_results)

    return "Reached maximum tool iterations.", plots


async def main() -> None:
    global _openai_singleton
    openai_client, _base_url = _get_openai_client()
    _openai_singleton = openai_client

    print(f"[Chat] model : {MODEL}")
    print(f"[Chat] server: {_base_url}")
    print("[Chat] connecting to MCP server...", end=" ", flush=True)

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[MCP_SERVER_SCRIPT],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_resp = await session.list_tools()

            # Convert MCP tools → OpenAI tool format
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.inputSchema,
                    },
                }
                for t in tools_resp.tools
            ]

            _raw_tool_count = len(tools)
            print(f"{_raw_tool_count} tools loaded ({sum(len(json.dumps(t)) for t in tools)//1024} KB schema).")
            print("[Chat] Type 'quit' to exit.\n")

            conversation: list[dict] = []
            _pending_data_path: str | None = None

            while True:
                try:
                    user_input = input("You: ").strip()
                    if not user_input:
                        continue
                    if user_input.lower() in ("quit", "exit"):
                        break

                    needs_tools, is_coding, extracted_path = _route(conversation)
                    if extracted_path and os.path.exists(extracted_path):
                        _pending_data_path = extracted_path

                    if is_coding:
                        ctx = "\n".join(
                            f"{m['role']}: {str(m.get('content',''))[:200]}"
                            for m in conversation[-8:]
                        )
                        reply = await _coder_supervisor_loop(
                            task=user_input,
                            context_summary=ctx,
                            session=session,
                            all_tools=tools,
                            openai_client=openai_client,
                            log_fn=lambda m: print(m),
                        )
                        plots = []
                    else:
                        reply, plots = await agent_loop(
                            user_message=user_input,
                            conversation=conversation,
                            session=session,
                            tools=tools,
                            openai_client=openai_client,
                        )

                    print(f"\nAssistant: {reply}")
                    _print_plots(plots)

                except KeyboardInterrupt:
                    print("\n[GPUStack Chat] Bye!")
                    break
                except Exception as e:
                    print(f"\n[Error] {e}\n")


if __name__ == "__main__":
    asyncio.run(main())