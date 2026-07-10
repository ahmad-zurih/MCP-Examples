"""
Autonomous Master Agent Client
================================
A time-budgeted autonomous AI that uses all MCP tools to accomplish tasks.
The master planner dispatches subtasks to parallel workers, tracks progress,
enforces safety rules, and produces a final report when time is up.

Usage:
  python mcp_client_autonomous.py                            # autonomous mode, 1 hour
  python mcp_client_autonomous.py --time 30m                 # 30-minute autonomous session
  python mcp_client_autonomous.py --time 2h --task "build a QR code web app"
  python mcp_client_autonomous.py -t 45m -g "analyse the Titanic CSV at ~/Titanic-Dataset.csv"
"""

import argparse
import asyncio
import dataclasses
import json
import os
import re
import sys
import textwrap
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from dotenv import load_dotenv
    _env_path = os.path.join(current_dir, ".env")
    if not os.path.exists(_env_path):
        _env_path = os.path.join(current_dir, "..", ".env")
    load_dotenv(_env_path)
except ImportError:
    pass

from openai import OpenAI, APIStatusError, APIConnectionError
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ── Configuration ──────────────────────────────────────────────────────────
PLANNER_MODEL        = "minimax-m2.7"   # change to any model you have
WORKER_MODEL         = "gpt-oss-120b"   # can differ from planner
MAX_PARALLEL_WORKERS = 3      # max workers running simultaneously
PLANNER_TICK_SECONDS = 100    # how often planner re-evaluates
WORKER_MAX_ITERS     = 300     # max tool calls per worker task
WORKER_MAX_TOKENS    = 10000
PLANNER_MAX_TOKENS   = 8000
WRAP_UP_SECONDS      = 180    # stop starting new tasks when this much time is left
# ───────────────────────────────────────────────────────────────────────────

MCP_SERVER_SCRIPT = os.path.join(current_dir, "mcp_server.py")
_HOME = os.path.expanduser("~")
REPORTS_DIR = os.path.join(_HOME, "mcp-server-example", "session-reports")


# ── Safety Rules ───────────────────────────────────────────────────────────

_BLOCKED_CMD_PATTERNS = [
    (r"rm\s+(-[a-zA-Z]*[fr][a-zA-Z]*\s+){0,2}(/\s*$|/\s|~/|/home/[^/]+/\s*$)", "destructive rm on root/home"),
    (r"rm\s+-[a-zA-Z]*r[a-zA-Z]*f[a-zA-Z]*\s", "rm -rf"),
    (r"rm\s+-[a-zA-Z]*f[a-zA-Z]*r[a-zA-Z]*\s", "rm -fr"),
    (r"\|\s*(ba|z|da|k)?sh(\s|$)", "pipe-to-shell"),
    (r"curl[^|]+\|\s*(ba|z|da|k)?sh", "curl-pipe-shell"),
    (r"wget[^|]+\|\s*(ba|z|da|k)?sh", "wget-pipe-shell"),
    (r"mkfs\.", "filesystem format"),
    (r"dd\s+if=", "disk destroyer"),
    (r">\s*/dev/[sh]d[a-z]", "write to block device"),
    (r":\(\)\s*\{.*\}", "fork bomb"),
    (r"\bshutdown\b", "shutdown"),
    (r"\breboot\b", "reboot"),
    (r"\bhalt\b", "halt"),
    (r"\bsudo\s", "sudo"),
    (r"\bsu\s+-", "su -"),
    (r"passwd\s", "passwd change"),
    (r"chmod\s+[0-7]*7[0-7]*\s+/", "chmod 777 on /"),
    (r"chown\s+root", "chown root"),
]

_BLOCKED_READ_PATHS = [
    "/.ssh/", "/.gnupg/", "/etc/shadow", "/etc/passwd",
    "/.aws/credentials", "/.config/google-chrome",
]

_BLOCKED_WRITE_PATHS = [
    "/.ssh/", "/.gnupg/", "/etc/", "/sys/", "/proc/", "/boot/", "/.aws/",
]


def _check_command(cmd: str) -> str | None:
    for pat, label in _BLOCKED_CMD_PATTERNS:
        if re.search(pat, cmd, re.IGNORECASE):
            return f"SAFETY BLOCK [{label}]: '{cmd[:80]}'"
    return None


def _check_path(path: str, write: bool = False) -> str | None:
    blocked = _BLOCKED_WRITE_PATHS if write else _BLOCKED_READ_PATHS
    norm = os.path.normpath(path)
    for b in blocked:
        if b in norm:
            return f"SAFETY BLOCK: path '{path}' is restricted"
    if write:
        if not (norm.startswith(_HOME) or norm.startswith("/tmp")):
            return f"SAFETY BLOCK: writes outside home/tmp not allowed (path: {path})"
        # Block writes directly into home root — must be in a subdirectory
        parent = os.path.dirname(norm)
        if parent == _HOME:
            return (
                f"SAFETY BLOCK: cannot write directly into home directory ('{path}'). "
                f"Create a project subdirectory first, e.g. {_HOME}/my-project/{os.path.basename(norm)}"
            )
    return None


# ── State ──────────────────────────────────────────────────────────────────

@dataclass
class TaskRecord:
    id: int
    title: str
    description: str
    worker_type: str          # "coder" | "research" | "general"
    status: str = "pending"   # pending / running / done / failed
    result_summary: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0


@dataclass
class AgentState:
    deadline: float
    goal: str
    session_start: float = dataclasses.field(default_factory=time.time)
    tasks: list = dataclasses.field(default_factory=list)
    log_lines: list = dataclasses.field(default_factory=list)
    artifacts: list = dataclasses.field(default_factory=list)
    _counter: int = 0

    def time_remaining(self) -> float:
        return max(0.0, self.deadline - time.time())

    def time_remaining_str(self) -> str:
        s = int(self.time_remaining())
        h, r = divmod(s, 3600)
        m, sec = divmod(r, 60)
        if h:
            return f"{h}h {m}m"
        if m:
            return f"{m}m {sec}s"
        return f"{sec}s"

    def elapsed_str(self) -> str:
        s = int(time.time() - self.session_start)
        h, r = divmod(s, 3600)
        m, sec = divmod(r, 60)
        if h:
            return f"{h}h {m}m elapsed"
        if m:
            return f"{m}m {sec}s elapsed"
        return f"{sec}s elapsed"

    def next_id(self) -> int:
        self._counter += 1
        return self._counter

    def pending(self) -> list:
        return [t for t in self.tasks if t.status == "pending"]

    def running(self) -> list:
        return [t for t in self.tasks if t.status == "running"]

    def done(self) -> list:
        return [t for t in self.tasks if t.status in ("done", "failed")]

    def log(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self.log_lines.append(line)
        print(line)


# ── API Helper ─────────────────────────────────────────────────────────────

def _api_call(
    client: OpenAI,
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    temperature: float = 0.4,
    max_tokens: int = 4000,
    log_fn: Callable | None = None,
) -> Any:
    for attempt in range(3):
        try:
            kwargs: dict = dict(model=model, messages=messages,
                                temperature=temperature, max_tokens=max_tokens)
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            return client.chat.completions.create(**kwargs)
        except (APIStatusError, APIConnectionError) as exc:
            code = getattr(exc, "status_code", 0)
            if code in (502, 503, 504) or isinstance(exc, APIConnectionError):
                if attempt < 2:
                    wait = 6 * (attempt + 1)
                    if log_fn:
                        log_fn(f"  [retry {attempt+1}/2 after {wait}s — {code}]")
                    time.sleep(wait)
                    continue
            raise


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _msg_to_dict(msg) -> dict:
    d: dict = {"role": msg.role, "content": msg.content or ""}
    if getattr(msg, "tool_calls", None):
        d["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls
        ]
    return d


def _truncate(text: str, limit: int = 3500) -> str:
    if len(text) > limit:
        return text[:limit] + f"\n... [truncated — {len(text):,} chars]"
    return text


def _trim_tools(tools: list[dict]) -> list[dict]:
    out = []
    for t in tools:
        fn = dict(t["function"])
        desc = fn.get("description", "")
        fn["description"] = desc.split("\n")[0].split(". ")[0].rstrip(".") + "."
        params = dict(fn.get("parameters") or {})
        props = {}
        for k, v in params.get("properties", {}).items():
            p = dict(v)
            p.pop("title", None)
            if isinstance(p.get("description"), str):
                p["description"] = p["description"][:100]
            props[k] = p
        if props:
            params = dict(params)
            params["properties"] = props
            params.pop("additionalProperties", None)
            fn["parameters"] = params
        out.append({"type": "function", "function": fn})
    return out


# ── Safe Tool Caller ───────────────────────────────────────────────────────

async def _safe_call(
    session: "ClientSession",
    name: str,
    args: dict,
    state: AgentState,
) -> str:
    if name == "run_shell_command":
        err = _check_command(args.get("command", ""))
        if err:
            state.log(f"  [SAFETY] {err}")
            return err

    if name in ("write_file", "patch_file", "read_file"):
        raw_path = args.get("file_path", "")
        # Resolve relative paths: assume workers write under home if not absolute
        if raw_path and not os.path.isabs(raw_path):
            resolved = os.path.join(_HOME, raw_path)
            args = dict(args)
            args["file_path"] = resolved
        err = _check_path(args["file_path"], write=(name != "read_file"))
        if err:
            state.log(f"  [SAFETY] {err}")
            return err

    try:
        result = await session.call_tool(name, arguments=args)
        raw = "\n".join(c.text for c in result.content if c.type == "text")

        # Track plot artifacts
        if "|||" in raw and not raw.startswith("Error"):
            path_part, _ = raw.split("|||", 1)
            path = path_part.strip()
            if path not in state.artifacts:
                state.artifacts.append(path)
            return "Saved: " + path

        # Track written files
        if name in ("write_file", "patch_file") and not raw.lower().startswith("error"):
            fpath = args.get("file_path", "")
            if fpath and fpath not in state.artifacts:
                state.artifacts.append(fpath)

        return raw
    except Exception as e:
        return f"Tool error: {e}"


# ── Worker ─────────────────────────────────────────────────────────────────

_WORKER_TOOL_SCOPE: dict[str, list[str] | None] = {
    "coder": None,   # all tools
    "general": None,
    "research": [
        "search_web", "fetch_webpage", "screenshot_webpage",
        "run_shell_command", "write_file", "read_file", "list_directory",
    ],
}

_WORKER_SYSTEM = """\
You are an autonomous worker agent with a specific task to complete.
Use your tools methodically. Be efficient and practical.

RULES:
1. Verify every command output before moving to the next step.
2. Never delete existing files or touch system paths.
3. Build things that actually run — test them, fix errors, retry.
4. Keep summaries lean; do not echo large file contents back.
5. When done: report what you built, where it is, and how to run it.
6. You are working within a time-budgeted session — be efficient.
7. ALWAYS use absolute file paths (e.g. /home/ahmad-unibe/myproject/app.py).
   Never use bare filenames like app.py or README.md without a full path.
   Determine home dir with: run_shell_command("echo $HOME")
8. NEVER write files directly into the home directory (~/ or /home/username/).
   Every project MUST live in its own subdirectory.
   Good: /home/ahmad-unibe/my-project/app.py
   Bad:  /home/ahmad-unibe/app.py
   If the supervisor gave you a project directory, use it exactly.
   If not, create one: run_shell_command("mkdir -p ~/autonomous-projects/my-project-name")
9. For data files and reports: write them into the project directory too,
   not into /tmp or bare home. /tmp is only for throwaway intermediates.
"""


async def run_worker(
    task: TaskRecord,
    session: "ClientSession",
    all_tools: list[dict],
    client: OpenAI,
    state: AgentState,
) -> str:
    allowed = _WORKER_TOOL_SCOPE.get(task.worker_type)
    tools = _trim_tools([
        t for t in all_tools
        if allowed is None or t["function"]["name"] in allowed
    ])

    prefix = f"  [#{task.id} {task.worker_type}]"
    messages = [
        {"role": "system", "content": _WORKER_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Time remaining in session: {state.time_remaining_str()}\n\n"
                f"Your task:\n{task.title}\n\n{task.description}"
            ),
        },
    ]

    state.log(f"{prefix} Starting: {task.title[:65]}")

    for i in range(WORKER_MAX_ITERS):
        if state.time_remaining() < 20:
            return "Stopped: time budget exhausted."

        resp = _api_call(client, WORKER_MODEL, messages,
                         tools=tools, temperature=0.3,
                         max_tokens=WORKER_MAX_TOKENS, log_fn=state.log)
        msg = resp.choices[0].message
        messages.append(_msg_to_dict(msg))

        if not msg.tool_calls:
            result = _strip_thinking(msg.content or "Done.")
            state.log(f"{prefix} Finished in {i+1} step(s).")
            return result

        results = []
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            detail = _worker_detail(name, args)
            state.log(f"{prefix} > {name}" + (f"  |  {detail}" if detail else ""))

            raw = await _safe_call(session, name, args, state)
            results.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": _truncate(raw, 3000),
            })

        messages.extend(results)

    return "Reached iteration limit."


def _worker_detail(name: str, args: dict) -> str:
    def _s(v, n=65):
        v = str(v).replace("\n", " ").strip()
        return v[:n] + "..." if len(v) > n else v

    if name == "run_shell_command":
        return _s(args.get("command", ""), 70)
    if name in ("write_file", "patch_file"):
        path = args.get("file_path", "")
        size = len(args.get("content", args.get("new_str", "")))
        return f"{path}  ({size:,} chars)" if size else path
    if name == "read_file":
        return args.get("file_path", "")
    if name == "list_directory":
        return args.get("dir_path", "")
    if name in ("search_web",):
        return _s(args.get("query", ""), 60)
    if name in ("fetch_webpage", "screenshot_webpage"):
        return args.get("url", "")
    if name == "http_request":
        return f"{args.get('method','GET').upper()}  {args.get('url','')}"
    if "data_file_path" in args:
        return os.path.basename(args["data_file_path"])
    return ""


# ── Planner ────────────────────────────────────────────────────────────────

_PLANNER_SYSTEM = """\
You are the Master Autonomous Planner. You manage a time-budgeted AI work session.
You dispatch tasks to workers and generate a final report when time is up.

TOOLS AVAILABLE TO WORKERS (grouped by worker_type):
  coder/general  : search_web, fetch_webpage, screenshot_webpage, run_shell_command,
                   read_file, write_file, patch_file, list_directory, find_in_files,
                   run_background_process, stop_background_process, list_background_processes,
                   http_request
  research       : search_web, fetch_webpage, screenshot_webpage, run_shell_command,
                   write_file, read_file, list_directory

PLANNING RULES:
1. Plan tasks that produce REAL, RUNNABLE, USEFUL output — not just exploration.
2. Dispatch up to 3 tasks at once for parallel execution.
3. Keep tasks atomic: each worker should be able to finish in 10-20 tool calls.
4. Workers have NO memory of previous tasks — include all needed context in description.
5. If time < 2 minutes: stop dispatching, generate report immediately.
6. If all requested work is done, generate report immediately.
7. NEVER ask workers to delete files, access /etc, ~/.ssh, or system paths.
8. Include exact file paths, package names, and commands in task descriptions.
9. DIRECTORY DISCIPLINE — mandatory:
   - Every project MUST have its own directory under the home folder.
   - Decide the project directory in your FIRST dispatch and pass it to ALL workers.
   - Example: if building a weather app, use ~/weather-dashboard/ for every file.
   - Never instruct workers to write files directly into ~/ (home root).
   - Use a clear, lowercase-hyphenated name: ~/qr-generator/, ~/titanic-analysis/, etc.
   - All workers for the same project must receive the SAME project directory path.

AUTONOMOUS MODE IDEAS (no user task given):
  - Build a polished Streamlit or Flask web app or JavaScript or any framework you choose around a useful feature
  - Scrape interesting public data, analyze it, produce charts and a written summary
  - Create a CLI tool or Python library with tests and README
  - Research a technology topic and produce a mini-tutorial with working code examples
  - Come up with your own ideas sometimes 
  Do something genuinely creative and useful. Never just print Hello World.
"""

_DISPATCH_TOOL = {
    "type": "function",
    "function": {
        "name": "dispatch_tasks",
        "description": "Queue one or more tasks for parallel worker execution.",
        "parameters": {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "description": "Short task name."},
                            "description": {"type": "string", "description": "Full instructions for the worker including file paths, goals, dependencies."},
                            "worker_type": {"type": "string", "enum": ["coder", "research", "general"]},
                        },
                        "required": ["title", "description", "worker_type"],
                    },
                }
            },
            "required": ["tasks"],
        },
    },
}

_REPORT_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_report",
        "description": "End the session and produce the final report.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Executive summary of the session."},
                "accomplishments": {"type": "array", "items": {"type": "string"}},
                "how_to_run": {"type": "string", "description": "Exact commands to use/run what was built."},
                "next_steps": {"type": "array", "items": {"type": "string"}, "description": "What could be done next."},
            },
            "required": ["summary", "accomplishments", "how_to_run"],
        },
    },
}


async def planner_tick(
    client: OpenAI,
    state: AgentState,
    worker_results: dict[int, str],
    force_report: bool = False,
) -> dict | None:
    done_parts = []
    for t in state.done():
        r = worker_results.get(t.id, "")
        status_icon = "V" if t.status == "done" else "X"
        done_parts.append(
            f"[{status_icon}] #{t.id} {t.title}\n  Summary: {r[:250]}"
        )

    running_parts = [f"[~] #{t.id} {t.title}" for t in state.running()]
    pending_parts = [f"[ ] #{t.id} {t.title}" for t in state.pending()]

    time_warn = ""
    if force_report:
        remaining_s = int(state.time_remaining())
        time_warn = (
            f"\n\n!!! TIME CRITICAL: Only {state.time_remaining_str()} remaining. "
            f"You MUST call generate_report RIGHT NOW. "
            f"Do NOT dispatch any new tasks. Summarise everything that was accomplished.\n"
        )

    ctx = (
        f"GOAL: {state.goal}\n"
        f"TIME REMAINING: {state.time_remaining_str()}  |  {state.elapsed_str()}\n"
        f"ARTIFACTS SO FAR: {', '.join(state.artifacts) or 'none'}\n\n"
        f"COMPLETED ({len(state.done())}):\n" + ("\n".join(done_parts) or "none") + "\n\n"
        f"RUNNING ({len(state.running())}):\n" + ("\n".join(running_parts) or "none") + "\n\n"
        f"PENDING ({len(state.pending())}):\n" + ("\n".join(pending_parts) or "none")
        + time_warn
    )

    messages = [
        {"role": "system", "content": _PLANNER_SYSTEM},
        {"role": "user", "content": ctx},
    ]

    # When time is low: only offer generate_report — physically remove dispatch tool
    available_tools = [_REPORT_TOOL] if force_report else [_DISPATCH_TOOL, _REPORT_TOOL]

    resp = _api_call(client, PLANNER_MODEL, messages,
                     tools=available_tools,
                     temperature=0.5, max_tokens=PLANNER_MAX_TOKENS,
                     log_fn=state.log)
    msg = resp.choices[0].message

    if not msg.tool_calls:
        if force_report:
            # Model didn't call the tool — build report from state directly
            state.log("  [planner] No tool call on force_report — auto-building report.")
            return None  # caller will use auto-report
        return None

    for tc in msg.tool_calls:
        name = tc.function.name
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}

        if name == "dispatch_tasks":
            if force_report:
                state.log("  [planner] Ignoring dispatch during wrap-up phase.")
                continue
            for tdef in args.get("tasks", []):
                task = TaskRecord(
                    id=state.next_id(),
                    title=tdef.get("title", "Task"),
                    description=tdef.get("description", ""),
                    worker_type=tdef.get("worker_type", "general"),
                )
                state.tasks.append(task)
                state.log(f"  [planner] Queued #{task.id}: {task.title[:60]}")
            return None

        if name == "generate_report":
            return args

    return None


# ── Orchestration Loop ─────────────────────────────────────────────────────

async def autonomous_loop(
    client: OpenAI,
    session: "ClientSession",
    all_tools: list[dict],
    state: AgentState,
) -> None:
    running_tasks: dict[int, asyncio.Task] = {}
    worker_results: dict[int, str] = {}
    last_tick = 0.0
    report: dict | None = None

    state.log(f"[Master] Session started — goal: {state.goal[:80]}")
    state.log(f"[Master] Budget: {state.time_remaining_str()}  |  {len(all_tools)} tools available")

    while state.time_remaining() > 5:
        # 1. Reap completed workers
        for tid in list(running_tasks):
            fut = running_tasks[tid]
            if fut.done():
                task = next((t for t in state.tasks if t.id == tid), None)
                if task:
                    try:
                        result = fut.result()
                        task.status = "done"
                        task.result_summary = result[:500]
                        worker_results[tid] = result
                        state.log(f"  [#{tid}] Done: {task.title[:55]}")
                    except Exception as e:
                        task.status = "failed"
                        task.result_summary = str(e)
                        worker_results[tid] = f"Error: {e}"
                        state.log(f"  [#{tid}] Failed: {e}")
                    task.finished_at = time.time()
                del running_tasks[tid]

        # 2. Start pending tasks up to MAX_PARALLEL_WORKERS
        in_wrap_up = state.time_remaining() < WRAP_UP_SECONDS
        if not in_wrap_up:
            for task in state.pending():
                if len(running_tasks) >= MAX_PARALLEL_WORKERS:
                    break
                task.status = "running"
                task.started_at = time.time()
                fut = asyncio.create_task(
                    run_worker(task, session, all_tools, client, state)
                )
                running_tasks[task.id] = fut
        elif in_wrap_up and running_tasks:
            # Grace period: let running workers finish, but stop starting new ones
            pass

        # 3. Decide if planner should tick
        now = time.time()
        idle = (len(running_tasks) == 0)
        no_pending = (len(state.pending()) == 0)
        tick_due = (now - last_tick) >= PLANNER_TICK_SECONDS
        # force_report fires whenever time is low — does NOT require idle
        force_report = state.time_remaining() < WRAP_UP_SECONDS

        should_tick = (
            force_report          # always tick when time is low
            or (tick_due and no_pending)
            or (idle and no_pending and last_tick > 0)
            or last_tick == 0
        )

        if should_tick:
            last_tick = now
            state.log(f"\n[Master] Planner tick — {state.time_remaining_str()} remaining...")
            report = await planner_tick(client, state, worker_results, force_report=force_report)
            if report:
                state.log("[Master] Planner issued final report.")
                break

        await asyncio.sleep(4)

    # Cancel remaining workers
    if running_tasks:
        state.log(f"[Master] Cancelling {len(running_tasks)} running worker(s)...")
        for fut in running_tasks.values():
            fut.cancel()
        await asyncio.gather(*running_tasks.values(), return_exceptions=True)

    # Auto-report if planner never called generate_report
    if not report:
        state.log("[Master] Time expired — auto-generating report.")
        done_tasks = state.done()
        report = {
            "summary": (
                f"Autonomous session completed ({state.elapsed_str()}). "
                f"{len(done_tasks)} task(s) finished out of {len(state.tasks)} total."
            ),
            "accomplishments": [
                f"{t.title}: {t.result_summary[:200]}"
                for t in done_tasks if t.status == "done"
            ],
            "how_to_run": "See individual task summaries above. Check ~/mcp-server-example/screenshots/ and ~/plots/ for any generated files.",
            "next_steps": ["Review generated artifacts.", "Extend the project with more time."],
        }

    _print_report(report, state)


# ── Report Printer ─────────────────────────────────────────────────────────

def _build_report_text(report: dict, state: AgentState) -> str:
    """Render the report as a plain-text / Markdown string."""
    lines: list[str] = []
    W = 68
    lines.append("=" * W)
    lines.append("  AUTONOMOUS SESSION REPORT")
    lines.append("=" * W)
    lines.append("")
    lines.append(f"  {report.get('summary', '')}")
    lines.append("")

    acc = report.get("accomplishments", [])
    if acc:
        lines.append("  ACCOMPLISHMENTS:")
        for a in acc:
            for ln in textwrap.wrap(str(a), 62):
                lines.append(f"    * {ln}")
        lines.append("")

    arts = list(dict.fromkeys(report.get("artifacts", []) + state.artifacts))
    if arts:
        lines.append("  ARTIFACTS CREATED:")
        for a in arts:
            lines.append(f"    [file] {a}")
        lines.append("")

    htr = report.get("how_to_run", "")
    if htr:
        lines.append("  HOW TO USE / RUN:")
        for ln in htr.splitlines():
            for wrapped in textwrap.wrap(ln.strip(), 62) or [""]:
                lines.append(f"    {wrapped}")
        lines.append("")

    nxt = report.get("next_steps", [])
    if nxt:
        lines.append("  SUGGESTED NEXT STEPS:")
        for n in nxt:
            lines.append(f"    -> {n}")
        lines.append("")

    lines.append(f"  FULL SESSION LOG ({len(state.log_lines)} entries):")
    for ln in state.log_lines:
        lines.append(f"    {ln}")
    lines.append("=" * W)
    return "\n".join(lines)


def _save_report(report: dict, state: AgentState) -> str:
    """Save report + full log to REPORTS_DIR. Returns the saved file path."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r"[^\w]+", "_", state.goal[:40]).strip("_").lower()
    filename = f"{ts}_{slug}.md"
    filepath = os.path.join(REPORTS_DIR, filename)
    text = _build_report_text(report, state)
    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write(text)
    return filepath


def _print_report(report: dict, state: AgentState) -> None:
    # Print to terminal (last 35 log lines to keep it readable)
    arts = list(dict.fromkeys(report.get("artifacts", []) + state.artifacts))
    W = 68
    lines = _build_report_text({**report, "artifacts": arts}, state).splitlines()
    # Show everything except the full log block in the terminal
    in_log = False
    for ln in lines:
        if "FULL SESSION LOG" in ln:
            in_log = True
        if not in_log:
            print(ln)

    # Print last 35 log lines in terminal
    print(f"\n  SESSION LOG (last 35 of {len(state.log_lines)}):")
    for ln in state.log_lines[-35:]:
        print(f"    {ln}")
    print("=" * W)

    # Save full report + log to disk
    saved_path = _save_report({**report, "artifacts": arts}, state)
    print(f"\n  >> Report saved to: {saved_path}")
    print(f"  >> Read it with:    cat \"{saved_path}\"")
    print()


# ── Entry Point ────────────────────────────────────────────────────────────

def _parse_time(s: str) -> int:
    s = s.strip().lower()
    total = 0
    for val, unit in re.findall(r"(\d+)\s*([hms]?)", s):
        v = int(val)
        if unit == "h":
            total += v * 3600
        elif unit == "m":
            total += v * 60
        elif unit == "s":
            total += v
        else:
            total += v
    return total or 3600


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autonomous time-budgeted AI agent with MCP tools"
    )
    parser.add_argument(
        "--time", "-t", default="1h",
        help="Time budget e.g. 30m, 1h, 2h30m (default: 1h)",
    )
    parser.add_argument(
        "--task", "-g", default="",
        help="Goal for the session. Omit for fully autonomous creative mode.",
    )
    args = parser.parse_args()

    seconds = _parse_time(args.time)
    goal = args.task.strip() or (
        "AUTONOMOUS MODE: You have complete creative freedom for this session. "
        "Decide what to build entirely on your own — a web app, a CLI tool, a data pipeline, "
        "an API, a game, a utility, anything you find genuinely interesting or useful. "
        "Do NOT default to GitHub trending scrapers or weather apps. Surprise us. "
        "Whatever you build must actually run. Create a dedicated project directory, "
        "write clean code, test it, fix errors, and leave clear launch instructions."
    )

    api_key = (
        os.environ.get("API_KEY") or os.environ.get("api_key") or ""
    ).strip()
    api_base_url = (
        os.environ.get("API_BASE_URL") or os.environ.get("api_base_url") or
        "https://api.openai.com/v1"
    ).strip()
    if not api_key:
        print("[Error] api_key not set. Add a .env file in the project root")
        print("  Example .env:")
        print("    api_base_url=https://api.openai.com/v1")
        print("    api_key=sk-...")
        sys.exit(1)

    openai_client = OpenAI(base_url=api_base_url, api_key=api_key)

    W = 68
    print("\n" + "=" * W)
    print("  AUTONOMOUS MASTER AGENT")
    print("=" * W)
    print(f"  Planner model   : {PLANNER_MODEL}")
    print(f"  Worker model    : {WORKER_MODEL}")
    print(f"  API base URL    : {api_base_url}")
    print(f"  Time budget     : {args.time}  ({seconds}s)")
    print(f"  Max workers     : {MAX_PARALLEL_WORKERS}")
    print(f"  Goal            : {goal[:70]}{'...' if len(goal) > 70 else ''}")
    print("=" * W + "\n")

    state = AgentState(deadline=time.time() + seconds, goal=goal)

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[MCP_SERVER_SCRIPT],
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_resp = await session.list_tools()
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
                print(f"  {len(tools)} MCP tools loaded.\n")

                try:
                    await autonomous_loop(openai_client, session, tools, state)
                except KeyboardInterrupt:
                    print("\n[Autonomous] Interrupted — printing partial report.")
                    _print_report(
                        {
                            "summary": f"Session interrupted by user. {state.elapsed_str()}.",
                            "accomplishments": [
                                f"{t.title}: {t.result_summary[:150]}"
                                for t in state.done() if t.status == "done"
                            ],
                            "how_to_run": "Session ended early. Check partially created files.",
                        },
                        state,
                    )
    except* Exception as eg:
        # Suppress BrokenResourceError from anyio when cancelled workers close the MCP pipe.
        # The report is already saved at this point — this is purely cleanup noise.
        non_broken = [
            e for e in eg.exceptions
            if "BrokenResourceError" not in type(e).__name__
            and "BrokenResourceError" not in str(e)
        ]
        if non_broken:
            raise ExceptionGroup("unhandled errors", non_broken)


if __name__ == "__main__":
    asyncio.run(main())
