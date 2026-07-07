"""
Agentic Multi-Agent System (MAS) for the AI Visualization Engine.
Implements a Supervisor-Worker pattern to route tasks and manage tool hallucinations.
"""

import asyncio
import sys
import threading
import traceback
from typing import Any, Callable

import ollama
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

from plot_data import get_all_columns_summary_impl
from viz_config import (
    AGENT_TOOLS,
    CODER_PROMPT,
    CODER_SUPERVISOR_PROMPT,
    INTERACTIVE_PROMPT,
    STATIC_PROMPT,
    STATS_PROMPT,
    SUPERVISOR_PROMPT,
    PlotArtifact,
    StatsArtifact,
    VizAnalysisResult,
    get_tool_label,
)


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
    # Viz / stats tools — show filename + key column args
    if "data_file_path" in args:
        fname = os.path.basename(args["data_file_path"])
        col_args = [v for k, v in args.items()
                    if k not in ("data_file_path", "python_code", "plot_filename_keyword")
                    and isinstance(v, str) and v]
        extra = "  ".join(col_args[:3])
        return f"{fname}" + (f"  |  {_s(extra, 45)}" if extra else "")
    return ""

WORKER_PROMPTS = {
    "interactive": INTERACTIVE_PROMPT,
    "static": STATIC_PROMPT,
    "stats": STATS_PROMPT,
    "coder": CODER_PROMPT,
}

DELEGATE_TASK_TOOL = {
    "type": "function",
    "function": {
        "name": "delegate_task",
        "description": "Delegate a sub-task to a specialist agent.",
        "parameters": {
            "type": "object",
            "properties": {
                "agent_role": {
                    "type": "string",
                    "enum": ["interactive", "static", "stats", "coder"],
                    "description": "The specific agent to assign the task to."
                },
                "task_instruction": {
                    "type": "string",
                    "description": "Clear instructions on what the agent should do."
                }
            },
            "required": ["agent_role", "task_instruction"]
        }
    }
}



DELEGATE_CODING_TASK_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "delegate_coding_task",
        "description": "Delegate one atomic coding subtask to a worker agent.",
        "parameters": {
            "type": "object",
            "properties": {
                "subtask": {
                    "type": "string",
                    "description": "The exact task for the worker. Include file paths and specific goals."
                },
                "context": {
                    "type": "string",
                    "description": "Project context the worker needs: directory layout, already-created files, env paths."
                },
            },
            "required": ["subtask", "context"],
        },
    },
}

async def _get_mcp_tools(session: ClientSession, allowed_names: list[str] | None = None) -> list[dict[str, Any]]:
    tool_list_response = await session.list_tools()
    ollama_tools: list[dict[str, Any]] = []
    
    for tool in tool_list_response.tools:
        if allowed_names is None or tool.name in allowed_names:
            ollama_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    },
                }
            )
            
    return ollama_tools



async def _run_coder_worker(
    subtask: str,
    context: str,
    model_name: str,
    mcp_server_script: str,
    global_plots: list[PlotArtifact],
    global_stats: list[StatsArtifact],
    global_logs: list[tuple[str, str]],
    log_callback: Callable[[str, str], None] | None,
    cancel_event: threading.Event | None,
) -> str:
    """Run one atomic coding subtask in its own fresh MCP session (small context window)."""
    def _log(l_type: str, msg: str):
        global_logs.append((l_type, msg))
        if log_callback:
            log_callback(l_type, msg)

    full_task = f"Supervisor context:\n{context}\n\nYour task:\n{subtask}" if context else subtask
    server_params = StdioServerParameters(command="python3", args=[mcp_server_script])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await _run_worker_agent_inner(
                session=session,
                agent_role="coder",
                task_instruction=full_task,
                data_file_path="",
                model_name=model_name,
                global_plots=global_plots,
                global_stats=global_stats,
                global_logs=global_logs,
                max_iterations=8,
                log_callback=log_callback,
                cancel_event=cancel_event,
                _log=_log,
            )


async def _run_coder_supervisor(
    task_instruction: str,
    model_name: str,
    mcp_server_script: str,
    global_plots: list[PlotArtifact],
    global_stats: list[StatsArtifact],
    global_logs: list[tuple[str, str]],
    log_callback: Callable[[str, str], None] | None,
    cancel_event: threading.Event | None,
) -> str:
    """
    Supervisor agent that breaks a coding project into atomic subtasks,
    delegates each to a fresh worker (small context), tracks progress, and returns a summary.
    """
    def _log(l_type: str, msg: str):
        global_logs.append((l_type, msg))
        if log_callback:
            log_callback(l_type, msg)

    _log("info", "Coder supervisor: analysing project and planning subtasks...")

    supervisor_messages = [
        {"role": "system", "content": CODER_SUPERVISOR_PROMPT},
        {"role": "user", "content": task_instruction},
    ]
    completed: list[str] = []

    for round_num in range(20):
        if cancel_event and cancel_event.is_set():
            return "Coder supervisor cancelled."
        try:
            response = ollama.chat(
                model=model_name,
                messages=supervisor_messages,
                tools=[DELEGATE_CODING_TASK_TOOL],
            )
        except Exception as e:
            _log("error", f"Coder supervisor LLM error: {e}")
            return f"Supervisor failed: {e}"

        msg = response["message"]
        supervisor_messages.append(msg)

        if not msg.get("tool_calls"):
            _log("info", f"Coder supervisor done ({len(completed)} subtask(s) completed).")
            return msg.get("content", "Project completed.")

        for tool_call in msg["tool_calls"]:
            if tool_call["function"]["name"] != "delegate_coding_task":
                continue
            args = tool_call["function"].get("arguments") or {}
            if not isinstance(args, dict):
                args = {}
            subtask = args.get("subtask", "")
            context = args.get("context", "")

            _log("info", f"  [subtask {round_num + 1}] {subtask[:80]}")

            worker_result = await _run_coder_worker(
                subtask=subtask,
                context=context,
                model_name=model_name,
                mcp_server_script=mcp_server_script,
                global_plots=global_plots,
                global_stats=global_stats,
                global_logs=global_logs,
                log_callback=log_callback,
                cancel_event=cancel_event,
            )

            completed.append(subtask[:70])
            progress = "\n".join(f"  v {t}" for t in completed[-10:])

            supervisor_messages.append({
                "role": "tool",
                "content": (
                    f"Worker completed the subtask.\n"
                    f"Report: {worker_result}\n\n"
                    f"Completed so far ({len(completed)} subtask(s)):\n{progress}"
                ),
            })

    return "Coder supervisor reached max rounds. Project may be incomplete — check the last worker report."

async def _run_worker_agent(
    agent_role: str,
    task_instruction: str,
    data_file_path: str,
    model_name: str,
    mcp_server_script: str,
    global_plots: list[PlotArtifact],
    global_stats: list[StatsArtifact],
    global_logs: list[tuple[str, str]],
    max_iterations: int = 6,
    log_callback: Callable[[str, str], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> str:
    def _log(l_type: str, msg: str):
        global_logs.append((l_type, msg))
        if log_callback:
            log_callback(l_type, msg)

    # Coder tasks use a dedicated supervisor+worker hierarchy
    # so each atomic step gets a fresh small context window.
    if agent_role == "coder":
        return await _run_coder_supervisor(
            task_instruction=task_instruction,
            model_name=model_name,
            mcp_server_script=mcp_server_script,
            global_plots=global_plots,
            global_stats=global_stats,
            global_logs=global_logs,
            log_callback=log_callback,
            cancel_event=cancel_event,
        )

    server_params = StdioServerParameters(
        command="python3",
        args=[mcp_server_script],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await _run_worker_agent_inner(
                session=session,
                agent_role=agent_role,
                task_instruction=task_instruction,
                data_file_path=data_file_path,
                model_name=model_name,
                global_plots=global_plots,
                global_stats=global_stats,
                global_logs=global_logs,
                max_iterations=max_iterations,
                log_callback=log_callback,
                cancel_event=cancel_event,
                _log=_log,
            )


async def _run_worker_agent_inner(
    session: ClientSession,
    agent_role: str,
    task_instruction: str,
    data_file_path: str,
    model_name: str,
    global_plots: list[PlotArtifact],
    global_stats: list[StatsArtifact],
    global_logs: list[tuple[str, str]],
    max_iterations: int,
    log_callback: Callable[[str, str], None] | None,
    cancel_event: threading.Event | None,
    _log: Callable[[str, str], None],
) -> str:
    allowed_tools = AGENT_TOOLS.get(agent_role, [])
    tools = await _get_mcp_tools(session, allowed_names=allowed_tools)

    system_prompt = WORKER_PROMPTS.get(agent_role, "You are a helpful assistant.")

    # Pre-compute the dataset schema for viz/stats agents.
    # Coder agent works on the filesystem directly and does not need dataset context.
    if agent_role == "coder" or not data_file_path:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_instruction},
        ]
    else:
        schema = get_all_columns_summary_impl(data_file_path)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Dataset schema (already loaded — use your tools to analyze it):\n{schema}\n\n"
                    f"Task: {task_instruction}"
                ),
            },
        ]

    _log("info", f"Supervisor delegated task to '{agent_role}' agent.")

    STATS_TOOLS = {"run_correlation", "run_group_comparison", "run_linear_regression", "rank_target_correlations"}

    for iteration in range(max_iterations):
        if cancel_event and cancel_event.is_set():
            _log("warning", f"Worker '{agent_role}' cancelled by user.")
            return f"Worker '{agent_role}' was cancelled."
        if agent_role == "coder" and iteration > 0 and iteration % 5 == 0:
            _log("info", f"Coder agent: iteration {iteration}/{max_iterations}...")
        try:
            response = ollama.chat(
                model=model_name,
                messages=messages,
                tools=tools,
            )
            messages.append(response["message"])
        except Exception as e:
            error_msg = f"Worker '{agent_role}' failed to communicate with Ollama: {e}"
            _log("error", error_msg)
            return error_msg

        if not response["message"].get("tool_calls"):
            worker_summary = response["message"].get("content", f"{agent_role} agent completed task silently.")
            _log("info", f"Worker '{agent_role}' finished task successfully.")
            return worker_summary

        tool_calls = response["message"]["tool_calls"]
        mcp_tool_results: list[dict[str, Any]] = []

        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            tool_args = tool_call["function"].get("arguments")
            if not isinstance(tool_args, dict):
                tool_args = {}
            # System/coder tools don't take data_file_path
            if agent_role != "coder":
                tool_args["data_file_path"] = data_file_path
            _detail = _format_tool_call(tool_name, tool_args)
            _log("info", f"  > {tool_name}" + (f"  |  {_detail}" if _detail else ""))

            try:
                result = await session.call_tool(tool_name, arguments=tool_args)
                
                tool_output_raw = ""
                if result.content and isinstance(result.content[0], types.TextContent):
                    tool_output_raw = result.content[0].text
                
                is_error = tool_output_raw.strip().startswith("Error")

                # Route by response format:
                # - Plot tools always return "path|||code"
                # - Stats tools return plain text with an embedded ```python code block
                # - Data summary tools return plain text
                is_plot_tool = "|||" in tool_output_raw

                if not is_plot_tool:
                    if is_error:
                        _log("warning", f"Worker '{agent_role}' stat tool '{tool_name}' failed. Retrying...")
                        mcp_tool_results.append({
                            "role": "tool",
                            "content": f"Execution Error: {tool_output_raw.strip()}\nPlease correct your code/parameters and try again.",
                        })
                    else:
                        # For stats tools, capture result + code as a StatsArtifact for the UI.
                        if tool_name in STATS_TOOLS:
                            result_text, code_snippet = _extract_stats_code(tool_output_raw)
                            global_stats.append({
                                "title": get_tool_label(tool_name),
                                "result": result_text,
                                "code": code_snippet,
                            })
                        mcp_tool_results.append({
                            "role": "tool",
                            "content": tool_output_raw,
                        })

                else:
                    # Plot tool — extract path+code, store artifact, return generic ack.
                    if not is_error:
                        path_part, code_part = tool_output_raw.split("|||", 1)
                        global_plots.append({
                            "path": path_part.strip(),
                            "code": code_part.strip(),
                            "tool_name": tool_name,
                        })
                        # Generic message — never expose internal file paths to the model.
                        mcp_tool_results.append({
                            "role": "tool",
                            "content": "Plot generated successfully. It will be displayed to the user.",
                        })
                    else:
                        _log("warning", f"Worker '{agent_role}' plot tool '{tool_name}' failed: {tool_output_raw.strip()}. Retrying...")
                        mcp_tool_results.append({
                            "role": "tool",
                            "content": f"Execution Error: {tool_output_raw.strip()}\nPlease correct your code or parameters and try again.",
                        })

            except Exception as e:
                error_msg = f"Tool '{tool_name}' crashed: {str(e)}"
                _log("error", error_msg)
                mcp_tool_results.append({
                    "role": "tool",
                    "content": error_msg,
                })

        messages.extend(mcp_tool_results)

    _log("error", f"Worker '{agent_role}' reached maximum iterations without resolving issues.")
    return f"Worker '{agent_role}' reached maximum iterations and terminated. Last known state appended."


def _extract_stats_code(tool_output: str) -> tuple[str, str]:
    """
    Splits a stats tool output into (result_text, code_snippet).
    Stats tools embed a ```python ... ``` block at the end of their output.
    Returns the markdown table/summary and the code block separately.
    """
    import re
    match = re.search(r"```python\n(.*?)```", tool_output, re.DOTALL)
    if match:
        code = match.group(1).strip()
        result_text = tool_output[:match.start()].strip()
    else:
        code = ""
        result_text = tool_output.strip()
    return result_text, code


async def run_analysis(
    messages: list[dict[str, Any]],
    data_file_path: str,
    model_name: str,
    mcp_server_script: str,
    max_iterations: int = 7,
    log_callback: Callable[[str, str], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> VizAnalysisResult:
    def _log(l_type: str, msg: str):
        logs.append((l_type, msg))
        if log_callback:
            log_callback(l_type, msg)

    plot_results: list[PlotArtifact] = []
    stats_results: list[StatsArtifact] = []
    logs: list[tuple[str, str]] = []

    supervisor_messages = [{"role": "system", "content": SUPERVISOR_PROMPT}] + messages

    try:
        for iteration in range(max_iterations):
            if cancel_event and cancel_event.is_set():
                _log("warning", "Analysis cancelled by user.")
                break
            try:
                response = ollama.chat(
                    model=model_name,
                    messages=supervisor_messages,
                    tools=[DELEGATE_TASK_TOOL],
                )
                supervisor_messages.append(response["message"])
            except Exception as e:
                _log("error", f"Error communicating with Supervisor: {e}")
                break

            if not response["message"].get("tool_calls"):
                _log("info", "Supervisor completed task delegation and synthesized final summary.")
                break

            # Collect all valid delegate_task calls for this iteration
            pending: list[tuple[str, str]] = []  # (agent_role, task_instruction)
            invalid_results: list[dict[str, Any]] = []

            for tool_call in response["message"]["tool_calls"]:
                if tool_call["function"]["name"] != "delegate_task":
                    continue
                agent_role = tool_call["function"]["arguments"].get("agent_role")
                task_instruction = tool_call["function"]["arguments"].get("task_instruction")
                if agent_role not in WORKER_PROMPTS:
                    invalid_results.append({
                        "role": "tool",
                        "content": f"Error: Agent role '{agent_role}' does not exist.",
                    })
                else:
                    pending.append((agent_role, task_instruction))

            # Run all valid workers concurrently — each opens its own MCP subprocess
            if pending:
                worker_coroutines = [
                    _run_worker_agent(
                        agent_role=role,
                        task_instruction=instruction,
                        data_file_path=data_file_path,
                        model_name=model_name,
                        mcp_server_script=mcp_server_script,
                        global_plots=plot_results,
                        global_stats=stats_results,
                        global_logs=logs,
                        max_iterations=30 if role == "coder" else 10,
                        log_callback=log_callback,
                        cancel_event=cancel_event,
                    )
                    for role, instruction in pending
                ]
                worker_outputs = await asyncio.gather(*worker_coroutines, return_exceptions=True)
            else:
                worker_outputs = []

            supervisor_tool_results = list(invalid_results)
            for (role, _), output in zip(pending, worker_outputs):
                if isinstance(output, Exception):
                    _log("error", f"Worker '{role}' raised an exception: {output}")
                    content = f"Error from {role}: {output}"
                else:
                    content = f"Results from {role}:\n{output}"
                supervisor_tool_results.append({"role": "tool", "content": content})

            supervisor_messages.extend(supervisor_tool_results)

    except Exception as e:
        _log("error", f"Fatal error in MAS session: {str(e)}\n{traceback.format_exc()}")

    summary = ""
    if supervisor_messages and supervisor_messages[-1].get("role") == "assistant":
        summary = supervisor_messages[-1].get("content", "")
        
    if not summary:
        summary = "Analysis complete. Please review the generated visualizations below."

    final_logs: list[tuple[Any, str]] = []
    for log_type, msg in logs:
        if log_type in ("info", "warning", "error"):
            final_logs.append((log_type, msg))
        else:
            final_logs.append(("info", msg))

    return {
        "summary": summary,
        "plots": plot_results,
        "stats": stats_results,
        "logs": final_logs
    }