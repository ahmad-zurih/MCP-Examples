"""
autonomous_agent/mcp_server.py
================================
A lean MCP server that exposes system and internet tools only.
Designed for the autonomous time-budgeted agent — deliberately excludes
data-science tools so the model stays focused on software engineering tasks.

Tools exposed:
  System  : run_shell_command, read_file, write_file, patch_file,
            list_directory, find_in_files, run_background_process,
            stop_background_process, list_background_processes, http_request
  Internet: fetch_webpage, search_web, screenshot_webpage
"""

import logging

from mcp.server.fastmcp import FastMCP

from system_tools import (
    find_in_files_impl,
    http_request_impl,
    list_background_processes_impl,
    list_directory_impl,
    patch_file_impl,
    read_file_impl,
    run_background_process_impl,
    run_shell_command_impl,
    stop_background_process_impl,
    write_file_impl,
)

from web_tools import (
    fetch_webpage_impl,
    screenshot_webpage_impl,
    search_web_impl,
)

logging.basicConfig(level=logging.ERROR)
logging.getLogger("mcp").setLevel(logging.ERROR)

mcp = FastMCP("Autonomous Agent MCP Server")


# ── System Tools ──────────────────────────────────────────────────────────────

@mcp.tool()
def run_shell_command(command: str, working_dir: str | None = None) -> str:
    """
    Execute a shell command and return its stdout/stderr output.
    Use this to create directories, install packages, run scripts, check output, etc.
    working_dir: optional absolute path to run the command in (defaults to home directory).
    """
    return run_shell_command_impl(command, working_dir)


@mcp.tool()
def read_file(file_path: str) -> str:
    """
    Read and return the full text content of a file.
    Use this to inspect existing code, configs, logs, or any text file before editing.
    """
    return read_file_impl(file_path)


@mcp.tool()
def write_file(file_path: str, content: str) -> str:
    """
    Write (or overwrite) a file with the given content.
    Parent directories are created automatically.
    Always write the COMPLETE file content — this replaces the entire file.
    """
    return write_file_impl(file_path, content)


@mcp.tool()
def patch_file(file_path: str, old_str: str, new_str: str) -> str:
    """
    Find the FIRST occurrence of old_str in a file and replace it with new_str.
    PREFER this over write_file for editing existing files — only send the changed part.
    old_str must match EXACTLY, including whitespace and indentation.
    Returns the line number of the edit on success, or an error with a file preview.
    """
    return patch_file_impl(file_path, old_str, new_str)


@mcp.tool()
def list_directory(dir_path: str) -> str:
    """
    List the contents of a directory (one level deep).
    Use this to explore project structure before reading or writing files.
    """
    return list_directory_impl(dir_path)


@mcp.tool()
def find_in_files(pattern: str, directory: str, file_glob: str = "*.py") -> str:
    """
    Search file contents for a regex pattern across a directory tree (like grep -rn).
    Returns matching lines with file:line references (max 60 results).
    pattern: a regular expression (e.g. "def train", "import pandas", "TODO")
    directory: root directory to search from
    file_glob: filter by filename pattern, e.g. "*.py", "*.ts", "*.json", "*" for all files
    """
    return find_in_files_impl(pattern, directory, file_glob)


@mcp.tool()
def run_background_process(command: str, label: str, working_dir: str | None = None) -> str:
    """
    Start a long-running process (e.g. a web server) in the background and track it by label.
    Use a short descriptive label like "streamlit-app" or "flask-server".
    After starting, use http_request to verify it is responding.
    working_dir: optional absolute path to run the command in.
    """
    return run_background_process_impl(command, label, working_dir)


@mcp.tool()
def stop_background_process(label: str) -> str:
    """
    Stop a background process previously started with run_background_process.
    label: the label you used when starting the process.
    """
    return stop_background_process_impl(label)


@mcp.tool()
def list_background_processes() -> str:
    """List all background processes currently tracked (with their status and PIDs)."""
    return list_background_processes_impl()


@mcp.tool()
def http_request(url: str, method: str = "GET", body: str = "") -> str:
    """
    Make an HTTP request and return the status code + response body (up to 4 KB).
    Use this to test running web servers after starting them with run_background_process.
    method: GET, POST, PUT, DELETE, PATCH
    body: optional JSON string for POST/PUT requests.
    """
    return http_request_impl(url, method, body)


# ── Internet / Web Tools ──────────────────────────────────────────────────────

@mcp.tool()
def search_web(query: str, max_results: int = 5) -> str:
    """
    Search the internet using DuckDuckGo and return titles, URLs, and text snippets.
    No API key required. Use this to find documentation, discover libraries, look up
    best practices, or research any topic before starting a coding task.
    max_results: number of results to return (1-10, default 5).
    """
    return search_web_impl(query, max_results)


@mcp.tool()
def fetch_webpage(url: str) -> str:
    """
    Fetch a webpage and return structured content: title, meta description, navigation,
    headings, CSS color palette, font families, and main page text (up to 4000 chars).
    Use this to research a site before cloning its design, extract information, or
    understand its structure.
    """
    return fetch_webpage_impl(url)


@mcp.tool()
def screenshot_webpage(url: str, save_path: str | None = None) -> str:
    """
    Take a 1440x900 screenshot of a webpage using headless Chromium and save it as a PNG.
    Returns the file path of the saved screenshot.
    save_path: optional absolute path for the PNG; auto-generated if omitted.
    Requires playwright: pip install playwright && playwright install chromium
    """
    return screenshot_webpage_impl(url, save_path)


if __name__ == "__main__":
    mcp.run(transport="stdio")
