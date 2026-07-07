# Autonomous Agent

A time-budgeted AI agent that works autonomously — given a goal and a time limit,
it plans, builds, tests, and reports, all on its own.

---

## How It Works

```
┌─────────────────────────────────────────────────┐
│              Master Planner (GPUStack)           │
│  - Holds the clock and the goal                  │
│  - Breaks work into atomic subtasks              │
│  - Dispatches up to 3 workers in parallel        │
│  - Stops dispatching with 3 min left             │
│  - Calls generate_report when time is up         │
└──────────┬──────────────┬──────────────┬─────────┘
           │              │              │
     Worker #1       Worker #2      Worker #3
   (fresh context) (fresh context) (fresh context)
        │                │               │
        └────────────────┴───────────────┘
                         │
                   MCP Server (13 tools)
          system tools + internet tools (no data science)
```

Each worker gets a completely fresh context window — no shared memory.
The supervisor includes all necessary context in each task description.

---

## Usage

```bash
cd autonomous_agent

# Autonomous mode — 1 hour, model decides what to build
python mcp_client_autonomous.py

# With a time limit
python mcp_client_autonomous.py --time 30m

# With a specific goal
python mcp_client_autonomous.py --time 45m --task "build a CLI password manager in Python"

# Longer session
python mcp_client_autonomous.py -t 2h -g "research and build a web scraper with a Flask dashboard"
```

**Time formats:** `30m`, `1h`, `2h30m`, `90m`, `3600s`

---

## Tools Available (13 total)

### System Tools
| Tool | Description |
|------|-------------|
| `run_shell_command` | Execute any bash command |
| `read_file` | Read a file from disk |
| `write_file` | Write/create a file (creates parent dirs) |
| `patch_file` | Surgically replace a string in an existing file |
| `list_directory` | List directory contents |
| `find_in_files` | Regex search across a directory tree (like grep -rn) |
| `run_background_process` | Start a server or daemon, tracked by label |
| `stop_background_process` | Stop a tracked background process |
| `list_background_processes` | Show what is running |
| `http_request` | Make HTTP requests to test running servers |

### Internet Tools
| Tool | Description |
|------|-------------|
| `search_web` | DuckDuckGo search — no API key needed |
| `fetch_webpage` | Fetch a URL: title, headings, CSS colors/fonts, page text |
| `screenshot_webpage` | Headless Chromium screenshot → saved as PNG |

---

## Safety Rules

The agent enforces hard safety rules at the code level — model instructions cannot override them:

- **Blocked commands:** `rm -rf`, `sudo`, pipe-to-shell (`| bash`), `shutdown`, `mkfs`, `dd if=`, fork bombs, and more
- **Blocked read paths:** `~/.ssh/`, `~/.gnupg/`, `/etc/shadow`, `~/.aws/credentials`
- **Blocked write paths:** `~/.ssh/`, `/etc/`, `/sys/`, `/proc/`, `/boot/`, `~/.aws/`
- **Write scope:** Only `~/` subdirectories and `/tmp` — never system paths
- **No home root writes:** Files must be written into a named subdirectory (e.g. `~/my-project/app.py`), never directly into `~/`

---

## Output

Every session saves a full Markdown report to:
```
~/mcp-server-example/session-reports/YYYYMMDD_HHMMSS_<goal-slug>.md
```

The report includes:
- Executive summary
- List of accomplishments
- All artifact paths (files created)
- How to run what was built
- Suggested next steps
- Full session log

---

## Configuration

Edit the constants at the top of `mcp_client_autonomous.py`:

```python
api_base_url and api_key are read from .env in the project root
PLANNER_MODEL        = "qwen3-coder-30b-a3b-instruct"
WORKER_MODEL         = "qwen3-coder-30b-a3b-instruct"
MAX_PARALLEL_WORKERS = 3       # max concurrent workers
PLANNER_TICK_SECONDS = 100     # how often the planner re-evaluates
WORKER_MAX_ITERS     = 30      # max tool calls per worker task
WRAP_UP_SECONDS      = 180     # stop new tasks when this much time is left
```


---

## File Structure

```
autonomous_agent/
├── mcp_server.py              # Lean MCP server (system + web tools only)
├── mcp_client_autonomous.py   # Master planner + parallel workers
├── system_tools.py            # Tool implementations (shell, file I/O, HTTP)
├── web_tools.py               # Tool implementations (search, fetch, screenshot)
└── (no .env here — use the .env in the project root)
```
