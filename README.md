# MCP Server Examples

A collection of **Model Context Protocol (MCP)** implementations ranging from a minimal
hello-world introduction to a full autonomous AI agent that can build software on its own.

> **What is MCP?**
> The Model Context Protocol is an open standard that lets LLMs call external tools — functions
> you define — in a structured, safe way. The model sees a schema; you control the implementation.

---

## Repository Structure

```
mcp-server-example/
│
├── mcp_server.py              # Intro: minimal MCP server (2 tools)
├── mcp_client.py              # Intro: Ollama client that calls those tools
├── requirements.txt           # All Python dependencies
│
├── data_science_mcp/          # Advanced: data visualisation + stats + coding tools
│   ├── mcp_server.py          #   Full server (36 tools: plots, stats, system, web)
│   ├── mcp_client.py          #   Ollama multi-agent client (MAS with supervisor)
│   ├── mcp_client_gpustack.py #   GPUStack API client (OpenAI-compatible)
│   └── README.md
│
└── autonomous_agent/          # Expert: time-budgeted autonomous AI agent
    ├── mcp_server.py          #   Lean server (13 tools: system + web only)
    ├── mcp_client_autonomous.py #  Master planner + parallel workers
    └── README.md
```

---

## 1. Intro — Root Level

The simplest possible MCP setup. Two tools, one model, one conversation.

**Tools exposed:**
- `get_current_time` — returns the current timestamp
- `calculate_sum` — adds two numbers

**Run it:**
```bash
# Install dependencies
pip install -r requirements.txt

# Start a chat
python mcp_client.py
```

The root client uses [Ollama](https://ollama.com) with a local model.
Change `OLLAMA_MODEL` at the top of `mcp_client.py` to switch models.

---

## 2. Data Science MCP — `data_science_mcp/`

A production-grade multi-agent system for data analysis and software development.
See [`data_science_mcp/README.md`](data_science_mcp/README.md) for full details.

**Highlights:**
- 36 MCP tools: interactive Plotly charts, static Matplotlib plots, statistical tests,
  shell commands, file I/O, web search, and more
- Two clients: local Ollama (`mcp_client.py`) and GPUStack API (`mcp_client_gpustack.py`)
- Supervisor + specialist agent architecture (routing, delegation, parallel execution)

---

## 3. Autonomous Agent — `autonomous_agent/`

Give it a time budget and a goal (or no goal at all) and it works autonomously.
See [`autonomous_agent/README.md`](autonomous_agent/README.md) for full details.

**Highlights:**
- Master planner dispatches subtasks to up to 3 parallel workers
- Focused on software engineering: shell, files, web search, HTTP testing
- Safety guard blocks destructive commands and restricts write paths
- Produces a full Markdown report at the end of every session

---

## Installation

```bash
# Clone and enter the repo
git clone <repo-url>
cd mcp-server-example

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows

# Install all dependencies
pip install -r requirements.txt

# Add your API credentials once in the project root (used by all clients)
cp .env.example .env
# then edit .env with your api_key and api_base_url

# For Playwright screenshots (optional)
playwright install chromium
sudo venv/bin/playwright install-deps chromium
```

---

## Learning Path

| Step | What to read/run | Concept learned |
|------|-----------------|-----------------|
| 1 | Root `mcp_server.py` | Defining MCP tools with `@mcp.tool()` |
| 2 | Root `mcp_client.py` | Connecting a client, listing tools, calling them |
| 3 | `data_science_mcp/mcp_server.py` | Scaling to many tools, complex implementations |
| 4 | `data_science_mcp/mcp_client.py` | Multi-agent routing and delegation |
| 5 | `autonomous_agent/mcp_client_autonomous.py` | Planner + parallel workers + safety + reporting |
