# Data Science MCP

A multi-agent MCP setup for interactive data analysis, visualization, and software development.
Connects a powerful LLM to 36 tools organised into specialist agents.

---

## Tools

| Category | Tools |
|----------|-------|
| **Interactive charts** (Plotly HTML) | histogram, scatter, box, line, bar, scatter matrix, correlation heatmap, custom |
| **Static charts** (Matplotlib/Seaborn PNG) | histogram, scatter, box, line, bar, pair plot, correlation heatmap, word cloud, custom |
| **Statistics** | Pearson/Spearman correlation, T-test/ANOVA, OLS linear regression, feature ranking |
| **Data exploration** | column summary, full dataset summary |
| **System / coding** | shell commands, read/write/patch files, directory listing, grep, background processes, HTTP |
| **Internet** | DuckDuckGo search, webpage fetch (text + CSS), Playwright screenshot |

---

## Clients

### `mcp_client.py` — Local Ollama

Uses a locally running Ollama model. A supervisor agent routes requests to specialist
sub-agents (interactive, static, stats, coder) and synthesizes the final response.

```bash
# Make sure Ollama is running and you have a model pulled
ollama pull qwen3.5:9b

python mcp_client.py
```

Change `OLLAMA_MODEL` at the top of the file to switch models.

### `mcp_client_gpustack.py` — GPUStack API

Uses the OpenAI-compatible GPUStack API. Supports the same multi-agent architecture
plus a coder supervisor/worker pattern for long coding tasks.

```bash
# Add your credentials once in the project root (shared by all clients)
# Copy .env.example to .env and fill in your values
cp ../.env.example ../.env
# then edit ../.env

python mcp_client_gpustack.py
```

Change `MODEL` at the top of the file to switch models.

---

## Example Queries

```
You: plot the survival rate by passenger class in /home/user/Titanic-Dataset.csv
You: show me the correlation heatmap for all numeric columns
You: run a t-test on age grouped by survived
You: create a Streamlit app that visualises this dataset
You: search the web for the best Python charting library in 2025
```

---

## File Structure

```
data_science_mcp/
├── mcp_server.py          # Registers all 36 tools
├── mcp_client.py          # Ollama multi-agent client
├── mcp_client_gpustack.py # GPUStack multi-agent client
├── plot_data.py           # Dataset loading and summary tools
├── plot_interactive.py    # Plotly chart implementations
├── plot_static.py         # Matplotlib/Seaborn/WordCloud implementations
├── stats_analysis.py      # Statistical test implementations
├── system_tools.py        # Shell, file I/O, HTTP tools
├── web_tools.py           # Web search, fetch, screenshot
├── viz_agent.py           # Multi-agent orchestration logic
├── viz_config.py          # Agent prompts, tool scopes, labels
└── viz_utils.py           # Shared plotting utilities
```

---

## Output

- **Interactive plots** → `~/plots/*.html` (open in any browser)
- **Static plots** → `~/plots/*.png`
- **Screenshots** → `~/mcp-server-example/screenshots/*.png`
