# LangGraph Multi-Agent System Design

## Overview

A multi-agent system built on LangGraph that orchestrates specialized agents (Search, Code, Read, Math) through a pipeline of intent routing, planning, execution, reflection, and synthesis. Outputs via CLI with streaming support.

## Architecture

Single `StateGraph` with conditional routing and fan-out parallelism. All agents are nodes in one graph; routing is driven by the `current_step` field in shared state.

### Project Structure

```
agent_demo/
├── states/
│   └── state.py            # AgentState TypedDict
├── nodes/
│   ├── __init__.py
│   ├── intent_router.py    # 意图分类 + 风险检查
│   ├── planner.py          # 任务拆解 + DAG 生成
│   ├── search_agent.py     # DuckDuckGo 搜索 + 摘要
│   ├── code_agent.py       # 代码生成 + 本地沙箱执行
│   ├── read_agent.py       # 文档/URL 阅读摘要
│   ├── math_agent.py       # 数学推理 + 计算
│   ├── reflection.py       # 证据检查 + 幻觉检测
│   ├── memory.py           # session 记忆管理
│   └── synthesizer.py      # 结果融合 + 最终答案
├── tools/
│   ├── __init__.py
│   ├── search.py           # DuckDuckGo 搜索工具
│   └── code_executor.py    # 本地 Python 沙箱
├── prompts/
│   └── templates.py        # 所有 Agent 的系统提示词
├── graph.py                # LangGraph 图定义 + 编译
├── config.py               # .env 读取 + LLM 初始化
├── main.py                 # CLI 入口 + 流式输出
├── .env.example            # 环境变量模板
└── requirements.txt        # 依赖
```

## State Definition

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class Task(TypedDict):
    id: str
    description: str
    agent_type: str      # "search" | "code" | "read" | "math"
    status: str          # "pending" | "running" | "done" | "failed"
    dependencies: list[str]  # 依赖的 task ids
    result: str

class AgentState(TypedDict):
    # Input
    user_input: str

    # Intent Router
    intent: str              # "search" | "code" | "math" | "mixed"
    risk_flag: bool
    risk_message: str

    # Planner
    tasks: list[Task]
    current_step: str
    retry_count: int

    # Agent outputs
    search_results: str
    code_results: str
    read_results: str
    math_results: str

    # Reflection
    reflection_score: float
    missing_info: list[str]
    hallucination_flag: bool

    # Memory
    memory_context: str

    # Synthesizer
    final_answer: str
```

## Graph Routing

```
START → Intent Router
  ├─ (risk_flag=True)  → Synthesizer → END
  └─ (risk_flag=False) → Planner
                          │
                          ├─ intent="search" → SearchAgent → ReadAgent → Reflection
                          ├─ intent="code"   → CodeAgent → MathAgent → Reflection
                          ├─ intent="math"   → MathAgent → Reflection
                          └─ intent="mixed"  → fan-out [SearchAgent, CodeAgent] → Reflection
                          │
                          Reflection
                          ├─ score >= 0.7 → Memory → Synthesizer → END
                          └─ score < 0.7  → Planner (retry, max 2)
```

## Node Specifications

### Intent Router
- **Input**: `user_input`
- **Output**: `intent`, `risk_flag`, `risk_message`
- **Logic**: LLM classifies intent into search/code/math/mixed. Rule-based scan for prompt injection patterns and sensitive operations. Risk hit → skip to Synthesizer with warning.

### Planner
- **Input**: `intent`, `user_input`, `memory_context`
- **Output**: `tasks` (DAG), `current_step`
- **Logic**: LLM decomposes query into tasks with dependencies. Outputs ordered task list with agent_type assignment. Respects intent to determine which agent pipeline to activate.

### SearchAgent
- **Input**: `tasks` (filtered to search type)
- **Output**: `search_results`
- **Logic**: For each search task, call DuckDuckGo tool → collect top results → LLM summarizes relevant findings.

### CodeAgent
- **Input**: `tasks` (filtered to code type)
- **Output**: `code_results`
- **Logic**: LLM generates Python code → code_executor sandbox runs it (5s timeout, stdout/stderr capture) → LLM interprets output. Blocks dangerous operations (os.system, subprocess, file deletion).

### ReadAgent
- **Input**: `tasks` (filtered to read type)
- **Output**: `read_results`
- **Logic**: If task contains URL or file path, fetch/read content → LLM summarizes key points.

### MathAgent
- **Input**: `tasks` (filtered to math type)
- **Output**: `math_results`
- **Logic**: LLM reasoning step-by-step + optional Python calculator tool for verification.

### Reflection
- **Input**: All agent outputs
- **Output**: `reflection_score`, `hallucination_flag`, `missing_info`
- **Logic**: Cross-validates results across agents. Checks: (1) consistency between sources, (2) completeness against original query, (3) hallucination indicators. Score 0-1, threshold 0.7.

### Memory
- **Input**: Current query + results + history
- **Output**: `memory_context`
- **Logic**: In-memory dict storing past QA pairs + key facts. Retrieves relevant context via keyword matching. Updates with current interaction. Persisted via LangGraph `MemorySaver` checkpoint.

### Synthesizer
- **Input**: All results + reflection score + memory context
- **Output**: `final_answer`
- **Logic**: LLM fuses multi-source results into coherent answer. If risk_flag, outputs safety warning instead. Handles streaming output.

## Tools

### DuckDuckGo Search (`tools/search.py`)
- Uses `duckduckgo_search` library
- Returns top 5 results with title, snippet, URL
- No API key required

### Code Executor (`tools/code_executor.py`)
- `subprocess.run()` with 5-second timeout
- Captures stdout and stderr
- Blacklist: `os.system`, `subprocess`, `shutil.rmtree`, `os.remove`, `eval`, `exec` (direct)
- Returns structured result: `{stdout, stderr, success, timeout}`

## Prompts

All prompts centralized in `prompts/templates.py` as functions returning system prompt strings. Templates use f-string interpolation to inject state context. Each agent has:
- System prompt (role + instructions + output format)
- User prompt template (context from state)

## Configuration

`.env` file:
```
OPENAI_API_KEY=your-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o
TEMPERATURE=0.7
MAX_TOKENS=4096
```

`config.py` reads via `python-dotenv`, initializes `langchain_openai.ChatOpenAI`.

## Dependencies

```
langgraph
langchain
langchain-openai
langchain-core
duckduckgo-search
python-dotenv
rich
```

## CLI Interface

`main.py` uses `rich` for formatted terminal output:
- Streaming token display via `astream_events`
- Color-coded agent status updates
- Final answer in a styled panel
- Interactive loop: accept multiple queries in sequence
