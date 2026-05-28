# Agent Demo

多智能体系统实验项目，包含两种不同的多智能体架构实现：基于 LangGraph 的显式图编排方案，和基于 DeepAgents 的声明式编排方案。

## 架构对比

| | `langgraph_demo` | `deepagent_demo` |
|---|---|---|
| **框架** | LangGraph (`StateGraph`) | DeepAgents (`create_deep_agent`) |
| **编排方式** | 显式有向图，节点间通过条件边路由 | 声明式，Orchestrator LLM 自主委派 |
| **状态管理** | 强类型 `AgentState` (TypedDict)，全局共享 | LangGraph messages，流式传递 |
| **智能体** | 9 个独立节点 | 1 Orchestrator + 3 子智能体 |
| **特殊能力** | 反思循环、记忆缓存、阅读智能体 | 提示注入检测、实时流式委派展示 |
| **控制流** | 确定性图拓扑 + 条件路由 | LLM 驱动的动态工具调用 |

## 项目结构

```
agent_demo/
├── langgraph_demo/          # 方案一：LangGraph 显式图编排
│   ├── graph.py             # StateGraph 定义：节点 + 条件边
│   ├── config.py            # LLM 初始化
│   ├── main.py              # CLI 入口 + Rich 渲染
│   ├── states/
│   │   └── state.py         # AgentState / Task 类型定义
│   ├── nodes/               # 9 个独立处理节点
│   │   ├── intent_router.py # 意图分类 + 风险检测
│   │   ├── planner.py       # 任务拆解 + 依赖分析
│   │   ├── search_agent.py  # 搜索执行
│   │   ├── code_agent.py    # 代码编写与执行
│   │   ├── read_agent.py    # 文件读取分析
│   │   ├── math_agent.py    # 数学推理
│   │   ├── reflection.py    # 结果质量评估
│   │   ├── memory.py        # 会话级 QA 缓存
│   │   └── synthesizer.py   # 结果综合输出
│   ├── prompts/
│   │   └── templates.py     # 所有节点的 System Prompt
│   ├── tools/
│   │   ├── search.py        # DuckDuckGo 搜索
│   │   └── code_executor.py # 沙箱化 Python 执行
│   └── tests/
│       └── test_smoke.py
│
├── deepagent_demo/           # 方案二：DeepAgents 声明式编排
│   ├── agents/
│   │   └── orchestrator.py  # 构建 Orchestrator + 3 子智能体
│   ├── config.py            # LLM 初始化
│   ├── main.py              # CLI 入口 + 流式事件渲染
│   ├── prompts/
│   │   └── templates.py     # Orchestrator + 子智能体 Prompt
│   ├── tools/
│   │   ├── search.py        # DuckDuckGo 搜索
│   │   └── code_executor.py # 沙箱化 Python 执行
│   └── tests/
│       └── test_smoke.py
│
├── .env.example             # 环境变量模板
└── requirements.txt         # 共享依赖
```

## 方案一：LangGraph 显式图编排

通过 `StateGraph` 定义一个显式的有向图，每个处理步骤是一个节点，节点间通过条件边路由。

### 执行流程

```
START
  │
  ▼
┌─────────────────┐
│  Intent Router   │── 风险检测 ──→ Synthesizer (安全拦截)
└────────┬────────┘
         │ 正常
         ▼
┌─────────────────┐
│     Planner      │── 任务拆解为多个子任务
└────────┬────────┘
         │ 逐个分发
    ┌────┼────┬────┐
    ▼    ▼    ▼    ▼
 Search Code  Read Math    ← 4 个专业执行节点
    │    │    │    │
    └────┴────┴────┘
         │
         ▼
┌─────────────────┐
│   Reflection     │── 质量评分 < 0.7 ──→ 回到 Planner 重新规划
└────────┬────────┘
         │ 通过
         ▼
┌─────────────────┐
│     Memory       │── 检索相关 QA 缓存，丰富上下文
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Synthesizer    │── 融合所有结果，生成最终回答
└────────┬────────┘
         │
        END
```

### 核心组件

**`states/state.py`** — 强类型全局状态，包含 5 大区块：
- 输入：`user_input`
- 意图分析：`intent`, `risk_flag`, `risk_message`
- 任务调度：`tasks`（含依赖关系）, `current_step`, `retry_count`
- 执行结果：`search_results`, `code_results`, `read_results`, `math_results`
- 质量控制：`reflection_score`, `hallucination_flag`, `missing_info`
- 最终输出：`memory_context`, `final_answer`

**`graph.py`** — 有向图定义，包含 4 个条件路由函数：
- `route_after_intent`：有风险直接跳到 Synthesizer
- `route_after_planner`：根据 `current_step` 分发到对应执行节点
- `route_after_reflection`：质量不达标回到 Planner，达标进入 Memory

**节点分工**：

| 节点 | 职责 |
|------|------|
| `intent_router` | LLM 分析意图（search/code/math/mixed），检测提示注入和有害请求 |
| `planner` | 将查询拆解为带依赖关系的 Task 列表 |
| `search_agent` | 调用 DuckDuckGo 搜索工具 |
| `code_agent` | 编写并执行 Python 代码 |
| `read_agent` | 读取和分析文件内容 |
| `math_agent` | 数学推理与计算验证 |
| `reflection` | 评估结果质量（完整性、幻觉检测），决定是否需要重试 |
| `memory` | 会话级 QA 缓存，避免重复计算 |
| `synthesizer` | 将所有结果融合为结构化的最终回答 |

## 方案二：DeepAgents 声明式编排

基于 `deepagents` 库，将编排逻辑委托给 LLM 自主决策。Orchestrator 通过 `task` 工具将子任务委派给专业子智能体。

### 执行流程

```
User Query
    │
    ▼
┌─────────────────────────────────┐
│        Orchestrator Agent       │
│                                 │
│  1. Intent Classification       │  ← LLM 自主分类
│  2. Task Planning & Decomposition│  ← LLM 自主拆解
│  3. Delegation via task tool    │  ← 动态工具调用
│  4. Reflection & Quality Check  │  ← LLM 自主评估
│  5. Synthesis → Final Answer    │
│                                 │
│  子智能体：                      │
│  ┌──────────┬──────────┬──────┐ │
│  │ Search   │   Code   │ Math │ │
│  │ Agent    │  Agent   │Agent │ │
│  └──────────┴──────────┴──────┘ │
└─────────────────────────────────┘
    │
    ▼
  Final Answer (Rich 终端渲染)
```

### 核心组件

**`agents/orchestrator.py`** — 编排器组装，将 3 个子智能体注册到 DeepAgent：
- `search-agent`：专用搜索，配备 `web_search` 工具
- `code-agent`：代码执行，配备 `run_python_code` 工具
- `math-agent`：数学推理，配备 `run_python_code` 工具

**`prompts/templates.py`** — 4 个 system prompt：
- `ORCHESTRATOR_SYSTEM`：定义 5 步工作流（意图分类 → 任务规划 → 委派 → 反思 → 综合）
- `SEARCH_AGENT_INSTRUCTIONS`：搜索摘要规范
- `CODE_AGENT_INSTRUCTIONS`：编码执行规范
- `MATH_AGENT_INSTRUCTIONS`：数学推理规范

## 共享工具

两个方案共用相同的安全工具实现：

**`tools/search.py`** — 基于 DuckDuckGo 的网络搜索，返回标题、摘要、链接。

**`tools/code_executor.py`** — Python 沙箱执行器：
- 危险模式拦截（`os.system`, `subprocess`, `shutil.rmtree` 等）
- 5 秒超时控制
- 临时文件写入 → 执行 → 清理

## 环境变量

复制 `.env.example` 为 `.env` 并填入配置：

```bash
OPENAI_API_KEY=your-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o
TEMPERATURE=0.7
MAX_TOKENS=4096
```

`OPENAI_BASE_URL` 支持兼容 OpenAI 接口的自定义端点（如 Azure、vLLM、Ollama 等）。

## 安装与运行

```bash
cd agent_demo
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # 编辑填入 API Key

# 运行 LangGraph 版本
cd langgraph_demo && python main.py

# 运行 DeepAgents 版本
cd deepagent_demo && python main.py
```

## 测试

```bash
pytest langgraph_demo/tests/
pytest deepagent_demo/tests/
```
