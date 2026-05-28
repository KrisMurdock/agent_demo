# DeepAgent Demo

基于 LangChain 的多智能体协作系统，通过一个 Orchestrator（编排器）统一调度多个专业子智能体，完成搜索、编码、数学等复杂任务。

## 架构总览

```
User Input
    │
    ▼
┌──────────────────────────────────────┐
│           Orchestrator Agent         │
│  ┌────────────────────────────────┐  │
│  │ 1. Intent Classification       │  │
│  │ 2. Task Planning & Decomposition│ │
│  │ 3. Delegation (via task tool)  │  │
│  │ 4. Reflection & Quality Check  │  │
│  │ 5. Synthesis → Final Answer    │  │
│  └────────────────────────────────┘  │
│                                      │
│  Sub-Agents:                         │
│  ┌──────────┐ ┌──────────┐ ┌──────┐ │
│  │  Search   │ │   Code   │ │ Math │ │
│  │  Agent    │ │  Agent   │ │Agent │ │
│  └────┬─────┘ └────┬─────┘ └──┬───┘ │
│       │             │          │      │
│  ┌────▼─────┐ ┌─────▼────┐ ┌──▼───┐  │
│  │ web_search│ │run_python│ │run_  │  │
│  │ (DuckDuck│ │_code     │ │python│  │
│  │  Go)     │ │(sandbox) │ │_code │  │
│  └──────────┘ └──────────┘ └──────┘  │
└──────────────────────────────────────┘
    │
    ▼
  Output (Rich terminal display)
```

## 整体流程

一次用户查询经历以下 5 个阶段：

**1. 意图分类 & 风险检查** — Orchestrator 分析用户输入，将其分类为 `search`、`code`、`math` 或 `mixed` 类型，同时检测潜在的提示注入或有害请求。

**2. 任务规划** — 将复合查询拆解为多个子任务，每个子任务绑定到对应的专业子智能体。

**3. 执行委派** — 通过 `task` tool 将子任务分发给 search-agent、code-agent 或 math-agent，子智能体使用各自的专用工具完成工作。

**4. 反思 & 质量检查** — 收集子智能体返回结果后，Orchestrator 检查一致性、完整性和是否存在幻觉，质量评分低于阈值时会重新规划。

**5. 综合输出** — 将所有结果融合成结构化的最终回答，通过 Rich 终端渲染展示。

## 目录结构

```
deepagent_demo/
├── main.py              # CLI 入口，REPL 交互循环 + 事件流渲染
├── config.py            # 模型配置（通过环境变量加载 LLM 参数）
├── agents/
│   └── orchestrator.py  # Orchestrator 构建：定义子智能体 + 注册工具
├── prompts/
│   └── templates.py     # 所有智能体的 System Prompt 模板
├── tools/
│   ├── search.py        # web_search: DuckDuckGo 搜索工具
│   └── code_executor.py # run_python_code: 沙箱化 Python 执行工具
├── tests/
│   └── test_smoke.py    # 冒烟测试
├── requirements.txt     # 依赖清单
└── .env.example         # 环境变量模板
```

## 各组件详解

### `main.py` — CLI 入口

- **`main()`**：启动 REPL 循环，读取用户输入并调用 `run_query`。
- **`run_query()`**：调用 agent 的 `stream()` 方法，逐事件打印 Orchestrator 的决策过程（意图分析、委派、工具调用、结果返回）。
- **`print_event()`**：根据消息类型（HumanMessage / AIMessage / ToolMessage）用 Rich 库渲染彩色终端输出，包括 Markdown 格式的 AI 回答和工具委派详情。

### `config.py` — 模型配置

通过 `langchain.chat_models.init_chat_model` 初始化 LLM。支持从 `.env` 读取配置：

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `MODEL_NAME` | `gpt-4o` | 模型名称 |
| `OPENAI_API_KEY` | — | API 密钥 |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | 兼容 OpenAI 接口的自定义端点 |
| `TEMPERATURE` | `0.7` | 生成温度 |
| `MAX_TOKENS` | `4096` | 最大输出 token 数 |

### `agents/orchestrator.py` — 编排器

核心组装逻辑，创建一个 DeepAgent 实例，配置：

- **3 个子智能体**：每个子智能体有独立的 name、description、system_prompt 和 tools
- **Orchestrator 自身**：拥有 `web_search` 和 `run_python_code` 两个全局工具，配合编排 prompt 指导其进行意图分类、任务分解和质量反思

### `prompts/templates.py` — Prompt 模板

包含 4 个 system prompt：

| Prompt | 用途 |
|--------|------|
| `ORCHESTRATOR_SYSTEM` | 定义编排器的 5 步工作流：意图分类 → 任务规划 → 执行委派 → 反思检查 → 综合输出 |
| `SEARCH_AGENT_INSTRUCTIONS` | 指导搜索智能体使用 web_search 工具检索信息并摘要 |
| `CODE_AGENT_INSTRUCTIONS` | 指导编码智能体编写 Python 代码并通过工具执行验证 |
| `MATH_AGENT_INSTRUCTIONS` | 指导数学智能体分步推理，并用 Python 代码验证计算 |

### `tools/search.py` — 网络搜索工具

封装 DuckDuckGo 搜索 API，返回标题、摘要和链接。参数：`query`（搜索词）、`max_results`（最大返回数，默认 5）。

### `tools/code_executor.py` — 代码执行工具

在临时文件中执行 Python 代码的沙箱化工具：

- **安全过滤**：拦截 `os.system`、`subprocess`、`shutil.rmtree` 等危险调用
- **超时控制**：默认 5 秒超时
- **返回格式**：stdout、stderr、success、timeout 四项结构化结果

## 安装与运行

```bash
cd deepagent_demo
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# 编辑 .env 填入你的 API Key

python main.py
```

输入查询后系统会自动分析意图、委派任务并输出结果。输入 `quit` / `exit` / `q` 退出。

## 测试

```bash
pytest tests/
```
