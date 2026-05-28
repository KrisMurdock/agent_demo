# LangGraph Multi-Agent System

基于 LangGraph 构建的多智能体协作系统，能够自动将用户请求分解为多个子任务，分发给不同的专家 Agent 执行，并通过反思机制保证输出质量。

## 架构总览

```
                            ┌─────────────┐
                            │  User Input  │
                            └──────┬──────┘
                                   │
                            ┌──────▼──────┐
                            │Intent Router │ ◄── 分类意图 + 风险评估
                            └──────┬──────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │ risk_flag    │ safe         │
                    ▼              ▼              │
             ┌──────────┐  ┌──────────┐          │
             │Synthesizer│  │ Planner  │          │
             │ (安全跳过) │  │ 分解任务  │          │
             └─────┬────┘  └────┬─────┘          │
                   │            │                 │
                   │     ┌──────┼──────┐          │
                   │     ▼      ▼      ▼          │
                   │  ┌──────┐┌──────┐┌──────┐   │
                   │  │Search││ Code ││ Math │   │
                   │  │Agent ││Agent ││Agent │   │
                   │  └──┬───┘└──┬───┘└──┬───┘   │
                   │     └───────┼───────┘       │
                   │             ▼               │
                   │     ┌──────────────┐        │
                   │     │ Reflection   │ ◄──────┘
                   │     │ 质量评估      │
                   │     └──────┬──────┘
                   │            │
                   │     score<0.7 且 retries<2 ?
                   │     ┌──────┴──────┐
                   │     │ YES: 回到    │ NO: 继续
                   │     │ Planner 重试 │
                   │     └─────────────┘
                   │             │
                   │      ┌──────▼──────┐
                   │      │   Memory    │
                   │      │ 会话记忆存储  │
                   │      └──────┬──────┘
                   │             │
                   └──────┬──────┘
                    ┌─────▼─────┐
                    │Synthesizer│
                    │ 最终回答    │
                    └─────┬─────┘
                          │
                   ┌──────▼──────┐
                   │     END     │
                   └─────────────┘
```

## 项目结构

```
langgraph_demo/
├── config.py              # LLM 配置 (OpenAI API)
├── graph.py               # LangGraph 图定义与路由逻辑
├── main.py                # CLI 入口，Rich 终端交互
├── requirements.txt       # 依赖
├── nodes/
│   ├── intent_router.py   # 意图分类 + 风险评估
│   ├── planner.py         # 任务分解
│   ├── search_agent.py    # 网络搜索专家
│   ├── code_agent.py      # 代码生成与执行专家
│   ├── read_agent.py      # 文档阅读专家
│   ├── math_agent.py      # 数学计算专家
│   ├── reflection.py      # 质量反思与重试
│   ├── memory.py          # 会话级记忆
│   └── synthesizer.py     # 最终答案合成
├── prompts/
│   └── templates.py       # 所有 Agent 的 System Prompt
├── states/
│   └── state.py           # AgentState 和 Task 类型定义
├── tools/
│   ├── search.py          # DuckDuckGo 搜索封装
│   └── code_executor.py   # 沙箱化 Python 代码执行
└── tests/
    └── test_smoke.py      # 冒烟测试
```

## 核心组件详解

### 1. State — 全局状态 (`states/state.py`)

整个系统通过一个共享的 `AgentState` TypedDict 在节点间传递数据。每个节点只修改自己关心的字段，其余字段保持不变。

```python
class AgentState(TypedDict):
    user_input: str              # 用户原始输入
    intent: str                  # 意图分类: search / code / math / mixed
    risk_flag: bool              # 是否触发风险标记
    risk_message: str            # 风险描述
    tasks: list[Task]            # Planner 生成的任务列表
    current_step: str            # 下一个要执行的节点
    retry_count: int             # 重试次数 (反思反馈重试)
    search_results: str          # 搜索 Agent 输出
    code_results: str            # 代码 Agent 输出
    read_results: str            # 阅读 Agent 输出
    math_results: str            # 数学 Agent 输出
    reflection_score: float      # 反思质量评分 (0.0-1.0)
    missing_info: list[str]      # 缺失信息列表
    hallucination_flag: bool     # 幻觉检测标记
    memory_context: str          # 从记忆中检索的相关上下文
    final_answer: str            # 最终合成答案
```

每个 `Task` 由 Planner 生成，包含:
- `id`: 唯一标识
- `description`: 任务描述
- `agent_type`: 执行 Agent 类型 (`search` / `code` / `read` / `math`)
- `status`: 状态 (`pending` -> `running` -> `done` / `failed`)
- `dependencies`: 依赖的前置任务 ID

### 2. Intent Router — 意图路由器 (`nodes/intent_router.py`)

系统的入口节点，负责两件事:
1. **意图分类**: 判断用户请求属于 search / code / math / mixed
2. **风险评估**: 检测 prompt 注入、有害请求、敏感系统访问

返回 JSON 格式的意图和风险信息。如果 `risk_flag=True`，系统会跳过整个执行流程，直接进入 Synthesizer 输出安全警告。

### 3. Planner — 任务规划器 (`nodes/planner.py`)

根据意图将用户请求分解为可执行的任务列表。支持:
- **任务依赖**: 任务间可以声明依赖关系
- **混合意图**: `mixed` 类型会拆分为多个并行子任务
- **记忆上下文**: 会参考 Memory 节点检索的历史上下文

Planner 会找出第一个 `pending` 状态的任务，将其 `agent_type` 作为 `current_step` 返回，驱动图的路由。

### 4. 专家 Agents

每个专家 Agent 负责处理特定类型的任务:

| Agent | 职责 | 工具 |
|-------|------|------|
| **Search Agent** | 网络信息检索 | DuckDuckGo 搜索 |
| **Code Agent** | 编写并执行 Python 代码 | 沙箱代码执行器 |
| **Read Agent** | 文档/URL 内容分析 | LLM 直接推理 |
| **Math Agent** | 数学推理与计算 | LLM + 代码执行验证 |

每个 Agent 的工作流程:
1. 从 `tasks` 中筛选出自己负责的 `pending` 任务
2. 对每个任务: 调用 LLM 生成结果 -> (可选)调用工具执行 -> 标记任务为 `done`
3. 将结果写入对应的 `*_results` 字段
4. 设置 `current_step = "reflection"` 进入反思阶段

### 5. Reflection — 质量反思 (`nodes/reflection.py`)

对所有 Agent 的输出进行质量评估，检查三个维度:
- **一致性**: 不同 Agent 的结果是否矛盾
- **完整性**: 结果是否完整回答了原始问题
- **幻觉**: 是否有无依据的断言

评估结果:
- `score >= 0.7`: 质量合格，继续后续流程
- `score < 0.7` 且 `retry_count < 2`: 回到 Planner 重新规划（最多重试 2 次）
- 重试次数用尽: 强制继续，将已知缺陷传递给 Synthesizer

### 6. Memory — 会话记忆 (`nodes/memory.py`)

基于关键词匹配的会话级记忆，不依赖外部数据库:
- 每次查询后将 `(query, answer)` 存入会话记忆
- 下次查询时通过关键词交集检索相关历史
- 记忆容量上限 50 条，超出后 FIFO 淘汰
- 检索到的记忆会作为上下文传递给 Planner 和 Synthesizer

### 7. Synthesizer — 最终合成 (`nodes/synthesizer.py`)

系统出口，将所有 Agent 结果 + 反思评分 + 记忆上下文融合为最终答案:
- 高分路径: 输出完整、结构化的综合回答
- 低分路径: 输出带 caveat 的最佳答案
- 风险路径: 输出安全警告

### 8. Tools

#### 搜索工具 (`tools/search.py`)
封装 DuckDuckGo 搜索 API，返回 `[{title, snippet, url}]` 格式的结果列表。

#### 代码执行器 (`tools/code_executor.py`)
沙箱化执行 Python 代码:
- **安全过滤**: 拦截 `os.system`、`subprocess`、`shutil.rmtree` 等危险模式
- **超时保护**: 默认 5 秒超时
- **临时文件**: 代码写入临时文件执行后自动清理

### 9. Graph — 图定义 (`graph.py`)

LangGraph `StateGraph` 定义了所有节点和路由:

```
START -> intent_router -> [planner | synthesizer]
planner -> [search_agent | code_agent | read_agent | math_agent]
*_agent -> reflection -> [planner (重试) | memory]
memory -> synthesizer -> END
```

关键路由函数:
- `route_after_intent`: 根据 `risk_flag` 决定跳过还是执行
- `route_after_planner`: 根据 `current_step` 分发到对应 Agent
- `route_after_reflection`: 根据评分决定重试还是继续
- `route_after_memory`: 固定进入 Synthesizer

## 快速开始

### 环境要求

- Python 3.10+
- OpenAI API Key (或兼容的 API)

### 安装

```bash
pip install -r requirements.txt
```

### 配置

创建 `.env` 文件:

```env
OPENAI_API_KEY=sk-your-key
OPENAI_BASE_URL=https://api.openai.com/v1   # 可选，用于自定义 API 端点
MODEL_NAME=gpt-4o                             # 可选，默认 gpt-4o
TEMPERATURE=0.7                               # 可选
MAX_TOKENS=4096                               # 可选
```

### 运行

```bash
python main.py
```

进入交互式终端，直接输入问题即可。系统会以彩色面板逐步展示每个节点的执行过程。

### 运行测试

```bash
pytest tests/
```

## 示例交互

```
> 帮我计算 (2^10 + 3^5) / 7 的结果，并搜索一下 Python 3.12 的新特性

  ============================================================
    [Intent Router]
  ============================================================
    Intent: mixed
    Risk:   safe
  ============================================================

    [Planner]
  ============================================================
    Tasks (2):
      task_1 [math] 计算 (2^10 + 3^5) / 7
      task_2 [search] 搜索 Python 3.12 新特性
  ============================================================

    [Math Agent]
  ============================================================
    ## 计算 (2^10 + 3^5) / 7
    2^10 = 1024
    3^5 = 243
    (1024 + 243) / 7 = 1267 / 7 = 181
  ============================================================

    [Reflection]
  ============================================================
    Score:        0.85
    Hallucination: no
  ============================================================

    [Synthesizer]
  ============================================================
    Final Answer
    ...
```

## 设计亮点

1. **声明式路由**: 所有路由逻辑通过 `current_step` 字段驱动，节点之间解耦清晰
2. **反思-重试机制**: 低质量输出自动回退到 Planner 重新规划，最多重试 2 次
3. **安全第一**: 入口处的 Intent Router 可拦截危险请求，代码执行器有模式过滤和超时保护
4. **会话记忆**: 轻量级关键词匹配记忆，无需外部向量数据库，适合演示场景
5. **流式输出**: CLI 使用 Rich 库实现彩色、结构化的逐步输出，调试友好
