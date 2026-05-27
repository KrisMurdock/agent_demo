from typing import TypedDict


class Task(TypedDict):
    id: str
    description: str
    agent_type: str  # "search" | "code" | "read" | "math"
    status: str  # "pending" | "running" | "done" | "failed"
    dependencies: list[str]
    result: str


class AgentState(TypedDict):
    # Input
    user_input: str

    # Intent Router
    intent: str  # "search" | "code" | "math" | "mixed"
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
