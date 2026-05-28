import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("OPENAI_API_KEY", "sk-test")
os.environ.setdefault("OPENAI_BASE_URL", "https://api.openai.com/v1")
os.environ.setdefault("MODEL_NAME", "gpt-4o")

from states.state import AgentState, Task


def test_state_creation():
    state: AgentState = {
        "user_input": "test",
        "intent": "",
        "risk_flag": False,
        "risk_message": "",
        "tasks": [],
        "current_step": "",
        "retry_count": 0,
        "search_results": "",
        "code_results": "",
        "read_results": "",
        "math_results": "",
        "reflection_score": 0.0,
        "missing_info": [],
        "hallucination_flag": False,
        "memory_context": "",
        "final_answer": "",
    }
    assert state["user_input"] == "test"


def test_task_creation():
    task: Task = {
        "id": "task_1",
        "description": "search for X",
        "agent_type": "search",
        "status": "pending",
        "dependencies": [],
        "result": "",
    }
    assert task["agent_type"] == "search"


def test_code_executor_blocks_dangerous():
    from tools.code_executor import execute_python_code
    result = execute_python_code("import os; os.system('ls')")
    assert result["success"] is False
    assert "Blocked" in result["stderr"]


def test_code_executor_runs_safe():
    from tools.code_executor import execute_python_code
    result = execute_python_code("print(2 + 2)")
    assert result["success"] is True
    assert "4" in result["stdout"]


def test_search_tool():
    from tools.search import duckduckgo_search
    results = duckduckgo_search("python programming", max_results=2)
    assert isinstance(results, list)
    assert len(results) <= 2
