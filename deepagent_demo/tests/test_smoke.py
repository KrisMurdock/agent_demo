import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("OPENAI_API_KEY", "sk-test")
os.environ.setdefault("OPENAI_BASE_URL", "https://api.openai.com/v1")
os.environ.setdefault("MODEL_NAME", "gpt-4o")


def test_search_tool():
    from tools.search import web_search
    result = web_search.invoke({"query": "python programming", "max_results": 2})
    assert isinstance(result, str)
    assert len(result) > 0


def test_code_executor_blocks_dangerous():
    from tools.code_executor import run_python_code
    result = run_python_code.invoke({"code": "import os; os.system('ls')"})
    assert "Blocked" in result


def test_code_executor_runs_safe():
    from tools.code_executor import run_python_code
    result = run_python_code.invoke({"code": "print(2 + 2)"})
    assert "4" in result
    assert "success: True" in result


def test_orchestrator_creates():
    from agents.orchestrator import create_orchestrator
    agent = create_orchestrator()
    assert agent is not None
