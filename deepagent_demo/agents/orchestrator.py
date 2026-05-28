from deepagents import create_deep_agent
from config import get_model
from tools.search import web_search
from tools.code_executor import run_python_code
from prompts.templates import (
    ORCHESTRATOR_SYSTEM,
    SEARCH_AGENT_INSTRUCTIONS,
    CODE_AGENT_INSTRUCTIONS,
    MATH_AGENT_INSTRUCTIONS,
)


def create_orchestrator():
    model = get_model()

    search_agent = {
        "name": "search-agent",
        "description": "Delegate web search tasks to this agent. Use for information retrieval, fact-checking, and research.",
        "system_prompt": SEARCH_AGENT_INSTRUCTIONS,
        "tools": [web_search],
    }

    code_agent = {
        "name": "code-agent",
        "description": "Delegate coding tasks to this agent. Use for writing, executing, and debugging Python code.",
        "system_prompt": CODE_AGENT_INSTRUCTIONS,
        "tools": [run_python_code],
    }

    math_agent = {
        "name": "math-agent",
        "description": "Delegate mathematical tasks to this agent. Use for calculations, proofs, and mathematical reasoning.",
        "system_prompt": MATH_AGENT_INSTRUCTIONS,
        "tools": [run_python_code],
    }

    agent = create_deep_agent(
        model=model,
        tools=[web_search, run_python_code],
        system_prompt=ORCHESTRATOR_SYSTEM,
        subagents=[search_agent, code_agent, math_agent],
    )

    return agent
