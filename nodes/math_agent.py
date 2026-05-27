import re
from states.state import AgentState
from config import get_llm
from tools.code_executor import execute_python_code
from prompts.templates import MATH_AGENT_SYSTEM


def math_agent_node(state: AgentState) -> dict:
    llm = get_llm()
    tasks = state.get("tasks", [])
    math_tasks = [t for t in tasks if t.get("agent_type") == "math" and t.get("status") == "pending"]

    results = []
    for task in math_tasks:
        messages = [
            {"role": "system", "content": MATH_AGENT_SYSTEM},
            {"role": "user", "content": task["description"]},
        ]
        response = llm.invoke(messages)
        content = response.content

        calc_match = re.search(r"```python\n(.*?)```", content, re.DOTALL)
        if calc_match:
            calc_result = execute_python_code(calc_match.group(1))
            content += f"\n\n**Calculator output:** {calc_result['stdout']}"

        task["result"] = content
        task["status"] = "done"
        results.append(f"## {task['description']}\n{content}")

    updated_tasks = [t if t["id"] not in [mt["id"] for mt in math_tasks] else next(mt for mt in math_tasks if mt["id"] == t["id"]) for t in tasks]

    return {
        "math_results": "\n\n".join(results),
        "tasks": updated_tasks,
        "current_step": "reflection",
    }
