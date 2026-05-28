import re
from states.state import AgentState
from config import get_llm
from tools.code_executor import execute_python_code
from prompts.templates import CODE_AGENT_SYSTEM


def code_agent_node(state: AgentState) -> dict:
    llm = get_llm()
    tasks = state.get("tasks", [])
    code_tasks = [t for t in tasks if t.get("agent_type") == "code" and t.get("status") == "pending"]

    results = []
    for task in code_tasks:
        messages = [
            {"role": "system", "content": CODE_AGENT_SYSTEM},
            {"role": "user", "content": task["description"]},
        ]
        response = llm.invoke(messages)
        content = response.content

        code_match = re.search(r"```python\n(.*?)```", content, re.DOTALL)
        if code_match:
            code = code_match.group(1)
            exec_result = execute_python_code(code)
            result_text = f"{content}\n\n**Execution output:**\nstdout: {exec_result['stdout']}\nstderr: {exec_result['stderr']}"
        else:
            result_text = content

        task["result"] = result_text
        task["status"] = "done"
        results.append(f"## {task['description']}\n{result_text}")

    updated_tasks = [t if t["id"] not in [ct["id"] for ct in code_tasks] else next(ct for ct in code_tasks if ct["id"] == t["id"]) for t in tasks]

    return {
        "code_results": "\n\n".join(results),
        "tasks": updated_tasks,
        "current_step": "reflection",
    }
