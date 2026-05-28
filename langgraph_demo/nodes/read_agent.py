from states.state import AgentState
from config import get_llm
from prompts.templates import READ_AGENT_SYSTEM


def read_agent_node(state: AgentState) -> dict:
    llm = get_llm()
    tasks = state.get("tasks", [])
    read_tasks = [t for t in tasks if t.get("agent_type") == "read" and t.get("status") == "pending"]

    results = []
    for task in read_tasks:
        messages = [
            {"role": "system", "content": READ_AGENT_SYSTEM},
            {"role": "user", "content": task["description"]},
        ]
        response = llm.invoke(messages)
        task["result"] = response.content
        task["status"] = "done"
        results.append(f"## {task['description']}\n{response.content}")

    updated_tasks = [t if t["id"] not in [rt["id"] for rt in read_tasks] else next(rt for rt in read_tasks if rt["id"] == t["id"]) for t in tasks]

    return {
        "read_results": "\n\n".join(results),
        "tasks": updated_tasks,
        "current_step": "reflection",
    }
