from states.state import AgentState
from config import get_llm
from tools.search import duckduckgo_search
from prompts.templates import SEARCH_AGENT_SYSTEM


def search_agent_node(state: AgentState) -> dict:
    llm = get_llm()
    tasks = state.get("tasks", [])
    search_tasks = [t for t in tasks if t.get("agent_type") == "search" and t.get("status") == "pending"]

    results = []
    for task in search_tasks:
        search_results = duckduckgo_search(task["description"])
        context = "\n".join(
            f"[{r['title']}]({r['url']})\n{r['snippet']}" for r in search_results
        )

        messages = [
            {"role": "system", "content": SEARCH_AGENT_SYSTEM},
            {"role": "user", "content": f"Task: {task['description']}\n\nSearch results:\n{context}"},
        ]
        response = llm.invoke(messages)
        task["result"] = response.content
        task["status"] = "done"
        results.append(f"## {task['description']}\n{response.content}")

    updated_tasks = [t if t["id"] not in [st["id"] for st in search_tasks] else next(st for st in search_tasks if st["id"] == t["id"]) for t in tasks]

    return {
        "search_results": "\n\n".join(results),
        "tasks": updated_tasks,
        "current_step": "reflection",
    }
