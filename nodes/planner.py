import json
from states.state import AgentState
from config import get_llm
from prompts.templates import PLANNER_SYSTEM


def planner_node(state: AgentState) -> dict:
    llm = get_llm()

    context = f"Intent: {state['intent']}\nQuery: {state['user_input']}"
    if state.get("memory_context"):
        context += f"\nRelevant memory: {state['memory_context']}"

    messages = [
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": context},
    ]
    response = llm.invoke(messages)
    content = response.content

    try:
        tasks = json.loads(content)
    except json.JSONDecodeError:
        tasks = [{"id": "task_1", "description": state["user_input"], "agent_type": state["intent"], "dependencies": []}]

    for task in tasks:
        task.setdefault("status", "pending")
        task.setdefault("result", "")

    first_pending = next((t for t in tasks if t["status"] == "pending"), None)
    next_step = first_pending["agent_type"] if first_pending else "reflection"

    return {
        "tasks": tasks,
        "current_step": next_step,
    }
