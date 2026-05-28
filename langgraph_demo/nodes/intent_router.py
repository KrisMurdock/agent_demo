import json
from states.state import AgentState
from config import get_llm
from prompts.templates import INTENT_ROUTER_SYSTEM


def intent_router_node(state: AgentState) -> dict:
    llm = get_llm()
    messages = [
        {"role": "system", "content": INTENT_ROUTER_SYSTEM},
        {"role": "user", "content": state["user_input"]},
    ]
    response = llm.invoke(messages)
    content = response.content

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {"intent": "search", "risk_flag": False, "risk_message": ""}

    return {
        "intent": parsed.get("intent", "search"),
        "risk_flag": parsed.get("risk_flag", False),
        "risk_message": parsed.get("risk_message", ""),
        "current_step": "planner",
        "retry_count": 0,
    }
