import json
from states.state import AgentState
from config import get_llm
from prompts.templates import REFLECTION_SYSTEM


def reflection_node(state: AgentState) -> dict:
    llm = get_llm()

    all_results = []
    if state.get("search_results"):
        all_results.append(f"Search results:\n{state['search_results']}")
    if state.get("code_results"):
        all_results.append(f"Code results:\n{state['code_results']}")
    if state.get("read_results"):
        all_results.append(f"Read results:\n{state['read_results']}")
    if state.get("math_results"):
        all_results.append(f"Math results:\n{state['math_results']}")

    context = f"Original query: {state['user_input']}\n\nResults:\n" + "\n\n".join(all_results)

    messages = [
        {"role": "system", "content": REFLECTION_SYSTEM},
        {"role": "user", "content": context},
    ]
    response = llm.invoke(messages)
    content = response.content

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {"score": 0.8, "hallucination_flag": False, "missing_info": [], "feedback": ""}

    retry_count = state.get("retry_count", 0)
    score = parsed.get("score", 0.8)

    if score < 0.7 and retry_count < 2:
        next_step = "planner"
        retry_count += 1
    else:
        next_step = "memory"

    return {
        "reflection_score": score,
        "hallucination_flag": parsed.get("hallucination_flag", False),
        "missing_info": parsed.get("missing_info", []),
        "current_step": next_step,
        "retry_count": retry_count,
    }
