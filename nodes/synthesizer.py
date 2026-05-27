from states.state import AgentState
from config import get_llm
from prompts.templates import SYNTHESIZER_SYSTEM


def synthesizer_node(state: AgentState) -> dict:
    llm = get_llm()

    context_parts = [f"Original query: {state['user_input']}"]
    if state.get("search_results"):
        context_parts.append(f"Search results:\n{state['search_results']}")
    if state.get("code_results"):
        context_parts.append(f"Code results:\n{state['code_results']}")
    if state.get("read_results"):
        context_parts.append(f"Read results:\n{state['read_results']}")
    if state.get("math_results"):
        context_parts.append(f"Math results:\n{state['math_results']}")
    if state.get("memory_context"):
        context_parts.append(f"Relevant memory:\n{state['memory_context']}")

    context_parts.append(f"Reflection score: {state.get('reflection_score', 'N/A')}")
    context_parts.append(f"Risk flag: {state.get('risk_flag', False)}")
    if state.get("risk_message"):
        context_parts.append(f"Risk message: {state['risk_message']}")

    messages = [
        {"role": "system", "content": SYNTHESIZER_SYSTEM},
        {"role": "user", "content": "\n\n".join(context_parts)},
    ]
    response = llm.invoke(messages)

    return {
        "final_answer": response.content,
        "current_step": "end",
    }
