from states.state import AgentState


_session_memory: list[dict] = []


def memory_node(state: AgentState) -> dict:
    global _session_memory

    relevant = []
    keywords = set(state["user_input"].lower().split())
    for entry in _session_memory:
        entry_keywords = set(entry["query"].lower().split())
        if keywords & entry_keywords:
            relevant.append(f"Q: {entry['query']}\nA: {entry['answer'][:200]}")

    all_results = " ".join(filter(None, [
        state.get("search_results", ""),
        state.get("code_results", ""),
        state.get("read_results", ""),
        state.get("math_results", ""),
    ]))

    _session_memory.append({
        "query": state["user_input"],
        "answer": all_results[:500],
    })

    if len(_session_memory) > 50:
        _session_memory = _session_memory[-50:]

    return {
        "memory_context": "\n\n".join(relevant) if relevant else "",
        "current_step": "synthesizer",
    }
