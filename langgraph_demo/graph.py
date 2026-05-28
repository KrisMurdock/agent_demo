import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from langgraph.graph import StateGraph, START, END
from states.state import AgentState
from nodes import (
    intent_router_node,
    planner_node,
    search_agent_node,
    code_agent_node,
    read_agent_node,
    math_agent_node,
    reflection_node,
    memory_node,
    synthesizer_node,
)


def route_after_intent(state: AgentState) -> str:
    if state.get("risk_flag"):
        return "synthesizer"
    return "planner"


def route_after_planner(state: AgentState) -> str:
    step = state.get("current_step", "reflection")
    return step


def route_after_reflection(state: AgentState) -> str:
    return state.get("current_step", "memory")


def route_after_memory(state: AgentState) -> str:
    return "synthesizer"


graph = StateGraph(AgentState)

graph.add_node("intent_router", intent_router_node)
graph.add_node("planner", planner_node)
graph.add_node("search_agent", search_agent_node)
graph.add_node("code_agent", code_agent_node)
graph.add_node("read_agent", read_agent_node)
graph.add_node("math_agent", math_agent_node)
graph.add_node("reflection", reflection_node)
graph.add_node("memory", memory_node)
graph.add_node("synthesizer", synthesizer_node)

graph.add_edge(START, "intent_router")
graph.add_conditional_edges("intent_router", route_after_intent, {
    "planner": "planner",
    "synthesizer": "synthesizer",
})
graph.add_conditional_edges("planner", route_after_planner, {
    "search": "search_agent",
    "code": "code_agent",
    "read": "read_agent",
    "math": "math_agent",
    "reflection": "reflection",
})
graph.add_edge("search_agent", "reflection")
graph.add_edge("code_agent", "reflection")
graph.add_edge("read_agent", "reflection")
graph.add_edge("math_agent", "reflection")
graph.add_conditional_edges("reflection", route_after_reflection, {
    "planner": "planner",
    "memory": "memory",
})
graph.add_edge("memory", "synthesizer")
graph.add_edge("synthesizer", END)

app = graph.compile()
