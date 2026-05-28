import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text
from graph import app


console = Console()


NODE_LABELS = {
    "intent_router": "Intent Router",
    "planner": "Planner",
    "search_agent": "Search Agent",
    "code_agent": "Code Agent",
    "read_agent": "Read Agent",
    "math_agent": "Math Agent",
    "reflection": "Reflection",
    "memory": "Memory",
    "synthesizer": "Synthesizer",
}


def print_step_output(node_name: str, output: dict):
    label = NODE_LABELS.get(node_name, node_name)
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold cyan]  [{label}][/bold cyan]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]")

    if node_name == "intent_router":
        intent = output.get("intent", "")
        risk = output.get("risk_flag", False)
        risk_msg = output.get("risk_message", "")
        console.print(f"  Intent: [bold]{intent}[/bold]")
        if risk:
            console.print(f"  Risk:   [bold red]FLAGGED[/bold red] - {risk_msg}")
        else:
            console.print(f"  Risk:   [green]safe[/green]")

    elif node_name == "planner":
        tasks = output.get("tasks", [])
        console.print(f"  Tasks ({len(tasks)}):")
        for t in tasks:
            status_color = "green" if t.get("status") == "done" else "yellow"
            console.print(f"    [{status_color}]{t['id']}[/{status_color}] [{t.get('agent_type', '?')}] {t.get('description', '')}")

    elif node_name in ("search_agent", "code_agent", "read_agent", "math_agent"):
        result_key = {
            "search_agent": "search_results",
            "code_agent": "code_results",
            "read_agent": "read_results",
            "math_agent": "math_results",
        }[node_name]
        results = output.get(result_key, "")
        if results:
            console.print(Markdown(results))
        else:
            console.print("  [dim]No results[/dim]")

    elif node_name == "reflection":
        score = output.get("reflection_score", 0)
        hallucination = output.get("hallucination_flag", False)
        missing = output.get("missing_info", [])
        score_color = "green" if score >= 0.7 else "red"
        console.print(f"  Score:        [{score_color}]{score:.2f}[/{score_color}]")
        console.print(f"  Hallucination: {'[red]YES[/red]' if hallucination else '[green]no[/green]'}")
        if missing:
            console.print(f"  Missing info:")
            for m in missing:
                console.print(f"    - {m}")

    elif node_name == "memory":
        ctx = output.get("memory_context", "")
        if ctx:
            console.print(f"  Retrieved context:")
            console.print(Markdown(ctx))
        else:
            console.print("  [dim]No relevant memory found[/dim]")

    elif node_name == "synthesizer":
        pass  # final answer printed separately

    console.print(f"[bold cyan]{'='*60}[/bold cyan]")


def run_query(query: str):
    console.print(f"\n[bold yellow]Query:[/bold yellow] {query}\n")

    initial_state = {
        "user_input": query,
        "intent": "",
        "risk_flag": False,
        "risk_message": "",
        "tasks": [],
        "current_step": "",
        "retry_count": 0,
        "search_results": "",
        "code_results": "",
        "read_results": "",
        "math_results": "",
        "reflection_score": 0.0,
        "missing_info": [],
        "hallucination_flag": False,
        "memory_context": "",
        "final_answer": "",
    }

    final_state = None
    for event in app.stream(initial_state, stream_mode="updates"):
        for node_name, node_output in event.items():
            print_step_output(node_name, node_output)
            final_state = node_output

    if final_state and "final_answer" in final_state:
        answer = final_state["final_answer"]
        console.print()
        console.print(Panel(Markdown(answer), title="Final Answer", border_style="bold green"))
    elif final_state:
        console.print(Panel(str(final_state), title="Result", border_style="yellow"))


def main():
    console.print("[bold magenta]LangGraph Multi-Agent System[/bold magenta]")
    console.print("[dim]Type 'quit' to exit[/dim]\n")

    while True:
        try:
            query = console.input("[bold]> [/bold]")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("quit", "exit", "q"):
            break

        if query.strip():
            run_query(query.strip())

    console.print("\n[dim]Goodbye![/dim]")


if __name__ == "__main__":
    main()
