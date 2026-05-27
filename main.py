from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from graph import app


console = Console()


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
            console.print(f"[dim]  [{node_name} done][/dim]")
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
