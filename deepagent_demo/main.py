import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from agents.orchestrator import create_orchestrator


console = Console()

MessageType = {
    "human": "User",
    "ai": "Orchestrator",
    "tool": "Tool",
}


def print_event(event: dict):
    for node_name, node_output in event.items():
        if node_output is None:
            continue

        if not isinstance(node_output, dict):
            continue

        messages = node_output.get("messages", [])
        if not messages:
            continue

        for msg in messages:
            msg_type = type(msg).__name__
            label = MessageType.get(msg_type.lower(), msg_type)

            if isinstance(msg, HumanMessage):
                console.print(f"\n[bold yellow]{'='*60}[/bold yellow]")
                console.print(f"[bold yellow]  [User Input][/bold yellow]")
                console.print(f"[bold yellow]{'='*60}[/bold yellow]")
                console.print(f"  {msg.content}")
                console.print(f"[bold yellow]{'='*60}[/bold yellow]")

            elif isinstance(msg, AIMessage):
                if msg.content:
                    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
                    console.print(f"[bold cyan]  [Orchestrator][/bold cyan]")
                    console.print(f"[bold cyan]{'='*60}[/bold cyan]")
                    console.print(Markdown(msg.content))
                    console.print(f"[bold cyan]{'='*60}[/bold cyan]")

                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_name = tc.get("name", "unknown")
                        args = tc.get("args", {})
                        console.print(f"\n[bold magenta]  -> Delegating to: {tool_name}[/bold magenta]")
                        if "description" in args:
                            console.print(f"[dim]     Task: {args['description'][:100]}...[/dim]")
                        elif "subagent_type" in args:
                            console.print(f"[dim]     Subagent: {args['subagent_type']}[/dim]")
                        elif "query" in args:
                            console.print(f"[dim]     Query: {args['query']}[/dim]")
                        elif "code" in args:
                            code_preview = args["code"][:80].replace("\n", " ")
                            console.print(f"[dim]     Code: {code_preview}...[/dim]")

            elif isinstance(msg, ToolMessage):
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                preview = content[:200].replace("\n", " ")
                console.print(f"\n[green]  <- Tool result ({msg.name}):[/green]")
                console.print(f"[dim]     {preview}...[/dim]")


def run_query(query: str, agent):
    console.print(f"\n[bold yellow]Query:[/bold yellow] {query}\n")

    for event in agent.stream(
        {"messages": [HumanMessage(content=query)]},
        stream_mode="updates",
    ):
        print_event(event)

    console.print()


def main():
    console.print("[bold magenta]DeepAgents Multi-Agent System[/bold magenta]")
    console.print("[dim]Building orchestrator agent...[/dim]")

    agent = create_orchestrator()

    console.print("[dim]Ready! Type 'quit' to exit[/dim]\n")

    while True:
        try:
            query = console.input("[bold]> [/bold]")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("quit", "exit", "q"):
            break

        if query.strip():
            try:
                run_query(query.strip(), agent)
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

    console.print("\n[dim]Goodbye![/dim]")


if __name__ == "__main__":
    main()
