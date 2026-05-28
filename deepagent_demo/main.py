import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from agents.orchestrator import create_orchestrator


console = Console()

STEP_COUNTER = [0]


def next_step():
    STEP_COUNTER[0] += 1
    return STEP_COUNTER[0]


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
            if isinstance(msg, HumanMessage):
                step = next_step()
                console.print(f"\n[bold yellow]{'='*60}[/bold yellow]")
                console.print(f"[bold yellow]  Step {step}: User Input[/bold yellow]")
                console.print(f"[bold yellow]{'='*60}[/bold yellow]")
                console.print(f"  {msg.content}")
                console.print(f"[bold yellow]{'='*60}[/bold yellow]")

            elif isinstance(msg, AIMessage):
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_name = tc.get("name", "unknown")
                        args = tc.get("args", {})

                        if tool_name == "task":
                            step = next_step()
                            subagent = args.get("subagent_type", "unknown")
                            desc = args.get("description", "")
                            console.print(f"\n[bold magenta]{'='*60}[/bold magenta]")
                            console.print(f"[bold magenta]  Step {step}: Delegation → {subagent}[/bold magenta]")
                            console.print(f"[bold magenta]{'='*60}[/bold magenta]")
                            console.print(f"  [dim]{desc}[/dim]")
                            console.print(f"[bold magenta]{'='*60}[/bold magenta]")
                        elif tool_name == "web_search":
                            step = next_step()
                            console.print(f"\n[bold blue]{'='*60}[/bold blue]")
                            console.print(f"[bold blue]  Step {step}: Web Search[/bold blue]")
                            console.print(f"[bold blue]{'='*60}[/bold blue]")
                            console.print(f"  Query: {args.get('query', '')}")
                            console.print(f"[bold blue]{'='*60}[/bold blue]")
                        elif tool_name == "run_python_code":
                            step = next_step()
                            code = args.get("code", "")
                            console.print(f"\n[bold green]{'='*60}[/bold green]")
                            console.print(f"[bold green]  Step {step}: Code Execution[/bold green]")
                            console.print(f"[bold green]{'='*60}[/bold green]")
                            console.print(f"  [dim]{code[:300]}{'...' if len(code) > 300 else ''}[/dim]")
                            console.print(f"[bold green]{'='*60}[/bold green]")
                        else:
                            step = next_step()
                            console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
                            console.print(f"[bold cyan]  Step {step}: Tool Call → {tool_name}[/bold cyan]")
                            console.print(f"[bold cyan]{'='*60}[/bold cyan]")
                            for ak, av in args.items():
                                console.print(f"  {ak}: {repr(str(av)[:200])}")
                            console.print(f"[bold cyan]{'='*60}[/bold cyan]")

                if msg.content:
                    step = next_step()
                    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
                    console.print(f"[bold cyan]  Step {step}: Orchestrator Reasoning[/bold cyan]")
                    console.print(f"[bold cyan]{'='*60}[/bold cyan]")
                    console.print(Markdown(msg.content))
                    console.print(f"[bold cyan]{'='*60}[/bold cyan]")

            elif isinstance(msg, ToolMessage):
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                tool_label = msg.name or "unknown"
                preview = content[:300].replace("\n", " ")
                console.print(f"\n[green]  <- Result ({tool_label}):[/green]")
                console.print(f"[dim]     {preview}{'...' if len(content) > 300 else ''}[/dim]")


def run_query(query: str, agent):
    STEP_COUNTER[0] = 0
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
