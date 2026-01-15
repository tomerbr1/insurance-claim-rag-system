#!/usr/bin/env python3
"""
Insurance Claims RAG System - Beautiful Interactive CLI

A multi-agent RAG system for insurance claim document analysis
with code-based, model-based, and human evaluation graders.

AI Agents & Automation Development Course - Final Project
Author: Tomer Brami
Submitted: January 15th, 2026
"""

import sys
import os

# Silence NLTK downloads before any imports
from src.utils.nltk_silencer import silence_nltk_downloads
silence_nltk_downloads()

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text
from rich import box
import questionary
from questionary import Style

# Initialize Rich console
console = Console()

# Custom style for questionary
custom_style = Style([
    ('qmark', 'fg:cyan bold'),
    ('question', 'bold'),
    ('answer', 'fg:cyan'),
    ('pointer', 'fg:cyan bold'),
    ('highlighted', 'fg:cyan bold'),
    ('selected', 'fg:green'),
])

# ASCII Art Banner
BANNER = '''
[cyan]╔═════════════════════════════════════════════════════════════════════════╗[/cyan]
[cyan]║[/cyan]                                                                         [cyan]║[/cyan]
[cyan]║[/cyan][bold yellow] ██╗███╗   ██╗███████╗██╗   ██╗██████╗  █████╗ ███╗   ██╗ ██████╗███████╗[/bold yellow][cyan]║[/cyan]
[cyan]║[/cyan][bold yellow] ██║████╗  ██║██╔════╝██║   ██║██╔══██╗██╔══██╗████╗  ██║██╔════╝██╔════╝[/bold yellow][cyan]║[/cyan]
[cyan]║[/cyan][bold yellow] ██║██╔██╗ ██║███████╗██║   ██║██████╔╝███████║██╔██╗ ██║██║     █████╗  [/bold yellow][cyan]║[/cyan]
[cyan]║[/cyan][bold yellow] ██║██║╚██╗██║╚════██║██║   ██║██╔══██╗██╔══██║██║╚██╗██║██║     ██╔══╝  [/bold yellow][cyan]║[/cyan]
[cyan]║[/cyan][bold yellow] ██║██║ ╚████║███████║╚██████╔╝██║  ██║██║  ██║██║ ╚████║╚██████╗███████╗[/bold yellow][cyan]║[/cyan]
[cyan]║[/cyan][bold yellow] ╚═╝╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝╚══════╝[/bold yellow][cyan]║[/cyan]
[cyan]║[/cyan]                                                                         [cyan]║[/cyan]
[cyan]║[/cyan]        [bold white]CLAIMS RAG SYSTEM[/bold white] [dim]• Multi-Agent Document Intelligence[/dim]            [cyan]║[/cyan]
[cyan]║[/cyan]                                                                         [cyan]║[/cyan]
[cyan]╠═════════════════════════════════════════════════════════════════════════╣[/cyan]
[cyan]║[/cyan]                                                                         [cyan]║[/cyan]
[cyan]║[/cyan]  [bold magenta]AI Agents & Automation Development Course - Final Project[/bold magenta]              [cyan]║[/cyan]
[cyan]║[/cyan]  [bold green]Author: Tomer Brami[/bold green]                                                    [cyan]║[/cyan]
[cyan]║[/cyan]  [dim]Submitted: January 15th, 2026[/dim]                                          [cyan]║[/cyan]
[cyan]║[/cyan]                                                                         [cyan]║[/cyan]
[cyan]╚═════════════════════════════════════════════════════════════════════════╝[/cyan]
'''


def show_banner():
    """Display the ASCII art banner."""
    console.print(BANNER)


def show_status(system: dict):
    """Display system status panel."""
    stats = system['metadata_store'].get_statistics()
    mcp_tools = system.get('mcp_tools', [])

    status_text = Text()
    status_text.append("System Status\n", style="bold cyan")
    status_text.append(f"├── Claims Loaded: {stats['total_claims']} documents\n")
    status_text.append(f"├── Vector Index: Ready ", style="green")
    status_text.append("✓\n", style="green bold")
    status_text.append(f"├── Summary Index: 3 hierarchy levels\n")
    status_text.append(f"└── MCP Tools: {len(mcp_tools)} available\n")

    console.print(Panel(status_text, border_style="cyan", box=box.ROUNDED))


def main_menu():
    """Show main menu and return choice."""
    choices = [
        questionary.Choice("  Query Mode         Ask questions about claims", value="query"),
        questionary.Choice("  Run Evaluation     Basic model-based evaluation", value="eval"),
        questionary.Choice("  Multi-Grader Eval  All graders + HTML report", value="graders"),
        questionary.Choice("  Human Grading      Manually grade responses", value="human"),
        questionary.Choice("  Show Statistics    View system metrics", value="stats"),
        questionary.Choice("  MCP Status         Check MCP integration", value="mcp"),
        questionary.Choice("  Help               Usage guide", value="help"),
        questionary.Choice("  Exit", value="exit"),
    ]

    return questionary.select(
        "What would you like to do?",
        choices=choices,
        style=custom_style,
        pointer=">"
    ).ask()


def run_query_mode(system: dict):
    """Interactive query mode."""
    console.print("\n[bold cyan]Query Mode[/bold cyan]")
    console.print("[dim]Type 'back' to return to main menu[/dim]\n")

    router = system['router']

    while True:
        query = questionary.text(
            "Your question:",
            style=custom_style
        ).ask()

        if query is None or query.lower() == 'back':
            break

        if not query.strip():
            continue

        with console.status("[cyan]Processing query...[/cyan]"):
            try:
                response, metadata = router.query_with_metadata(query)
                routed_to = metadata.get('routed_to', 'unknown')

                console.print(f"\n[bold cyan]Agent:[/bold cyan] {routed_to}")
                console.print(f"[bold cyan]Response:[/bold cyan]\n{response}\n")

                # Show sources if available
                if 'source_nodes' in metadata and metadata['source_nodes']:
                    console.print(f"[dim]Sources: {len(metadata['source_nodes'])} chunks[/dim]")

            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {str(e)}")

        console.print()


def run_evaluation_mode(system: dict):
    """Run basic evaluation."""
    from core import run_evaluation

    console.print("\n[bold cyan]Running Basic Evaluation[/bold cyan]")

    # Ask for subset
    subset = questionary.text(
        "Number of queries (leave empty for all):",
        default="",
        style=custom_style
    ).ask()

    subset_val = int(subset) if subset and subset.isdigit() else None

    console.print()
    run_evaluation(system, subset=subset_val)


def run_graders_mode(system: dict):
    """Run multi-grader evaluation."""
    from core import run_graders_evaluation

    console.print("\n[bold cyan]Multi-Grader Evaluation[/bold cyan]")
    console.print("[dim]Based on Anthropic's 'Demystifying Evals for AI Agents'[/dim]\n")

    # Ask for options
    subset = questionary.text(
        "Number of queries (leave empty for all):",
        default="",
        style=custom_style
    ).ask()

    include_human = questionary.confirm(
        "Include human grades (if available)?",
        default=False,
        style=custom_style
    ).ask()

    subset_val = int(subset) if subset and subset.isdigit() else None

    console.print()
    run_graders_evaluation(
        system,
        subset=subset_val,
        include_human=include_human,
        output_html=True
    )


def run_human_grading():
    """Launch human grading CLI."""
    console.print("\n[bold cyan]Human Grading[/bold cyan]")
    console.print("[dim]Launching human grading interface...[/dim]\n")

    # Import and run
    from src.graders.human_graders import HumanGraderStore, HumanGraderCLI
    from pathlib import Path

    store = HumanGraderStore()
    cli = HumanGraderCLI(store, grader_id="default")

    responses_path = Path("eval_runs/responses_to_grade.json")
    if not responses_path.exists():
        console.print("[yellow]Warning:[/yellow] No responses to grade found.")
        console.print("Run Multi-Grader Eval first to generate responses.\n")
        return

    graded = cli.grade_batch(responses_path)
    console.print(f"\n[green]Total graded: {graded}[/green]\n")


def show_statistics(system: dict):
    """Display detailed statistics."""
    from core import show_statistics as main_show_statistics
    console.print()
    main_show_statistics(system)


def show_mcp_status(system: dict):
    """Show MCP integration status."""
    from core import show_mcp_status as main_show_mcp
    console.print()
    main_show_mcp(system)


def show_help():
    """Display help and usage guide."""
    help_table = Table(title="Usage Guide", box=box.ROUNDED, border_style="cyan")
    help_table.add_column("Agent", style="cyan")
    help_table.add_column("Query Type", style="white")
    help_table.add_column("Example", style="dim")

    help_table.add_row(
        "STRUCTURED",
        "SQL queries, filters, aggregations",
        "Show me all claims over $100,000"
    )
    help_table.add_row(
        "SUMMARY",
        "Overviews, timelines, narratives",
        "What happened in claim CLM-2024-003012?"
    )
    help_table.add_row(
        "NEEDLE",
        "Precise facts, numbers, references",
        "What was the exact towing cost?"
    )

    console.print()
    console.print(help_table)

    console.print("\n[bold cyan]Example Queries:[/bold cyan]")
    console.print("  • Get claim CLM-2024-001847")
    console.print("  • What is the wire transfer reference for the life insurance payout?")
    console.print("  • Give me an overview of all auto-related claims")
    console.print("  • How long was the coffee spill on the floor?")
    console.print()


def initialize_system():
    """Initialize the RAG system."""
    from core import initialize_system as main_init
    return main_init(skip_cleanup_prompt=True)


def run_interactive():
    """Main interactive loop."""
    console.clear()
    show_banner()

    # Initialize system with progress spinner
    system = None
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Initializing system...[/bold cyan]"),
        console=console
    ) as progress:
        task = progress.add_task("init", total=None)
        try:
            system = initialize_system()
        except Exception as e:
            console.print(f"\n[bold red]Initialization failed:[/bold red] {str(e)}")
            console.print("[dim]Make sure .env file exists with API keys[/dim]")
            return

    console.print("[green]✓ System ready![/green]\n")
    show_status(system)

    while True:
        try:
            choice = main_menu()

            if choice is None or choice == "exit":
                console.print("\n[cyan]Goodbye! Happy claims processing![/cyan]\n")
                break
            elif choice == "query":
                run_query_mode(system)
            elif choice == "eval":
                run_evaluation_mode(system)
            elif choice == "graders":
                run_graders_mode(system)
            elif choice == "human":
                run_human_grading()
            elif choice == "stats":
                show_statistics(system)
            elif choice == "mcp":
                show_mcp_status(system)
            elif choice == "help":
                show_help()

        except KeyboardInterrupt:
            console.print("\n\n[cyan]Goodbye! Happy claims processing![/cyan]\n")
            break
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
            console.print("[dim]Returning to main menu...[/dim]\n")


def main():
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Insurance Claims RAG System - Interactive CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py              # Interactive mode
  python main.py --query      # Quick query mode
  python main.py --eval       # Run evaluation
  python main.py --graders    # Multi-grader evaluation
        """
    )

    parser.add_argument('--query', '-q', action='store_true',
                        help='Start in query mode')
    parser.add_argument('--eval', '-e', action='store_true',
                        help='Run basic evaluation')
    parser.add_argument('--graders', '-g', action='store_true',
                        help='Run multi-grader evaluation')
    parser.add_argument('--subset', '-s', type=int, default=None,
                        help='Limit number of queries for evaluation')

    args = parser.parse_args()

    # Non-interactive modes
    if args.eval or args.graders:
        show_banner()
        console.print("[cyan]Initializing...[/cyan]\n")
        system = initialize_system()

        if args.eval:
            from core import run_evaluation
            run_evaluation(system, subset=args.subset)
        elif args.graders:
            from core import run_graders_evaluation
            run_graders_evaluation(system, subset=args.subset, output_html=True)
        return

    # Interactive mode
    run_interactive()


if __name__ == "__main__":
    main()
