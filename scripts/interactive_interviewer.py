"""Interactive mock interviewer backed by the packaged question bank."""

import random

import typer
from rich.console import Console
from rich.panel import Panel

from interview_prep.questions import VALID_TOPICS, filter_questions

app = typer.Typer(help="VECANOVA AI Interactive Mock Interviewer CLI")
console = Console()


@app.command()
def interview(
    topic: str = typer.Option(
        None, help=f"Topic filter: {', '.join(sorted(VALID_TOPICS))}"
    ),
    difficulty: str = typer.Option(None, help="Difficulty filter: mid or senior"),
    count: int = typer.Option(1, min=1, max=10, help="Questions per session"),
) -> None:
    """Run an interactive mock interview session graded by rubric."""
    console.print(Panel.fit("VECANOVA AI Interview Simulator", style="bold cyan"))
    try:
        selected = filter_questions(topic, difficulty)
    except ValueError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=1)

    for question in random.sample(selected, min(count, len(selected))):
        header = (
            f"[{question['id']}] {question['topic'].upper()} - "
            f"{question['difficulty']} - {question['industry']}"
        )
        console.print(f"\n[bold yellow]{header}[/bold yellow]")
        console.print(Panel(question["question"], style="green"))
        console.input(
            "[bold cyan]Outline your answer, then press Enter for the rubric:[/bold cyan] "
        )
        rubric = "\n".join(
            f"{i}. {point}" for i, point in enumerate(question["rubric"], start=1)
        )
        console.print(Panel(rubric, title="Grading rubric", style="blue"))

    console.print(
        "\n[bold magenta]Session complete. Grade yourself against the rubric.[/bold magenta]"
    )


if __name__ == "__main__":
    app()
