import random
import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

app = typer.Typer(help="VECANOVA AI Interactive Mock Interviewer CLI")
console = Console()

QUESTIONS = {
    "system_design": [
        "Design a Clinical Analysis Copilot handling 10,000 requests/sec with zero PII leakage. Walk me through your hybrid retrieval and state management architecture.",
        "How would you design a real-time anomaly detection system for financial transactions or healthcare telemetry using XGBoost and GenAI agents?"
    ],
    "llmops": [
        "Explain how RAGAS metrics (Faithfulness and Answer Relevance) are computed in a production CI/CD pipeline.",
        "How do you trace cumulative token costs and latency bottlenecks across a multi-agent LangGraph workflow?"
    ],
    "behavioral": [
        "Tell me about a time you disagreed with a product manager regarding an AI feature release timeline. How did you resolve it?",
        "Describe a situation where a model deployed in production experienced severe concept drift. How did you debug and fix it?"
    ]
}

@app.command()
def interview(topic: str = typer.Option("system_design", help="Topic: system_design, llmops, behavioral")):
    """
    Run an interactive mock interview session in the terminal.
    """
    console.print(Panel.fit("🎯 VECANOVA AI Interview Simulator", style="bold cyan"))
    
    if topic not in QUESTIONS:
        console.print(f"[red]Invalid topic. Choose from: {list(QUESTIONS.keys())}[/red]")
        return
        
    question = random.choice(QUESTIONS[topic])
    
    console.print(f"\n[bold yellow]Mock Interview Question ({topic.upper()}):[/bold yellow]\n")
    console.print(Panel(question, style="green"))
    
    console.print("\n[dim]Tip: Structure your answer using the STAR method in English (Situation, Task, Action, Result).[/dim]\n")
    
    user_answer = console.input("[bold cyan]Type your verbal response outline (or press Enter to view recommended STAR pitch):[/bold cyan] ")
    
    console.print("\n[bold green]Recommended STAR Pitch & Key Technical Talking Points:[/bold green]\n")
    if topic == "system_design":
        pitch = "Emphasize hybrid retrieval (Dense + BM25 + Rerank), PII sanitization via Presidio, and deterministic state control using LangGraph."
    elif topic == "llmops":
        pitch = "Mention RAGAS Faithfulness & Answer Relevance auditing, Langfuse tracing, and automated regression testing in CI/CD."
    else:
        pitch = "Use the STAR framework: Situation, Task, Action, Result. Focus on measurable business impact and cross-functional collaboration."
        
    console.print(Panel(pitch, style="blue"))
    console.print("\n[bold magenta]Mock session completed successfully. Keep sharpening your craft![/bold magenta]")

if __name__ == "__main__":
    app()
