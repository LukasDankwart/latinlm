import argparse
from rich.panel import Panel
from rich.console import Console
from rich.prompt import Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from src.utils.script_utils import run_subscript

console = Console()

def main():
    parser = argparse.ArgumentParser(description="Running model Training.")

    # ──────────────── Parser Arguments ────────────────
    parser.add_argument("--train-roberta", action="store_true", help="Training of RoBERTa")

    args = parser.parse_args()
    console.print(Panel.fit("[bold magenta] LatinLM - Model Training [/bold magenta]"))

    """
        Depending on the given argument, the model that should be trained is selected
    """

    # 1. Step: ───────── Train RoBERTa ───────────────
    if args.train_roberta:
        confirm_config = Confirm.ask("[yellow] Does the config file for RoBERTa exists at 'configs/roberta.yaml'?")
        if confirm_config:
            with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    TimeElapsedColumn(),
                    console=console,
            ) as progress:
                roberta_task = progress.add_task("[cyan] 1. Training of RoBERTa")
                run_subscript(console, "scripts.training.01_train_roberta", "Training", progress)
                progress.update(roberta_task, description=f"[green]✓ 1. Training step successful![/green]",
                                total=1, completed=1)
        else:
            console.print(f"[red] Training aborted. Please create a model config file at 'configs/roberta.yaml'!")
    else:
        console.print(f"[yellow]>> No model selected for training![/yellow]")
        console.print(f"[yellow]>> Choose one of the following options:[/yellow]")
        console.print(f"[yellow]  --train-roberta [/yellow]")


if __name__ == "__main__":
    main()