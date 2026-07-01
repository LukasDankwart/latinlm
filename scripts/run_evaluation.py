import argparse
import os

from rich.panel import Panel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from src.utils.script_utils import run_subscript

console = Console()

def main():
    parser = argparse.ArgumentParser(description="Running evaluation of specified model")

    parser.add_argument("--eval-roberta", action="store_true", help="Which model type to use")
    parser.add_argument("--set-checkpoint", type=str, default=None, help="Checkpoint to use for evaluation")

    args = parser.parse_args()
    console.print(Panel.fit("[bold magenta] LatinLM - Model Evaluation [/bold magenta]"))

    if args.eval_roberta:
        checkpoint_path = args.set_checkpoint
        if checkpoint_path is None:
            console.print(f"[yellow]>> No checkpoint specified [/yellow]")
            console.print(f"[yellow]>> Please specify checkpoint to use by '--set-checkpoint=...' [/yellow]")
            return
        if not os.path.isdir(checkpoint_path):
            console.print(f"[yellow]>> Specified path '{checkpoint_path}' is not a valid checkpoint folder! [/yellow]")
            return

        with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=console,
        ) as progress:
            roberta_task = progress.add_task(f"[cyan] 1. Evaluation of RoBERTa at checkpoint '{checkpoint_path}' [/cyan]",)
            extra_args = ["--set-checkpoint", args.set_checkpoint]
            run_subscript(console,
                          "scripts.evaluation.01_eval_roberta",
                          "Evaluation",
                          progress,
                          extra_args=extra_args)
            progress.update(roberta_task, description=f"[green]✓ 1. Evaluation step successful![/green]",
                            total=1, completed=1)

    else:
        console.print(f"[yellow]>> No model selected for evaluation ![/yellow]")
        console.print(f"[yellow]>> Choose one of the following options:[/yellow]")
        console.print(f"[yellow]  --eval-roberta [/yellow]")


if __name__ == "__main__":
    main()
