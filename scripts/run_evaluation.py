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
    parser.add_argument("--eval-llama", action="store_true", help="Which model type to use")

    parser.add_argument("--set-checkpoint", type=str, default=None, help="Checkpoint to use for evaluation")
    parser.add_argument("--set-evaldata", type=str, default=None, help="Set data used for evaluation")
    parser.add_argument("--set-outputdir", type=str, default=None, help="Set output directory for results")

    # Llama specific args
    parser.add_argument("--skip-inference", action="store_true", help="Skips inference run of Llama")
    parser.add_argument("--skip-judgment", action="store_true", help="Skips inference run of Llama")

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
            extra_args = [
                "--set-checkpoint", args.set_checkpoint,
                "--set-evaldata", args.set_evaldata,
                "--set-outputdir", args.set_outputdir,
            ]
            try:
                run_subscript(console,
                              "scripts.evaluation.01_eval_roberta",
                              "Evaluation",
                              progress,
                              extra_args=extra_args)
            except Exception as e:
                print(f"[ERROR] Following error occurred during evaluation: {e}")
            progress.update(roberta_task, description=f"[green]✓ 1. Evaluation step successful![/green]",
                            total=1, completed=1)

    elif args.eval_llama:
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
            llama_task = progress.add_task(f"[cyan] 1. Evaluation of Llama at checkpoint '{checkpoint_path}' [/cyan]",)
            extra_args = [
                "--set-checkpoint", args.set_checkpoint,
                "--set-evaldata", args.set_evaldata,
                "--set-outputdir", args.set_outputdir,
                "--skip-inference", args.skip_inference,
                "--skip-judgment", args.skip_judgment
            ]
            try:
                run_subscript(console,
                              "scripts.evaluation.02_eval_llama",
                              "Evaluation",
                              progress,
                              extra_args=extra_args)
            except Exception as e:
                print(f"[ERROR] Following error occurred during evaluation: {e}")
            progress.update(llama_task, description=f"[green]✓ 1. Evaluation step successful![/green]",
                            total=1, completed=1)

    else:
        console.print(f"[yellow]>> No model selected for evaluation ![/yellow]")
        console.print(f"[yellow]>> Choose one of the following options:[/yellow]")
        console.print(f"[yellow]  --eval-roberta [/yellow]")
        console.print(f"[yellow]  --eval-llama [/yellow]")


if __name__ == "__main__":
    main()
