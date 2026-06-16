import argparse
from rich.panel import Panel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from src.utils.script_utils import run_subscript

console = Console()

def main():
    parser = argparse.ArgumentParser(description="Running tokenizer training and evaluation.")

    # ──────────────── Parser Arguments ────────────────
    parser.add_argument("--skip-train", action="store_true", help="Skips training of tokenizer")
    parser.add_argument("--skip-eval", action="store_true", help="Skips scraping specified websites")

    args = parser.parse_args()
    console.print(Panel.fit("[bold magenta] LatinLM - Tokenizer Train & Eval [/bold magenta]"))

    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
    ) as progress:

        # 1. Step: ───────── Calling the tokenizer train script ───────────────
        if not args.skip_train:
            train_task = progress.add_task("[cyan] 1. Training step...")
            run_subscript(console, "scripts.tokenizer.01_train_tokenizer", "Training", progress)
            progress.update(train_task, description=f"[green]✓ 1. Training step successful![/green]",
                            total=1, completed=1)
        else:
            console.print(f"[yellow]>> Skip training of tokenizer. [/yellow]")

        # 2. Step: ───────── Calling the tokenizer evaluation script ───────────────
        if not args.skip_eval:
            eval_task = progress.add_task("[cyan] 2. Evaluation step...")
            run_subscript(console, "scripts.tokenizer.02_test_tokenizer", "Eval", progress)
            progress.update(eval_task, description=f"[green]✓ 2. Evaluation step successful![/green]",
                            total=1, completed=1)
        else:
            console.print(f"[yellow]>> Skip evaluation of tokenizer. [/yellow]")

if __name__ == "__main__":
    main()