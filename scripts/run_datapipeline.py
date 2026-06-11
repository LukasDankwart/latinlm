import argparse
import subprocess
import sys
from rich.panel import Panel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

console = Console()

def run_subscript(script_path, step_name: str, progress) -> None:
    """ Runs a specified extern python script as subprocess and catches exceptions"""

    console.print(f"\n [yellow] Starting {step_name} step...[/yellow]")

    process = subprocess.Popen(
        ["uv", "run", "python", "-u", "-m", script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    #subprocess.run(["uv", "run", "python", "-m", script_path], check=True)
    for line in process.stdout:
        progress.console.print(line.strip())
    process.wait()

    if process.returncode != 0:
        console.print(f"[bold red]✗ Error in {step_name} step...[/bold red]")
        console.print(f"[bold red]- Pipeline will be stopped [/bold red]")
        console.print(f"[bold red]- Error code: {process.returncode}[/bold red]")
        sys.exit(1)

    console.print(f"[green]✓ {step_name} was successful![/green]")

def main():
    parser = argparse.ArgumentParser(description="Running complete Pipeline for Latin data aggregation.")

    # ──────────────── Parser Arguments ────────────────
    parser.add_argument("--skip-download", action="store_true", help="Skips downloading pre-existing datasets")
    parser.add_argument("--skip-scraping", action="store_true", help="Skips scraping specified websites")
    parser.add_argument("--skip-preprocessing", action="store_true", help="Skips preprocessing crawled data")
    parser.add_argument("--skip-tokenize", action="store_true", help="Skips tokenization of preprocesses data")
    parser.add_argument("--skip-binarize", action="store_true", help="Skips binarization of data")

    args = parser.parse_args()
    console.print(Panel.fit("[bold magenta] LatinLM - End-to-End Data Pipeline [/bold magenta]"))

    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
    ) as progress:

        # 1. Step: ───────── Downloading pre-existing datasets from huggingface ───────────────
        if not args.skip_download:
            download_task = progress.add_task("[cyan] Download step...")
            run_subscript("scripts.pipeline.01_run_download", "Download", progress)
            progress.update(download_task, description=f"[green]✓ Download step successful![/green]",
                            total=1, completed=1)
        else:
            console.print(f"[yellow]>> Skip downloading pre-existing datasets. [/yellow]")

        # 2. Step: ───────── Scraping specified data websites ───────────────
        if not args.skip_scraping:
            scrape_task = progress.add_task("[cyan] Scraping step...")
            run_subscript("scripts.pipeline.02_run_scraping", "Scraping", progress)
            progress.update(scrape_task, description=f"[green]✓ Scraping step successful![/green]",
                            total=1, completed=1)
        else:
            console.print(f"[yellow]>> Skip scraping websites. [/yellow]")


if __name__ == "__main__":
    main()

