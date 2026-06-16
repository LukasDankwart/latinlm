import argparse
from rich.panel import Panel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from src.utils.script_utils import run_subscript

console = Console()

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
            run_subscript(console, "scripts.pipeline.01_run_download", "Download", progress)
            progress.update(download_task, description=f"[green]✓ Download step successful![/green]",
                            total=1, completed=1)
        else:
            console.print(f"[yellow]>> Skip downloading pre-existing datasets. [/yellow]")

        # 2. Step: ───────── Scraping specified data websites ───────────────
        if not args.skip_scraping:
            scrape_task = progress.add_task("[cyan] Scraping step...")
            run_subscript(console, "scripts.pipeline.02_run_scraping", "Scraping", progress)
            progress.update(scrape_task, description=f"[green]✓ Scraping step successful![/green]",
                            total=1, completed=1)
        else:
            console.print(f"[yellow]>> Skip scraping websites. [/yellow]")

        # 3. Step: ───────── Calling pre-processing scripts ───────────────
        if not args.skip_preprocessing:
            scrape_task = progress.add_task("[cyan] Preprocessing step...")
            run_subscript(console, "scripts.pipeline.03_run_preprocessing", "Pre-processing", progress)
            progress.update(scrape_task, description=f"[green]✓ Pre-processing step successful![/green]",
                            total=1, completed=1)
        else:
            console.print(f"[yellow]>> Skip Pre-processing websites. [/yellow]")


if __name__ == "__main__":
    main()

