import subprocess
import sys

def run_subscript(console, script_path, step_name: str, progress) -> None:
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