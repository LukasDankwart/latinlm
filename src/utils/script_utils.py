import subprocess
import sys
from typing import Tuple

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import traceback

def run_subscript(console, script_path, step_name: str, progress, extra_args: list = None) -> None:
    """ Runs a specified extern python script as subprocess and catches exceptions"""
    if extra_args is None:
        extra_args = []

    console.print(f"\n [yellow] Starting {step_name} step...[/yellow]")

    cmd = ["uv", "run", "python", "-u", "-m", script_path] + extra_args

    process = subprocess.Popen(
        cmd,
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


def load_model_and_tokenizer(checkpoint_path: str) -> Tuple[AutoModelForMaskedLM, AutoTokenizer]:
    """ Loads model from specified checkpoint path """

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=True)

    model = AutoModelForMaskedLM.from_pretrained(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"-- [green] Loading model from {checkpoint_path} was successful![/green]")

    return model, tokenizer