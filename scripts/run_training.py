import argparse
from rich.panel import Panel
from rich.console import Console
from rich.prompt import Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
import os
from src.utils.utils import load_yaml_config

from src.utils.script_utils import run_subscript

console = Console()

def main():
    parser = argparse.ArgumentParser(description="Running model Training.")

    # ──────────────── Parser Arguments ────────────────
    # Arg for pre-training RoBERTa
    parser.add_argument("--train-roberta", action="store_true", help="Training of RoBERTa")
    # Args for POS training RoBERTa
    parser.add_argument("--train-roberta-pos", action="store_true", help="Train class head on RoBERTa for pos")
    parser.add_argument("--set-checkpoint", type=str, default=None, help="Specify checkpoint for POS tagging")
    parser.add_argument("--freeze-base", action="store_true", help="Does freeze all RoBERTa weights")
    parser.add_argument("--set-dataset", type=str, default=None, help="Specify which dataset should be used.")
    # Args for pre-training Llama
    parser.add_argument("--train-llama", action="store_true", help="Training of Llama")

    args = parser.parse_args()
    console.print(Panel.fit("[bold magenta] LatinLM - Model Training [/bold magenta]"))

    """
        Depending on the given argument, the model that should be trained is selected
    """

    # 1. Case: ───────── Train RoBERTa ───────────────
    if args.train_roberta:
        config_path = "configs/roberta.yaml"
        if os.path.exists(config_path):
            # Check that the model config refers to the correct dataset and that it exists!
            conf = load_yaml_config(config_path)
            dataset_path = conf["data"]["path"]
            if "roberta" not in dataset_path.lower():
                raise RuntimeError(f"[ERROR] Training is aborted. Config file '{config_path}' "
                                   f"currently might refer to the wrong dataset, since 'roberta' "
                                   f"is not included in datasetpath.")
            if not os.path.exists(dataset_path):
                raise RuntimeError(f"[ERROR] Training aborted. Dataset directory does not exists at '{dataset_path}'. "
                                   f"Ensure that the pretokenization step of the datapipeline has been executed for "
                                   f"the RoBERTa model beforehand !"
                                   f"\n Note: This requires manually changing the dataset_config_path to 'configs/roberta_dataset.yaml' "
                                   f"at scripts/pipeline/05_run_tokenization.py")

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

    # 2. Case: ───────── Train RoBERTa for POS-Tagging ───────────────
    elif args.train_roberta_pos:
        # Check if checkpoint is given and valid
        if not args.set_checkpoint:
            console.print(f"[red] You have to set a checkpoint for starting POS learning by --set-checkpoint=...")
            return
        if not os.path.exists(args.set_checkpoint):
            console.print(f"[red] Given checkpoint path '{args.set_checkpoint}' is invalid!")
            return
        if not args.set_dataset:
            console.print(f"[red] No dataset was given. Use --set-dataset=... to one of the following names: ")
            console.print(f"[red] 'perseus', 'proiel', 'ittb' ")
            return
        if args.set_dataset not in ['perseus', 'proiel', 'ittb']:
            console.print(f"[red] Given dataset name is invalid. Please use: 'perseus', 'proiel' or 'ittb' ")


        if os.path.exists("configs/roberta_pos_freezed.yaml") or os.path.exists("configs/roberta_pos_full_ft.yaml"):
            with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    TimeElapsedColumn(),
                    console=console,
            ) as progress:

                roberta_task = progress.add_task(
                    f"[cyan] 1. POS-Training of RoBERTa from checkpoint '{args.set_checkpoint}' [/cyan]", )

                extra_args = ["--set-checkpoint", args.set_checkpoint, "--set-dataset", args.set_dataset]
                if args.freeze_base:
                    extra_args.extend(["--freeze-base" , "true"])

                run_subscript(console,
                              "scripts.training.02_train_roberta_pos",
                              "POS-Training",
                              progress,
                              extra_args=extra_args)
                progress.update(roberta_task, description=f"[green]✓ 1. POS-Training of RoBERTa successful![/green]",
                                total=1, completed=1)
        else:
            console.print(f"[red] Training aborted. Please create a model config file!")
            console.print(f"[red] -- For Full-Finetuning: configure 'configs/roberta_pos_full_ft.yaml'")
            console.print(f"[red] -- For only training Classification-Head: configure 'configs/roberta_pos_freezed.yaml'")

    # 3. Case: ───────── Train Llama ───────────────
    elif args.train_llama:
        # Specify config for current model
        config_path = "configs/llama.yaml"
        if os.path.exists(config_path):
            # Check that the model config refers to the correct dataset and that it exists!
            conf = load_yaml_config(config_path)
            dataset_path = conf["data"]["path"]
            if "llama" not in dataset_path.lower():
                raise RuntimeError(f"[ERROR] Training is aborted. Config file '{config_path}' "
                                   f"currently might refer to the wrong dataset, since 'llama' "
                                   f"is not included in datasetpath.")
            if not os.path.exists(dataset_path):
                raise RuntimeError(f"[ERROR] Training aborted. Dataset directory does not exists at '{dataset_path}'. "
                                   f"Ensure that the pretokenization step of the datapipeline has been executed for "
                                   f"the Llama model beforehand !"
                                   f"\n Note: This requires manually changing the dataset_config_path to 'configs/llama_dataset.yaml' "
                                   f"at scripts/pipeline/05_run_tokenization.py")

            with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    TimeElapsedColumn(),
                    console=console,
            ) as progress:
                roberta_task = progress.add_task("[cyan] 1. Training Llama")
                run_subscript(console, "scripts.training.03_train_llama", "Training", progress)
                progress.update(roberta_task, description=f"[green]✓ 1. Training step successful![/green]",
                                total=1, completed=1)
        else:
            console.print(f"[red] Training aborted. Please create a model config file at 'configs/llama.yaml'!")

    else:
        console.print(f"[yellow]>> No model selected for training![/yellow]")
        console.print(f"[yellow]>> Choose one of the following options:[/yellow]")
        console.print(f"[yellow]  --train-roberta (# Pre-Trains RoBERTa)[/yellow]")
        console.print(f"[yellow]  --train-roberta-pos (# Trains RoBERTa checkpoint for POS tagging) [/yellow]")
        console.print(f"[yellow]  --train-llama (# Pre-Trains Llama) [/yellow]")


if __name__ == "__main__":
    main()