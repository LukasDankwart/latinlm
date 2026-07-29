from datasets import load_from_disk
from transformers import (
    LlamaTokenizerFast, DataCollatorForLanguageModeling, LlamaForCausalLM, Trainer, TrainingArguments
)
from src.utils.utils import load_yaml_config
from src.models.Llama import load_llama_from_config
import wandb


def main():
    # 1. Step: ───── Load model config file ─────
    config_path = "configs/llama.yaml"
    model_config = load_yaml_config(config_path)
    print(f"[yellow] -- [INFO] Model config successfully loaded...")

    # 2. Step: ───── Create tokenizer instance & load dataset ─────
    tokenizer_args = model_config["tokenizer"]
    tokenizer_path = tokenizer_args["tokenizer_path"]

    tokenizer = LlamaTokenizerFast.from_pretrained(
        tokenizer_path,
        bos_token=tokenizer_args["bos_token"],
        eos_token=tokenizer_args["eos_token"],
        unk_token=tokenizer_args["unk_token"],
        pad_token=tokenizer_args["pad_token"],
    )
    dataset_args = model_config["data"]
    data_path = dataset_args["path"]
    lm_dataset = load_from_disk(data_path)
    train_dataset = lm_dataset["train"]
    test_dataset = lm_dataset["test"]
    print(f"[yellow] -- [INFO] Dataset successfully initialized...")

    # Count for each split how many tokens are included
    def count_tokens(batch):
        return {"token_count": [len(seq) for seq in batch["input_ids"]]}

    counted_dataset = lm_dataset.map(count_tokens, batched=True, desc="Zähle Tokens...")
    total_tokens = 0
    for split_name, split_data in counted_dataset.items():
        split_tokens = sum(split_data["token_count"])
        print(f"-- [INFO] Number of tokens from '{split_name}': {split_tokens:,}")
        total_tokens += split_tokens

    # 3. Step: ───── Initialize model instance ─────
    model_args = model_config["model"]
    model_args["vocab_size"] = len(tokenizer)
    model_args["pad_token_id"] = tokenizer.pad_token_id
    model_args["bos_token_id"] = tokenizer.bos_token_id
    model_args["eos_token_id"] = tokenizer.eos_token_id

    model, llama_config = load_llama_from_config(config_path)

    # 4. Step: ───── Initialize data collator for CausalLM and trainer ─────
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    training_config = model_config["train"]
    meta_config = model_config["meta"]
    report_to = meta_config["report_to"]
    if report_to is not None:
        if report_to == "wandb":
            wandb.init(
                project=meta_config["project_name"],
                name=meta_config["run_name"],
                tags=[meta_config["tags"]],
                group=meta_config["group"],
            )
            training_config["run_name"] = meta_config["run_name"]
            print(f"[yellow] -- [INFO] Training procedure will be reported to '{report_to}'!")

    training_args = TrainingArguments(
        **training_config,
        report_to=report_to
    )
    print(f"[yellow] -- [INFO] Fetched training arguments successfully...")

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    print(f"[yellow] -- [INFO] Initialized Trainer successfully...")

    print(f"[cyan] [INFO] Starting training...")

    trainer.train()

    if report_to is not None:
        if report_to == "wandb":
            wandb.finish()

if __name__ == "__main__":
    main()