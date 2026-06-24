import math
import os
from typing import List

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerFast

""" Metrics for evaluating the RoBERTa base variant """

def perplexity_score(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerFast,
        texts: List[str],
        batch_size: int=16
) -> float:
    """ Computes the perplexity score for the model over the given eval texts """
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i: i + batch_size]

            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )

            inputs["labels"] = inputs["input_ids"].clone()
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model(**inputs)
            loss = outputs.loss
            num_tokens = inputs["attention_mask"].sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    if total_tokens == 0:
        return float("inf")

    average_loss = total_loss / total_tokens
    perplexity = math.exp(average_loss)
    return perplexity

