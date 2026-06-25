import math
import os
from typing import List, Tuple, Dict, Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerFast, DataCollatorForLanguageModeling

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


def top_k_accuracies(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerFast,
        texts: List[str],
        k = 5,
        mlm_probability=0.15,
        batch_size: int=16
) -> Tuple[float, float]:
    """ Calculates top k accuracies for the given eval texts """
    model.eval()

    # Masking
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability,
    )
    total_masked_tokens = 0
    top_1_correct = 0
    top_k_correct = 0
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i: i + batch_size]

            encoded_inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_special_tokens_mask=True
            )
            features = [
                {key: val[j] for key, val in encoded_inputs.items()}
                for j in range(len(batch_texts))
            ]

            collated_batch = data_collator(features)
            inputs = {key: val.to(model.device) for key, val in collated_batch.items()}
            # Calling inference
            outputs = model(**inputs)
            logits = outputs.logits
            labels = inputs["labels"]
            mask = labels != -100

            masked_logits = logits[mask]  # Shape of [masks, vocab_size]
            masked_labels = labels[mask]

            if masked_labels.numel() == 0:
                continue

            _, top_k_indices = torch.topk(masked_logits, k, dim=-1)
            top_1_correct += (top_k_indices[:, 0] == masked_labels).sum().item()

            top_k_correct += (top_k_indices == masked_labels.unsqueeze(-1)).sum().item()

    if total_masked_tokens == 0:
        return 0.0, 0.0
    return round(top_1_correct / total_masked_tokens, 4), round(top_k_correct / total_masked_tokens, 4)


def extract_sentence_embeddings(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerFast,
        texts: List[str],
        batch_size: int = 16
) -> List[Dict[str, Any]]:
    """ Computes the CLS (<s>) token embedding for each given sentence. """
    model.eval()
    results = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i: i + batch_size]

            encoded_inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            inputs = {key: val.to(model.device) for key, val in encoded_inputs.items()}
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]
            cls_embeddings = last_hidden_state[:, 0, :]
            cls_embeddings_cpu = cls_embeddings.cpu().numpy()
            for j, text in enumerate(batch_texts):
                results.append({
                    "input": text,
                    "embedding_vec": cls_embeddings_cpu[j].tolist()
                })
    return results


def find_top_k_similar_sentences(
    embeddings_data: List[Dict[str, Any]],
    k: int = 3
) -> List[Dict[str, Any]]:
    """ Finds top k similar sentences with cosine similarity w.r.t to the embedding vectors """
    if not embeddings_data:
        return []

    texts = [item["input"] for item in embeddings_data]
    vectors = torch.tensor([item["embedding_vec"] for item in embeddings_data])
    vectors_normalized = torch.nn.functional.normalize(vectors, p=2, dim=1)
    similarity_matrix = torch.matmul(vectors_normalized, vectors_normalized.T)
    results = []

    for i, text in enumerate(texts):
        sims = similarity_matrix[i]

        top_values, top_indices = torch.topk(sims, k + 1)

        similar_sentences = []
        for val, idx in zip(top_values, top_indices):
            if idx.item() != i and len(similar_sentences) < k:
                similar_sentences.append({
                    "text": texts[idx.item()],
                    "similarity_score": round(val.item(), 4)
                })

        results.append({
            "original_sentence": text,
            "top_similar_sentences": similar_sentences
        })

    return results