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
    """ 
    Computes the pseudo-perplexity score for the MLM over the given eval texts.
    Correctly applies MLM masking and ignores unmasked tokens.
    """
    model.eval()

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i: i + batch_size]

            encoded = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                return_special_tokens_mask=True
            )

            input_ids, labels = data_collator.torch_mask_tokens(
                encoded["input_ids"].clone(),
                encoded.get("special_tokens_mask")
            )

            inputs = {
                "input_ids": input_ids.to(model.device),
                "attention_mask": encoded["attention_mask"].to(model.device),
                "labels": labels.to(model.device)
            }

            outputs = model(**inputs)
            
            if outputs.loss is not None:
                total_loss += outputs.loss.item()
                total_batches += 1

    if total_batches == 0:
        return float("inf")

    average_loss = total_loss / total_batches
    perplexity = math.exp(average_loss)
    return round(perplexity, 4)

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
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=True, 
        mlm_probability=0.15
    )
    
    total_masked_tokens = 0
    top_1_correct = 0
    top_k_correct = 0
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            
            encoded_inputs = tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=512,
                return_special_tokens_mask=True,
                return_tensors="pt"
            )
            
            input_ids, labels = data_collator.torch_mask_tokens(
                encoded_inputs["input_ids"].clone(), 
                encoded_inputs.get("special_tokens_mask")
            )
            
            inputs = {
                "input_ids": input_ids.to(model.device),
                "attention_mask": encoded_inputs["attention_mask"].to(model.device),
                "labels": labels.to(model.device)
            }
            
            outputs = model(**inputs)
            logits = outputs.logits  # Shape: [Batch_size, Seq_len, Vocab_size]
            
            mask = inputs["labels"] != -100
            
            masked_logits = logits[mask] 
            masked_labels = inputs["labels"][mask] 
            
            if masked_labels.numel() == 0:
                continue 
                
            total_masked_tokens += masked_labels.numel()
            
            _, top_k_indices = torch.topk(masked_logits, k, dim=-1)
            
            top_1_correct += (top_k_indices[:, 0] == masked_labels).sum().item()
            top_k_correct += (top_k_indices == masked_labels.unsqueeze(-1)).sum().item()

    if total_masked_tokens == 0:
        return {"top_1_accuracy": 0.0, f"top_{k}_accuracy": 0.0}
        
    return {
        "top_1_accuracy": round(top_1_correct / total_masked_tokens, 4),
        f"top_{k}_accuracy": round(top_k_correct / total_masked_tokens, 4)
    }

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