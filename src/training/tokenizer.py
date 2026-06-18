import json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProcessor

"""
    This scripts trains a new tokenizer over the complete dataset
"""

def train_bpe_tokenizer(data_files: list[str], output_path: str="latin_bpe_tokenizer.json", vocab_size: int=32000):

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    # Configure trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"], # padding, sentence start/end, unknown, masking
        show_progress=True
    )

    # Streaming train data step by step
    def get_training_corpus():
        for jsonl_path in data_files:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    yield data["text"]

    print(f"[START] Training tokenizer with vocab size: {vocab_size}")
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
    tokenizer.post_processor = ByteLevelProcessor(trim_offsets=False)
    tokenizer.save(output_path)
    print(f"[END] Tokenizer saved to '{output_path}'")