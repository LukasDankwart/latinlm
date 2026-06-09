import json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProcessor


def train_latin_tokenizer(jsonl_path, output_name="latin_bpe_tokenizer.json", vocab_size=32000):

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
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                yield data["text"]

    print(f"[START] Training tokenizer with vocab size: {vocab_size}")
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
    tokenizer.post_processor = ByteLevelProcessor(trim_offsets=False)
    tokenizer.save(output_name)
    print(f"[FINISH] Tokenizer saved to '{output_name}'")