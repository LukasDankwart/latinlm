# ─── Directory paths ────────────────────────────
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
TOKENIZED_DATA_DIR = "data/tokenized"


# ─── Datapipeline Parameters ────────────────────────────
DATASETS_TO_DOWNLOAD = [
    #{
    #    "name":  "PleIAs/Latin-PD",
    #    "subset": None,
    #},
    {
        "name": "HuggingFaceFW/fineweb-2",
        "subset": "lat_Latn",
    },
    {
        "name": "wikimedia/wikipedia",
        "subset": "20231101.la",
    },
    {
        "name": "cltk",
        "subset": [
            "lat_text_perseus",
            "lat_text_latin_library",
            "lat_text_tesserae",
            "latin_text_antique_digiliblt",
            "latin_text_poeti_ditalia"
        ]
    }
]
DATASETS_SAMPLE_LIMIT = None


PRELOADED_DATASETS = [
    {
        "name": "GreLa",
        "path": "data/raw/grela/grela_v0.6.duckdb"
    },
    {
        "name": "canonical-latinLit",
        "path": "data/raw/canonical-latinLit"
    }
]

NUNTII_BASE_URL = "https://nuntiilatini.com/page/"
VATICAN_BASE_API_URL = "https://www.vaticannews.va/bin/servlet/solr/search"



# ─── Hardware Parameters ────────────────────────────
NUM_CPU_WORKERS = 12


# ─── Tokenizer Parameters ────────────────────────────
TOKENIZER_VOCAB_SIZE = 32768
TOKENIZER_PATH = "data/tokenized/bpe_tokenizer.json"