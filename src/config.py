# ─── Directory paths ────────────────────────────
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
TOKENIZED_DATA_DIR = "data/tokenized"


# ─── Datapipeline Parameters ────────────────────────────
DATASETS_TO_DOWNLOAD = [
    {
        "name":  "PleIAs/Latin-PD",
        "subset": None,
    },
    {
        "name": "HuggingFaceFW/fineweb-2",
        "subset": "lat_Latn",
    }
]
DATASETS_SAMPLE_LIMIT = 100

NUNTII_BASE_URL = "https://nuntiilatini.com/page/"
VATICAN_BASE_API_URL = "https://www.vaticannews.va/bin/servlet/solr/search"
