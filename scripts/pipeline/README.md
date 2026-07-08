# Datapipeline
This README is desired to provide more details about the datapipeline and subsequently, the data corpus that is created.
Every progressing step of the pipeline will be described in the following, starting with an overview over all data sources
that are currently integrated and their amount of words/tokens:

| **Datasource**          | **Words** | **Tokens** |
|-------------------------|-----------|------------|
| CanonicalLatinLit       | 7.75M     | T          |
| CLTK                    | 17.68M    | T          |
| GreLa (Corpus Corporum) | 233.12M   | T          |
| FineWeb2 (Lat-Latn)     | 519.15M   | T          |
| Nuntii Latini           | 67k       | T          |
| Vatican News            | 136k      | T          |
| Wikimedia Wikipedia     | 13.03M    | T          |

The reported numbers are the amount of words after preprocessing (e.g. filtering) the raw data and the number of token 
comes from applying a BPE trained tokenizer to the data. 

The datapipeline consists of four stages, that are described in the following:
1. Download
2. Scraping
3. Preprocessing
4. Pre-Tokenization (& Binarization)

## 1. Download
This progressing step might require some manual downloads, since not all datasets can be fetched via APIs.
For example, the downloading step will automatically download most of the datasets and convert them to a new .jsonl file of
raw aggregated data. With the exception for GreLa, no manual downloads are required.

For integrating GreLa (which mostly consists of Corpus Corporum), all subsets of the database have to be downloaded manually 
from [zenodo](https://zenodo.org/records/17866143?preview_file=CCS-ZCU%2FGreLa-v0.6.zip). 
Afterwards please move all parts to the `".data/raw/grela"` directory and concatenate all .duckdb parts to one combined database file via:
```
cat grela_v0.6.duckdb.part* > grela_v0.6.duckdb  # Linux / Mac
copy /b grela_v0.6.duckdb.part* grela_v0.6.duckdb  # Windows
```
During this step, CanonicalLatinLit, CLTK, GreLa, Fineweb, WikimediaWikipedia will be processed and stored as raw .jsonl
in your `".data/raw"` folder.

## 2. Scraping
This step should not need any manuall adjustments, since data is scraped from the two remaining sources of VaticanNews 
and Nuntii Latini. Subsequently, raw .jsonl files will be stored at `".data/raw"`.

## 3. Pre-Processing
This procedure is the key aspect of the datapipeline, since the raw data must be filtered and normalized for the subsequent steps.
Currently, the preprocessing progress follows these sub-steps:
1. Each .jsonl of the previous mentioned datasource is processed isolated.
2. Each line/sample is interpreted as paragraph, that gets split to sentences.
3. Each sentence is filtered isolated:
   1. Normalization 
   2. Filtering via heuristics and language detector prediction
4. All leftover sentences are then re-concatenated to one paragraph.

## 4. Pre-Tokenization
