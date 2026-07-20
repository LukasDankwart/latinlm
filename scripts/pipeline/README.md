# Description of the LatinLM - Datapipeline
The datapipeline consists of various downloading, scraping and preprocessing steps, that are explained in more detail here.
Currently, the following data source are integrated into the pipeline:

## 1. Sources:
Crawls / Pre-Existing datasets and the number of words, before filtering:

| *Name*                                                                             | *Subsets*                                                                                                                                              | *Type*  | *Number of words* |
|:-----------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------|:--------|:------------------|
| [FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)                | "lat_Latn"                                                                                                                                             | crawl   | 603,112,685       |
| [GreLa](https://zenodo.org/records/17866143?preview_file=CCS-ZCU%2FGreLa-v0.6.zip) | -                                                                                                                                                      | crawl   | 287,009,237       |
| [Wikimedia](https://huggingface.co/datasets/wikimedia/wikipedia)                   | "20231101.la"                                                                                                                                          | crawl   | 16,758,268        |
| [CLTK](https://huggingface.co/datasets/wikimedia/wikipedia)                        | "lat_text_perseus", <br/>"lat_text_latin_library", <br/> "lat_text_tesserae" <br/> "latin_text_antique_digiliblt"<br/> "latin_text_poeti_ditalia"<br/> | crawl   | 21,409,985        |
| [Canonical Latin](https://github.com/PerseusDL/canonical-latinLit.git)             | "20231101.la"                                                                                                                                          | crawl   | 17,700,787        |
| [Nuntii Latini](https://nuntiilatini.com/page/)                                    | -                                                                                                                                                      | scraped | 73,578            | 
| [Vatican News](https://www.vaticannews.va/bin/servlet/solr/search)                 | -                                                                                                                                                      | scraped | 341,785           |
| **Overall** | | | **946, 406, 325** |

The number of overall words represents results of simply splitting the sentences, so the exact number of results might be lower (e.g., due to HTML garbage etc.)
The data results from running the first two datapipeline steps, which can be executed by:
```
uv run python -m scripts.run_datapipeline --skip-preprocessing --skip-tokenize         # Only execute download and scraping step
```
**Note**: - For the [GreLa](https://zenodo.org/records/17866143?preview_file=CCS-ZCU%2FGreLa-v0.6.zip) dataset it is currently required to download all `.duckdb` parts, combine them into one file and store it to `data/raw/gerla/grela_v0.6.duckdb`

## 2. Pre-Processing: Normalization & Filtering:
This section describes all normalization and filtering steps applied to the raw data.

Current Pre-Processing steps, that are applied to each `*.jsonl` in `data/raw`:
1. Each sample (text) is divided into sentences by replacing `\n` with `.` and replacing shortcuts in namens (e.g., C. Iulius Caesar to C_ Iulius Caesar) in order to split sentences. The result is a list of left over sentences from the paragraph.
2. Each sentence is processed individually. At first, the following Normalization steps are applied:
   - Applying NFD Normalization (canonical decomposition) to split `é` into `e+´`
   - All redundant spaces are removed
   - Mapping different special tokens like `<<>>`, `“..”` to standard `".."`
   - Remove all parts included in typical HTML markers `{...}`, `<...>`, `[...]` because they potentially include HTML leftovers
   - Removing enumerations of sentences
   - Removing leftover sentence markers
   - Ensure that the first letter of each sentence is in upper case for consistency
3. Subsequently, each normalized sentence is filtered by the following quality criteria. If one criterion is violated, the sentences is omitted: 
   - No pipe-symbols like `|` are included.
   - Sentence has `3 < number of words < 50` and must consist of more than 10 characters
   - At least 60% of the sentence should be in lower case, otherwise this indicates wrong normalization or HTML leftovers.
   - The sentence should consist at least to 75% of latin letters or spaces.
4. To further validate each sentence, an instance of the `lingua` language detector is used to classify each sentence. If the sentence is not classified as `Language.LATIN`, it is omitted.
5. Finally, all leftover sentences of one original paragraph are re-concatenated to restore the paragraph.

By applying these Pre-Processing steps, the number of words for each datasource fall to:

| *Name*                                                                             | *Number of words*                |
|:-----------------------------------------------------------------------------------|:---------------------------------|
| [FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)                | 519,151,912                      |
| [GreLa](https://zenodo.org/records/17866143?preview_file=CCS-ZCU%2FGreLa-v0.6.zip) | 233,119,452                      |
| [Wikimedia](https://huggingface.co/datasets/wikimedia/wikipedia)                   | 13,030,722                       |                                                                                                                                        
| [CLTK](https://huggingface.co/datasets/wikimedia/wikipedia)                        | 17,689,706                       |
| [Canonical Latin](https://github.com/PerseusDL/canonical-latinLit.git)             | 421,077                          |                                                                                                                                      
| [Nuntii Latini](https://nuntiilatini.com/page/)                                    | 66,964                           |                                                                                                                                                    
| [Vatican News](https://www.vaticannews.va/bin/servlet/solr/search)                 | 135,519                          |                                                                                                                                                  
| **Overall**                                                                        |  **783,615,352**                 |

# 3. Tokenizer:
On the resulting data corpus, we train a Tokenizer by BPE with `VOCAB_SIZE=32768`. The resulting tokenizer has the following properties:
- Avg. Fertility: 1.31        # How many tokens on avg. to represent one word
- Avg. Compression: 5.36       # How many chars/bytes are represented on avg. by one token
- Avg. Single Chars: 1.04%      #  1.04% of vocabulary represent one single character

Using this tokenizer for pre-tokenization before model training results in the following number of tokens per split:

| *Name / Split* | *Number of tokens*                |
|:---------------|:---------------------------------|
| Train          | 1,068,935,460                      |
| Test           | 20,680,704   |

## 4. Tokenization & Binarization
In the last step, all cleaned data is pre-tokenized & binarized for consecutive training. Therefore, the dataset is tokenized into chunks of size 512.
The resulting dataset is stored in HuggingFace format at `data/tokenized` where three splits are stored:
- train # Binarized in .arrow format
- test # Binarized in .arrow format
- eval # Not tokenized, still in .jsonl format for semantic evaluation
