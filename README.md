# Latin LM
Repository for collecting Latin data and subsequently training a foundation language model for Latin 
language understanding.


## 1. Setup
We highly recommend using **[uv](https://docs.astral.sh/uv/)** as environment and package manager. 

Requirements: You shoudl have **uv** installed locally. Otherwise, try: 

```curl -LsSf https://astral.sh/uv/install.sh | sh```

If you have **uv** installed, clone the repository and synchronize the environment using:
```bash
git clone https://github.com/LukasDankwart/latinlm/tree/main
cd latinlm
uv sync
```
```
# Optional: Activate env in terminal
source .venv/bin/activate # Linux/Mac
.venv\Scripts\Activate # Windows
```
Scripts can be executed by using
`uv run python -m path.to.module`.

**Note**: Workflow 
- New packages can be added by `uv add [PACAKGE]` and are listed to the *[pyproject.toml](pyproject.toml)* automatically.
- If you pull a new state, it is recommended to also use `uv lock && uv sync` again to load new dependencies

## 2. Project Structure
This repository has the following code base structure:
```
latinlm/
├── data                # Is exluded from git! Will be created if the datapipeline is executed
├── scripts             # Python scripts to be exectued (e.g., datapipeline, train scripts...)
├── src                 # Implementation of functions / classes used by scripts
├── experimens          # Stores checkpoints of training approaches
├── configs             # Stores different .yaml configs of models to be trained
```

### 2.1 Data-Pipeline
Currently, the datapipeline includes the following processing steps:
```
├── Data-Pipeline
    └── 1. Download         # Downloads pre-existing Latin datasets from huggingface or other sources
    └── 2. Scraping         # Scrapes internet websites to collect raw latin data
    └── 3. Preprocessing    # Includes normalization and filtering of raw latin data
    └── 4. Tokenize         # Uses a trained tokenizer to translate data into tokenized representation
    └── 5. Binarize         # Compresses the final data into efficient format for training
```
For each of these steps, one corresponding script exists in "scripts/pipeline/" implements the respective pipeline step.
The pipeline can be executed by using:
```
uv run python scripts/run_datapipeline.py

# Optional: Skip specific steps of pipeline by using *--skip-[stepname]* as command
uv run python scripts/run_datapipeline.py --skip-download --skip-scraping
```
The complete datapipeline will create three subdirectories in the 'data' folder:
```
├── data                
    ├── processed       # Stores .jsonl with pre-processed data for each source
    ├── raw             # Stores .jsonl with raw data for each source
    └── tokenized       # Stores the differents splits of the dataset as .arrow formats          
```

### 2.2 Tokenizer
The script for training the tokenizer is designed to take all data available in your local "./data/processed/" directory. 
Currently, the script does train a standard BPE tokenizer and stores the final .json file at "./data/bpe_tokenizer.json".
The script can be run by using:
```
uv run python -m scripts.run_tokenizer     

# You can either skip the training or evaluation process, where some statistics are taken from the trained tokenizer
uv run python -m scripts.run_tokenizer --skip-train  # (does require an trained tokenizer for evaluation)
uv run python -m scripts.run_tokenizer --skip-eval # (does require pre-running datapipeline for data/processed directory)
```

**Note** - Currently the tokenizer training is not included into the datapipeline yet.  

### 2.3 Training
Currently, the training script is designed to train a RoBERTa variant in order to access the current state of the datapipeline.
Still in progress.