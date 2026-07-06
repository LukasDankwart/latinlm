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
At this state, a training script is implemented to train a RoBERTa base variant on the currently collected data.
This is mainly done in order to access the quality of the implemented data pipeline and subsequently, the quality of the data corpus.

The training of RoBERTa can be done only if the previously mentioned datapipeline has been executed and the tokenizer has been trained.
```
uv run python -m scripts.run_training --train-roberta 
```

Note that the 'roberta.yaml' do hold all parameters for the pre-training stage of the model. Currently, the config holds
parameters, in order to report the training procedure to "wandb". 
Therefore, authentication using a wandb API key is required upon initial execution. This can be done by:
```
wandb login [API Key]
```
Alternatively, this function can be disabled by setting the `report_to` flag to `None` in the configuration.

### 2.4 Evaluation
Currently, some metrics have been implemented in order to evaluate the results of the first RoBERTa approach.
Metrics can be found in `src.tests.roberta_metrics`. They can also be executed by the following script call:
```
uv run python -m script.run_evaluation --eval-roberta --set-checkpoint=[Path to your checkpoint folder]
```
Checkpoints should be stored at the `./experiments` directory. After running the evaluation script, the results will be
stored at `./experiments/evaluation` as .csv files, where the checkpoint name is included in the file name as suffix.

#### POS Tagging
In addition, inspired by the [LatinBERT paper](https://arxiv.org/abs/2009.10053), we also want to investigate the RoBERTa
model in context of POS tagging on three UD Latin datasets (Perseus, PROIEL, ITTB). 
For each dataset, two approaches can be trained:
- Full Fine-Tuning RoBERTa + new classfication head
- only train new classification head on RoBERTa embeddings 

The training can be performed by the following call:
```
uv run python -m script.run_training --train-roberta-pos --set-checkpoint=... --set-dataset=["Perseus", "PROIEL" or "ITTB"] # For Full Fine Tuning
uv run python -m script.run_training --train-roberta-pos --set-checkpoint=... --set-dataset=["Perseus", "PROIEL" or "ITTB"] --freeze-base # For Class.-Head only
```
Note: - For each approach (FT vs. Class-Head only), one specific .yaml config has to be configured:
- `configs/roberta_pos_full_ft.yaml` for FT approach, `configs/roberta_pos_freezed.yaml` for Class-Head only.

Currently, the evaluation of the POS trained models is included inside the training procedure. The results on the currently
best model variant are as follows:

| *Perseus* | *PROIEL*  | *ITTB*    |
|:----------|:----------|:----------|
| **95.01** | **94.83** | **98.11** |

*Eval accuracies of each dataset by Full-FT approach*

### Bash Scripts
For training and evaluation scripts that are desired to be executed on MIDAS, bash scripts have been written and 
stored inside the root folder. Calling this bash scripts requires specifying which GPU node should be used for the procedure:
```
# The GPU number has to be in [0, 1, 2, 4]
chmod +x run_roberta_training.sh    # If required, only once per script
./run_roberta_training.sh
```
The script does perform a check if the specified GPU is in usage. If so - the script is aborted. Ensure the corresponding
GPU is not in use! You can also skip this feature by copying the Python commands for the scripts at the end of the .sh 
files and executing them in isolation.
