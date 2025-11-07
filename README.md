# Telugu BPE Tokenizer

Custom Byte-Pair Encoding (BPE) tokenizer for Telugu: collects a Wikipedia corpus, trains a >5K token vocabulary, and demonstrates its compression performance.

## Project Layout

- `data/telugu_corpus.txt` — 400 Telugu Wikipedia article extracts.
- `scripts/download_telugu_corpus.py` — downloads or refreshes the corpus.
- `scripts/build_telugu_bpe.py` — trains BPE, saves merges/vocab, reports compression.
- `notebooks/telugu_bpe_training.ipynb` — step-by-step training and evaluation.

## Quick Start

1. Create a virtual environment:
   
2. (Optional) Refresh the corpus:

python -m venv .venv
.\.venv\Scripts\activate
pip install requests
2. (Optional) Refresh the corpus:
python scripts/download_telugu_corpus.py
3. Train and evaluate BPE:
python scripts/build_telugu_bpe.py
   Outputs include vocabulary size (>5000) and compression ratio (≥3).## Notebook Usage- Install notebook dependencies:
pip install notebook ipykernel
python -m ipykernel install --user --name telugu-bpe
- Open `notebooks/telugu_bpe_training.ipynb`, select the `telugu-bpe` kernel, and run all cells.## Repository Setup
git init
git add .
git commit -m "Initial commit: Telugu BPE tokenizer"
git remote add origin https://github.com/jssunil/ERAV4_s11.git
git branch -M main
git push -u origin main
## Notes- Corpus is filtered to focus on Telugu characters.- Rerunning `build_telugu_bpe.py` will overwrite `telugu_bpe_merges.txt` and `telugu_bpe_vocab.txt`.- Compression ratio ≥3 confirms the tokenizer meets the assignment requirement.