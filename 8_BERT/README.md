# BERT (Part 8)

Code for **[LLM Architectures Explained: BERT (Part 8)](https://medium.com/@vipra_singh/llm-architectures-explained-bert-part-8-c60c1d9ebc82)**.

`bert_from_scratch.ipynb` holds every code example from the article, in the article's order, with the output of a full run saved in the notebook, so you can read code and results together on GitHub. Cells marked *Added for this repository* are not in the article; they download data or set things up so the notebook runs on a fresh machine.

## Setup

Python 3.11 or newer (run with Python 3.12.10 on macOS, Apple M4 Pro).

```bash
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook bert_from_scratch.ipynb
```

The notebook downloads the Cornell Movie-Dialogs Corpus (about 10 MB) into `datasets/` and trains its own WordPiece tokenizer. The article's Colab shell commands (`!wget`, `!unzip`) are shown as text, and a plain-Python cell does the same download so it also works on macOS and Windows.

## Results compared with the article

| Example | Result of this run |
|---|---|
| 5.2 Training Masked LM (pretrained BERT) | Same output as the article |
| 8.1 Data Downloading (1) | Runs (defines code, prints nothing) |
| 8.1 Data Downloading (2) | Same sentence pair (only the quote marks differ) |
| 8.2 WordPiece Tokenization | Runs (defines code, prints nothing) |
| 8.3 Pre-Training Strategy (Pre-processing) (1) | Runs (defines code, prints nothing) |
| 8.3 Pre-Training Strategy (Pre-processing) (2) | A different example: the 15% masking and the sentence pairs are random, so every run differs |
| 8.4 Embedding | Runs (defines code, prints nothing) |
| 8.5.1 Multi Head Attention | Runs (defines code, prints nothing) |
| 8.6 Final BERT Model | Runs (defines code, prints nothing) |
| 8.7 Optimizer | Runs (defines code, prints nothing) |
| 8.8 Trainer | Runs (defines code, prints nothing) |

**Training.** The article's *test run* trains a 2-layer, 768-wide BERT for 20 epochs over all ~221,000 sentence pairs on the CPU, which takes days, so that cell is shown but not run. A quick run of the same model and trainer on 20 batches (640 pairs) shows the pre-training loop working: 46.7 million parameters, masked-LM and next-sentence losses, and accuracy. After so little training the loss is still about 11 and the next-sentence accuracy is at chance (48%).

## Figures

The animations made for the article can be redrawn with Pillow (`pip install pillow`):

- `figures/gif_mlm.py` draws the masked-language-model animation (section 8.3): `python figures/gif_mlm.py out/mlm_masking.gif`
