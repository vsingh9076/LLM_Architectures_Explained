# Coding a Transformer (Part 7)

Code for **[LLM Architectures Explained: Coding a Transformer (Part 7)](https://medium.com/@vipra_singh/llm-architectures-explained-coding-a-transformer-part-7-ca459ceceb61)**.

`transformer_from_scratch.ipynb` holds every code example from the article, in the article's order, with the output of a full run saved in the notebook, so you can read code and results together on GitHub. Cells marked *Added for this repository* are not in the article; they download data or set things up so the notebook runs on a fresh machine. One line of the article's code is changed so it runs with current libraries; the change is marked in the notebook.

## Setup

Python 3.11 or newer (run with Python 3.12.10 on macOS, Apple M4 Pro).

```bash
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook transformer_from_scratch.ipynb
```

The notebook downloads the OPUS Books English–Italian dataset (32,332 sentence pairs) and trains two word-level tokenizers. The quick run below saves a 904 MB checkpoint in `weights/`.

## Results compared with the article

| Example | Result of this run |
|---|---|
| 3. Transformer from Scratch | Runs (defines code, prints nothing) |
| 3.1 Input Embeddings | Runs (defines code, prints nothing) |
| 3.2 Positional Encoding | Runs (defines code, prints nothing) |
| 3.3 Layer Normalization | Runs (defines code, prints nothing) |
| 3.4 Feed-Forward Network | Runs (defines code, prints nothing) |
| 3.5 Multi-Head Attention | Runs (defines code, prints nothing) |
| 3.6 Residual Connection | Runs (defines code, prints nothing) |
| 3.7 Encoder (1) | Runs (defines code, prints nothing) |
| 3.7 Encoder (2) | Runs (defines code, prints nothing) |
| 3.8 Decoder (1) | Runs (defines code, prints nothing) |
| 3.8 Decoder (2) | Runs (defines code, prints nothing) |
| 3.8 Decoder (3) | Runs (defines code, prints nothing) |
| 3.9 Building the Transformer (1) | Runs (defines code, prints nothing) |
| 3.9 Building the Transformer (2) | Runs (defines code, prints nothing) |
| 3.9.1 Tokenizer | Runs (defines code, prints nothing) |
| 3.9.2 Loading Dataset (1) | Runs (defines code, prints nothing) |
| 3.9.2 Loading Dataset (2) | Runs after one change: the dataset name `'opus_books'` becomes `'Helsinki-NLP/opus_books'` (see below) |
| 3.9.2 Loading Dataset (3) | Runs (defines code, prints nothing) |
| 3.9.2 Loading Dataset (4) | Runs (defines code, prints nothing) |
| 3.9.3 Validation Loop (1) | Runs (defines code, prints nothing) |
| 3.9.3 Validation Loop (2) | Runs (defines code, prints nothing) |
| 3.9.4 Training Loop (1) | Runs (defines code, prints nothing) |
| 3.9.4 Training Loop (2) | Runs (defines code, prints nothing) |
| 3.9.4 Training Loop (3) | Runs (defines code, prints nothing) |

**The dataset name.** Current versions of the Hugging Face `datasets` library (tested with 5.2.0) reject `load_dataset('opus_books', ...)` with *Invalid HF URI* and need the full name `'Helsinki-NLP/opus_books'`. This is the only change to the article's code.

**Training.** The article's full run (20 epochs, about 15 minutes per epoch on Eduardo Muñoz Torres's GPU) would take many hours on a laptop CPU, so the notebook shows that cell and the article's output from Torres's run, and then does a quick run of the same `train_model()`: one epoch stopped after 300 batches. It took about 77 minutes on an Apple M4 Pro CPU while other programs were also using it. After so little training the model only repeats frequent tokens (*— — — non non non …*); the full run is what produces the article's translations.

## Figures

The animations made for the article can be redrawn with Pillow (`pip install pillow`):

- `figures/gif_causal.py` draws the decoder mask animation (section 3.9.2): `python figures/gif_causal.py out/decoder_mask.gif`
