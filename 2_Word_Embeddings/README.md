# Word Embeddings (Part 2)

Code for **[LLM Architectures Explained: Word Embeddings (Part 2)](https://medium.com/@vipra_singh/llm-architectures-explained-word-embeddings-part-2-ff6b9cf1d82d)**.

`word_embeddings.ipynb` holds every code example from the article, in the article's order, with the output of a full run saved in the notebook, so you can read code and results together on GitHub. The ELMo example needs an older environment and is in [`elmo/`](elmo), with its own notebook and requirements.

## Setup

Python 3.11 or newer (run with Python 3.12.10 on macOS, Apple M4 Pro).

```bash
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook word_embeddings.ipynb
```

The BERT example downloads `bert-base-uncased` (about 440 MB) on first use.

## Results compared with the article

| Example | Result of this run |
|---|---|
| 3.1.1 Count Vectorizer | Same output as the article |
| 3.1.3 Term Frequency-Inverse Document Frequency (TF-IDF) | Same output as the article |
| 3.1.4 N-Grams | Same output as the article |
| 3.1.5 Co-occurrence Matrices | Same values; current pandas prints every column instead of abbreviating the normalized matrix with “...” |
| 3.1.6 One-Hot Encoding | Same output as the article |
| 3.3.2 BERT | Same output as the article |

## Figures

The animations made for the article can be redrawn with Pillow (`pip install pillow`):

- `figures/gif_cooc.py` draws the co-occurrence matrix animation (section 3.1.5): `python figures/gif_cooc.py out/cooccurrence_matrix.gif`
