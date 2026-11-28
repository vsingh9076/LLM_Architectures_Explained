# NLP Fundamentals (Part 1)

Code for **[LLM Architectures Explained: NLP Fundamentals (Part 1)](https://medium.com/@vipra_singh/llm-architectures-explained-nlp-fundamentals-part-1-de5bf75e553a)**.

`nlp_fundamentals.ipynb` holds every code example from the article, in the article's order, with the output of a full run saved in the notebook, so you can read code and results together on GitHub. Cells marked *Added for this repository* are not in the article; they download data or set things up so the notebook runs on a fresh machine.

## Setup

Python 3.11 or newer (run with Python 3.12.10 on macOS, Apple M4 Pro).

```bash
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
jupyter notebook nlp_fundamentals.ipynb
```

The notebook's first cell downloads the NLTK data the examples need. The BERT examples download `bert-base-uncased` (about 440 MB) on first use.

## Results compared with the article

| Example | Result of this run |
|---|---|
| 4.1.1 Tokenization | Same output as the article |
| 4.1.3 Lemmatisation (1) | Same output as the article |
| 4.1.3 Lemmatisation (2) | Same output as the article |
| 4.1.4 Normalization | Same output as the article |
| 4.1.5 Part of Speech (POS) Tagging | Same output as the article |
| 4.2.1 Bag-of-Words (BoW) | Same output as the article |
| 4.2.2 Term Frequency-Inverse Document Frequency (TF-IDF) | Same output as the article |
| 4.2.3 N-grams | Same output as the article |
| 4.2.4 Word Embeddings | Same vector; the similarity scores differ only from the 9th digit (floating-point) |
| 4.2.5 Contextual Word Embeddings | Same shape and tokens; embeddings agree to 5–6 decimals (floating-point differences between PyTorch versions) |
| 4.3.1 Named Entity Recognition (NER) | Same output as the article |
| 4.3.2 Language Model | A different sentence (*i enjoy learning new*): the bigram model picks words at random, so every run differs |
| 4.3.4 Deep Learning NLP Techniques | Same output as the article |

## Figures

The animations made for the article can be redrawn with Pillow (`pip install pillow`):

- `figures/gif_part1.py` draws preprocessing_pipeline.gif and bpe_merges.gif (sections 4.1 and 4.1.1): `python figures/gif_part1.py out`
