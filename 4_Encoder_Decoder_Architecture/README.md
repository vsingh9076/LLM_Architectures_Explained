# Encoder-Decoder Architecture (Part 4)

Code for **[LLM Architectures Explained: Encoder-Decoder Architecture (Part 4)](https://medium.com/@vipra_singh/llm-architectures-explained-encoder-decoder-architecture-part-4-b96ace71394c)**.

`seq2seq_translation.ipynb` holds every code example from the article, in the article's order, with the output of a full run saved in the notebook, so you can read code and results together on GitHub.

## Setup

Python 3.11 or newer (run with Python 3.12.10 on macOS, Apple M4 Pro).

```bash
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
jupyter notebook seq2seq_translation.ipynb
```

The notebook downloads the Multi30k German–English dataset from Hugging Face (`bentrevett/multi30k`) and the two spaCy models listed above. The article's `pip install` line is shown in the notebook as text; `requirements.txt` installs the same packages.

## Results compared with the article

| Example | Result of this run |
|---|---|
| Importing the libraries | Runs (defines code, prints nothing) |
| Tokenization | Runs (defines code, prints nothing) |
| Fields and vocabulary (1) | Runs (defines code, prints nothing) |
| Fields and vocabulary (2) | Runs (defines code, prints nothing) |
| Loading and building the vocabulary | Runs; its output is in the notebook (the article shows none) |
| Batching | Runs (defines code, prints nothing) |
| Encoder | Runs (defines code, prints nothing) |
| Decoder | Runs (defines code, prints nothing) |
| Seq2Seq model | Runs (defines code, prints nothing) |
| Training loop | Runs (defines code, prints nothing) |
| Evaluation | Runs (defines code, prints nothing) |
| Model, optimizer and loss | Runs (defines code, prints nothing) |
| Training | Full 10-epoch run: final losses 2.986 (train) and 3.945 (validation), close to the article's 2.990 and 3.862 |
| Translating and BLEU (1) | Runs (defines code, prints nothing) |
| Translating and BLEU (2) | Same example sentence; the model writes *a man with a hat hat is looking at something .* and the test BLEU is 15.72 (article: 16.55) |

**Training time and small differences.** The 10 epochs ran on the CPU (the code uses a GPU only if CUDA is available). They took about 2 hours here because other programs were using the CPU at the same time; the article's run took about 59 minutes. The losses, translation and BLEU differ slightly from the article's because the weights start from random values and the batches are shuffled.
