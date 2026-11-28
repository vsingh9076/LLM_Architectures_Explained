# RNNs, LSTMs & GRUs (Part 3)

Code for **[LLM Architectures Explained: RNNs, LSTMs & GRUs (Part 3)](https://medium.com/@vipra_singh/llm-architectures-explained-rnn-lstm-grus-part-3-c5e1cbfeda1d)**.

`rnn_lstm_gru.ipynb` holds every code example from the article, in the article's order, with the output of a full run saved in the notebook, so you can read code and results together on GitHub. Cells marked *Added for this repository* are not in the article; they download data or set things up so the notebook runs on a fresh machine. Code excerpts that the article shows while explaining a class appear as text; the complete classes are what run.

## Setup

Python 3.11 or newer (run with Python 3.12.10 on macOS, Apple M4 Pro).

```bash
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook rnn_lstm_gru.ipynb
```

The LSTM example reads `data/google.csv`. The article uses a Google (GOOGL) price file from Kaggle, which needs a Kaggle account; the notebook instead downloads the same daily prices for 2010–2020 from Yahoo Finance with `yfinance` (or uses your Kaggle file if you put it in `data/google.csv`). The Keras examples in section 5 run on Keras 3 with the PyTorch backend, so TensorFlow is not needed. The whole notebook took about 15 minutes on an Apple M4 Pro CPU.

## Results compared with the article

| Example | Result of this run |
|---|---|
| 5.1 Proper Weight Initialization (1) | Runs; its output is in the notebook (the article shows none) |
| 5.1 Proper Weight Initialization (2) | Runs; its output is in the notebook (the article shows none) |
| 5.1 Proper Weight Initialization (3) | Runs; its output is in the notebook (the article shows none) |
| 5.3 Batch Normalization (1) | Runs (defines code, prints nothing) |
| 5.3 Batch Normalization (2) | Runs; its output is in the notebook (the article shows none) |
| 5.4 Gradient Clipping (1) | Runs (defines code, prints nothing) |
| 5.4 Gradient Clipping (2) | Runs (defines code, prints nothing) |
| 6.1 Defining the RNN Class | Runs (defines code, prints nothing) |
| 6.2 Early Stopping Mechanism | Runs (defines code, prints nothing) |
| 6.3 RNN Trainer Class | Runs (defines code, prints nothing) |
| 6.4 Data Loading and Preprocessing (1) | Runs (defines code, prints nothing) |
| 6.4 Data Loading and Preprocessing (2) | Runs (defines code, prints nothing) |
| 6.4 Data Loading and Preprocessing (3) | Runs (defines code, prints nothing) |
| 6.5 Training the RNN | Runs; its output is in the notebook (the article shows none) |
| 11.1 Imports and Initial Setup (1) | Runs (defines code, prints nothing) |
| 11.1 Imports and Initial Setup (2) | Runs (defines code, prints nothing) |
| 11.1 Imports and Initial Setup (3) | Runs (defines code, prints nothing) |
| 11.2 LSTM Class | Runs (defines code, prints nothing) |
| 11.3 Training and Validation | Runs (defines code, prints nothing) |
| 11.4 Data Preprocessing | Runs (defines code, prints nothing) |
| 11.5 Model Training (1) | Runs (defines code, prints nothing) |
| 11.5 Model Training (2) | Runs (defines code, prints nothing) |
| 11.5 Model Training (3) | Runs (defines code, prints nothing) |
| 11.5 Model Training (4) | Runs to early stopping at epoch 86 (validation loss 0.00019); the numbers differ from the article (see below) |
| 11.5 Model Training (5) | Runs; draws the training and validation loss plot |
| 14. Implementation of a Simple GRU | Runs (defines code, prints nothing) |

**About the LSTM numbers.** The article's training log (epoch 1: loss 0.257, early stopping after epoch 141) comes from the original version of this example, before the three backward-pass lines that the article now marks as corrected, and from the Kaggle file. With the corrected code and Yahoo Finance prices, this run starts at a loss of 0.0004 and stops early at epoch 86, with a validation loss of 0.00019.
