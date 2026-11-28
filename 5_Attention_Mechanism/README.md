# Attention Mechanism (Part 5)

Code for **[LLM Architectures Explained: Attention Mechanism (Part 5)](https://medium.com/@vipra_singh/llm-architectures-explained-attention-part-5-495bbe3d278e)**.

`attention_mechanism.ipynb` holds every code example from the article, in the article's order, with the output of a full run saved in the notebook, so you can read code and results together on GitHub. Cells marked *Added for this repository* are not in the article; they download data or set things up so the notebook runs on a fresh machine.

## Setup

Python 3.11 or newer (run with Python 3.12.10 on macOS, Apple M4 Pro).

```bash
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook attention_mechanism.ipynb
```

Runs in a few seconds on a CPU.

## Results compared with the article

| Example | Result of this run |
|---|---|
| 4.5.1 Masked Multi-Head Attention | Same output as the article |

## Figures

The animations made for the article can be redrawn with Pillow (`pip install pillow`):

- `figures/gif_mask.py` draws the masked attention animation (section 4.5.1): `python figures/gif_mask.py out/masked_attention.gif`
