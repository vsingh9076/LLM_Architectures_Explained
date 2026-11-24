# Code for "LLM Architectures Explained: GPT (Part 9)"

Everything needed to run the examples in the article and to reproduce its numbers and figures.

## Setup

Python 3.11 or newer (tested with Python 3.12.10 on macOS, Apple M4 Pro).

```bash
python3 -m venv gpt-env
source gpt-env/bin/activate            # Windows Command Prompt: gpt-env\Scripts\activate.bat
                                       # Windows PowerShell:     .\gpt-env\Scripts\Activate.ps1
pip install -r requirements.txt        # torch==2.15.0, transformers==5.24.0
```

For an NVIDIA GPU, install PyTorch with the command for your CUDA version from https://pytorch.org/get-started/locally/ first.

The first run downloads Tiny Shakespeare (about 1 MB) and GPT-2's weights (about 550 MB, cached in `~/.cache/huggingface`).

## The article's examples

| File | Article section | What it does | Time (M4 Pro) |
|---|---|---|---|
| `mini_gpt.py` | 7 | Builds a 0.82M-parameter character-level GPT, trains it for exactly 3,000 steps on Tiny Shakespeare and generates 300 characters | 1 min 9 s on the GPU (`mps`), 4 min 13 s on the CPU |
| `gpt2_demo.py` | 8 | Loads GPT-2 (124M), shows its tokens and next-token probabilities, and compares greedy decoding with temperature + top-p sampling (`top_k=0`, because `generate()` otherwise adds a top-k of 50) | a few seconds after download |

`outputs/` holds the logs from the runs shown in the article. Training is random and PyTorch's kernels differ between hardware and versions: expect the same broad behaviour (validation loss from about 4.2 to about 1.6–1.7), but not identical numbers, text or timings. Times include evaluation, not the download.

## Checks and the numbers behind the figures

| File | What it does |
|---|---|
| `audit.py` | Checks the mini GPT's model code: parameter count (818,048), tied input/output weights, no information leaking from later tokens, the `F.scaled_dot_product_attention` version from DIY exercise 2 matching the explicit mask (`torch.allclose`), and the edge cases of `generate()`. See `outputs/audit.log`. |
| `vis_data.py` | Runs GPT-2 (eager attention, eval mode, no gradients) on "The cat sat on the mat" and writes `vis_data.json`: token IDs, embedding values, next-token probabilities at full precision, and the attention weights used in the article's GIFs. Layers and heads are counted from 0 in the file and from 1 in the article (layer index 11, head 8 = "layer 12, head 9"). |
| `decoding_check.py` | Recomputes every probability in `data/gif_data.json` (the decoding GIF) with GPT-2 and checks it against the stored, rounded values. |

`data/` holds the JSON files the figures were drawn from.

## Figures

The scripts in `figures/` draw the article's GIFs and diagrams with Pillow (`pip install pillow`); they use macOS system fonts and fall back to Pillow's default font elsewhere.

```bash
python figures/gif_gpt_basics.py data/vis_data.json out/          # tokens, attention, block and training GIFs
python figures/gif_sampling.py data/gif_data.json out/decoding.gif
python figures/fig_gpt_timeline.py out/gpt_timeline.png
python figures/fig_gpt.py out/                                    # architecture diagram and comparison table
python figures/fig_loss_curve.py outputs/mini_gpt_gpu.log out/mini_gpt_loss.png
```

## Credits

The mini GPT's structure follows Andrej Karpathy's nanoGPT (https://github.com/karpathy/nanoGPT). GPT-2 weights: https://huggingface.co/openai-community/gpt2. Tiny Shakespeare: https://github.com/karpathy/char-rnn.
