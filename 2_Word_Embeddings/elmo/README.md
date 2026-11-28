# ELMo example (Part 2, section 3.3.3)

The article's ELMo code uses [AllenNLP](https://github.com/allenai/allennlp), which was archived in 2022. It no longer installs with current Python and PyTorch, so it lives in this separate folder with its own, older environment.

`elmo.ipynb` runs the article's code unchanged and saves its output.

## Setup

Python 3.9 is needed (tested with Python 3.9.6 on macOS).

```bash
python3.9 -m venv elmo-env
source elmo-env/bin/activate
pip install --no-deps -r requirements.txt    # --no-deps: the exact versions that work together are all listed
pip install jupyter
jupyter notebook elmo.ipynb
```

The notebook downloads ELMo's published files (the small 54 MB model) and saves them as `elmo_options.json` and `elmo_weights.hdf5`, the names the article uses.

## Results

- Output shape: (2 sentences, 5 words, 256 numbers per word).
- `Elmo` uses dropout 0.5 by default and starts in training mode, so the article's code gives different vectors on every run (in two runs, the similarity below came out 0.35 and 0.42). After `elmo.eval()`, every run gives the same vectors: "green" in *green apple* and *green thumb* has a cosine similarity of 0.953 (0.951 if `elmo.eval()` is set before the first call, because ELMo's LSTM carries its state from one call to the next): the same word gets a different vector in each context, which a static embedding such as Word2Vec cannot do.
- On macOS, `torchvision` prints a warning that it could not load its image library; it is not used here and can be ignored.
