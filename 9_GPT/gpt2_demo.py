import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
cfg = model.config
print(f"layers={cfg.n_layer}, d_model={cfg.n_embd}, heads={cfg.n_head}, "
      f"context={cfg.n_positions}, vocab={cfg.vocab_size}, "
      f"parameters={sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")

prompt = "The Transformer architecture changed natural language processing because"
inputs = tokenizer(prompt, return_tensors="pt")
print(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))   # GPT-2's byte-level BPE tokens

# The model's view of the next token: the five most likely continuations
with torch.no_grad():
    logits = model(**inputs).logits[0, -1]
probs = torch.softmax(logits, dim=-1)
top = torch.topk(probs, 5)
for p, i in zip(top.values, top.indices):
    print(f"{tokenizer.decode(i)!r:>12}  {p:.3f}")

# Greedy decoding vs. sampling with temperature and top-p
# (top_k=0 switches off the top-k of 50 that generate() otherwise adds, so only temperature and top-p are used)
torch.manual_seed(0)
for name, kwargs in [("greedy", dict(do_sample=False)),
                     ("sampled", dict(do_sample=True, temperature=0.8, top_p=0.95, top_k=0))]:
    out = model.generate(**inputs, max_new_tokens=30, pad_token_id=tokenizer.eos_token_id, **kwargs)
    print(f"\n[{name}] {tokenizer.decode(out[0], skip_special_tokens=True)}")
