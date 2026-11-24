# Recomputes every probability in data/gif_data.json (the decoding GIF) with GPT-2 and checks it matches the stored, rounded values.
import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained("openai-community/gpt2"); m = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").eval()
data = json.load(open("data/gif_data.json")); worst = 0.0
for s in data:
    ids = tok(s["context"], return_tensors="pt")["input_ids"]
    with torch.no_grad(): logits = m(ids).logits[0, -1]
    for T, ref in s["top"].items():
        p = torch.softmax(logits / float(T), -1); top = torch.topk(p, len(ref["tokens"]))
        assert [tok.decode(i) for i in top.indices] == ref["tokens"], (s["context"], T)
        worst = max(worst, max(abs(a - b) for a, b in zip(top.values.tolist(), ref["probs"])))
    p1 = torch.softmax(logits, -1); top = torch.topk(p1, len(s["renorm"])); rn = (top.values / top.values.sum()).tolist()
    worst = max(worst, max(abs(a - b) for a, b in zip(rn, s["renorm"])))
    print(f"{s['context']!r}: top-6 tokens match; sampled {s['picked']!r}")
print(f"largest difference from the stored (rounded) values: {worst:.5f}")
