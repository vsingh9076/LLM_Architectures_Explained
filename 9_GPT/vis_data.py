# Real GPT-2 (124M) numbers for the Part 9 visuals: tokens and IDs, causal attention weights, next-token predictions.
import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", attn_implementation="eager").eval()
text = "The cat sat on the mat"
ids = tok(text, return_tensors="pt")["input_ids"]
toks = tok.convert_ids_to_tokens(ids[0])
with torch.no_grad():
    out = model(ids, output_attentions=True)
att = torch.stack(out.attentions)[:, 0]          # (layers, heads, T, T)
probs = torch.softmax(out.logits[0], -1)          # (T, vocab)
T = ids.shape[1]
nxt = []
for t in range(T):
    top = torch.topk(probs[t], 3)
    true_p = probs[t, ids[0, t + 1]].item() if t + 1 < T else None
    nxt.append(dict(top=[(tok.decode(i), round(p.item(), 3)) for p, i in zip(top.values, top.indices)], true=tok.decode(ids[0, t + 1]) if t + 1 < T else None, p_true=true_p))   # full precision; round only for display
# pick, for each layer and head, how much attention goes to the first token (the "attention sink")
sink = att[:, :, 1:, 0].mean(-1)                  # (layers, heads)
res = dict(text=text, tokens=toks, ids=ids[0].tolist(), next=nxt,
           layer_mean=[att[l].mean(0).tolist() for l in range(att.shape[0])],
           sink_by_layer=[round(x, 3) for x in sink.mean(1).tolist()],
           emb_tok=model.transformer.wte.weight[ids[0]][:, :8].tolist(), emb_pos=model.transformer.wpe.weight[:T, :8].tolist())
# least sink-dominated single heads, to show a readable pattern
cands = sorted(((sink[l, h].item(), l, h) for l in range(att.shape[0]) for h in range(att.shape[1])))[:6]
res["low_sink_heads"] = [(l, h, round(s, 3)) for s, l, h in cands]
res["heads"] = {f"{l}.{h}": att[l, h].tolist() for _, l, h in cands}
json.dump(res, open("vis_data.json", "w"), indent=1)
print(toks, ids[0].tolist())
for t, n in zip(toks, nxt): print(repr(t), n)
print("sink by layer", res["sink_by_layer"]); print("low-sink heads", res["low_sink_heads"])
