# Checks on the mini GPT from mini_gpt.py (model classes only, no training): causal masking, SDPA equivalence, weight tying, edge cases.
import torch, torch.nn.functional as F, re, types
src = open("mini_gpt.py").read()
model_src = src.split("# ---------- 7.4")[0].split("# ---------- 7.3")[1].split("\n", 1)[1]
ns = {}; exec("import math, torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n" + model_src, ns)
torch.manual_seed(0)
m = ns["GPT"](65, 128).eval()
print("parameters:", sum(p.numel() for p in m.parameters()), "| tied:", m.lm_head.weight is m.tok_emb.weight)
a = torch.randint(65, (1, 40)); b = a.clone(); b[0, 30:] = torch.randint(65, (10,))
la, _ = m(a); lb, _ = m(b); print("causal leak (max diff on shared prefix):", (la[0, :30] - lb[0, :30]).abs().max().item())
att = m.blocks[0].attn; x = torch.randn(2, 50, 128)
def sdpa_forward(self, x):
    B, T, C = x.shape
    q, k, v = self.qkv(x).split(C, dim=2)
    q, k, v = [t.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) for t in (q, k, v)]
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.attn_drop.p if self.training else 0.0)
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    return self.resid_drop(self.proj(y))
y1 = att(x); y2 = sdpa_forward(att, x); print("SDPA vs explicit mask, allclose:", torch.allclose(y1, y2, atol=1e-5), "max diff", (y1 - y2).abs().max().item())
try: m.generate(torch.zeros(1, 1, dtype=torch.long), 2, temperature=0); print("temperature 0: no error")
except ValueError as e: print("temperature 0 ->", e)
print("top_k=1000 ok:", m.generate(torch.zeros(1, 1, dtype=torch.long), 3, top_k=1000).shape)
print("long generation shape:", m.generate(torch.zeros(1, 6, dtype=torch.long), 140, top_k=40).shape)
