# ---------- 7.2 Data: Tiny Shakespeare, one token per character ----------
import math, urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text = urllib.request.urlopen(url).read().decode("utf-8")

chars = sorted(set(text))                       # the vocabulary: every distinct character
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}    # character -> token ID
itos = {i: ch for ch, i in stoi.items()}        # token ID -> character
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]       # 90% train, 10% validation

block_size = 128     # context length: how many tokens the model can look back at
batch_size = 32
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))      # random start positions
    x = torch.stack([d[i:i + block_size] for i in ix])          # inputs
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])  # targets: the same text shifted by one
    return x.to(device), y.to(device)


# ---------- 7.3 Model: a decoder-only Transformer ----------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, block_size, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)      # queries, keys and values in one matrix multiply
        self.proj = nn.Linear(d_model, d_model)         # output projection
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        # lower-triangular causal mask: position t may only look at positions 0..t
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_heads, T, head_dim)
        q, k, v = [t.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) for t in (q, k, v)]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))       # scaled dot-product scores
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))  # hide the future
        att = self.attn_drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)      # concatenate the heads
        return self.resid_drop(self.proj(y))


class Block(nn.Module):
    """One GPT decoder block: pre-norm attention and MLP, each with a residual connection."""
    def __init__(self, d_model, n_heads, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),   # expand 4x, GELU as in GPT-1/2/3
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # GPT-2 style: LayerNorm before each sub-layer
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, d_model=128, n_layers=4, n_heads=4, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)      # token embeddings
        self.pos_emb = nn.Embedding(block_size, d_model)      # learned position embeddings
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, block_size, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)                     # final LayerNorm (added in GPT-2)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight             # weight tying: input and output embeddings are shared
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # small normal weights (std 0.02, as in GPT-2) and zero biases; GPT-2's extra 1/sqrt(N) scaling of residual layers is left out
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))                    # (B, T, vocab_size): a score for every possible next token
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        # no KV cache here: every step re-runs the model on the (cropped) context, which is simple but slow
        if temperature <= 0:
            raise ValueError("temperature must be > 0 (for greedy decoding, take the argmax instead)")
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]               # crop to the context length
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature            # only the last position predicts the next token
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")    # keep only the k most likely tokens
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # sample one token
            idx = torch.cat([idx, next_id], dim=1)             # append it and repeat
        return idx


# ---------- 7.4 Training: next-token prediction ----------
model = GPT(vocab_size, block_size).to(device)
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)

@torch.no_grad()
def estimate_loss(iters=50):
    model.eval()
    out = {}
    for split in ("train", "val"):
        out[split] = torch.stack([model(*get_batch(split))[1] for _ in range(iters)]).mean().item()
    model.train()
    return out

max_steps = 3000
for step in range(max_steps + 1):
    if step % 500 == 0:                                  # check progress on train and validation text
        losses = estimate_loss()
        print(f"step {step:4d}: train loss {losses['train']:.3f}, val loss {losses['val']:.3f}")
    if step == max_steps:
        break                                            # exactly 3,000 updates
    x, y = get_batch("train")
    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# ---------- 7.5 Generating text ----------
model.eval()
prompt = torch.tensor([encode("ROMEO:")], dtype=torch.long, device=device)
print(decode(model.generate(prompt, max_new_tokens=300, temperature=0.8, top_k=40)[0].tolist()))
