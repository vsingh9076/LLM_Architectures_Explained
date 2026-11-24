# Static figures for Part 9: (1) GPT decoder-only architecture, (2) GPT family comparison table.
import sys
from PIL import Image, ImageDraw, ImageFont
outdir = sys.argv[1]
def font(size, bold=False, mono=False):
    paths = ["/System/Library/Fonts/Menlo.ttc"] if mono else (["/System/Library/Fonts/Supplemental/Arial Bold.ttf"] if bold else ["/System/Library/Fonts/Supplemental/Arial.ttf"])
    for p in paths + ["/System/Library/Fonts/Helvetica.ttc"]:
        try: return ImageFont.truetype(p, size)
        except Exception: pass
    return ImageFont.load_default()
INK, MUTE, GRID = (33, 37, 41), (100, 106, 114), (190, 196, 204)
C = {"emb": (254, 226, 196), "ln": (221, 214, 254), "att": (252, 205, 220), "mlp": (254, 243, 199), "head": (209, 250, 229), "add": (255, 255, 255)}

# ---------- (1) architecture ----------
W, H = 1000, 1240
im = Image.new("RGB", (W, H), "white"); dr = ImageDraw.Draw(im)
F, FB, FS, FM = font(24), font(24, True), font(20), font(20, mono=True)
cx = 420; bw = 380
def box(y, h, text, fill, sub=None, w=None):
    bw2 = (w or bw)
    dr.rounded_rectangle([cx - bw2 / 2, y, cx + bw2 / 2, y + h], radius=12, fill=fill, outline=INK, width=2)
    dr.text((cx, y + h / 2 - (11 if sub else 0)), text, font=FB, fill=INK, anchor="mm")
    if sub: dr.text((cx, y + h / 2 + 16), sub, font=FS, fill=MUTE, anchor="mm")
def arrow(y1, y2, x=None):
    x = cx if x is None else x
    dr.line([x, y1, x, y2], fill=INK, width=3); dr.polygon([(x - 8, y2 + 2), (x + 8, y2 + 2), (x, y2 - 12)], fill=INK)
def plus(y):
    dr.ellipse([cx - 16, y - 16, cx + 16, y + 16], fill="white", outline=INK, width=3)
    dr.line([cx - 9, y, cx + 9, y], fill=INK, width=3); dr.line([cx, y - 9, cx, y + 9], fill=INK, width=3)
def skip(y_from, y_to):
    x = cx + bw / 2 + 40
    dr.line([cx, y_from, x, y_from, x, y_to, cx + 16, y_to], fill=INK, width=3)
    dr.polygon([(cx + 16, y_to), (cx + 30, y_to - 7), (cx + 30, y_to + 7)], fill=INK)
# bottom: inputs
dr.text((cx, 1205), "input tokens:  The  cat  sat  on  the", font=FM, fill=INK, anchor="mm")
arrow(1185, 1163)
box(1080, 70, "token embedding + position embedding", C["emb"], "learned position vectors (GPT-1, 2, 3)", w=540)
arrow(1080, 1053)
# block frame
top, bot = 330, 1040
dr.rounded_rectangle([cx - bw / 2 - 40, top, cx + bw / 2 + 80, bot], radius=18, outline=(120, 126, 134), width=3)
dr.text((cx - bw / 2 - 60, (top + bot) / 2), "× N", font=font(34, True), fill=INK, anchor="rm")
dr.text((cx + bw / 2 + 100, top + 20), "decoder block", font=FS, fill=MUTE)
dr.text((cx + bw / 2 + 100, top + 48), "(no cross-attention,", font=FS, fill=MUTE)
dr.text((cx + bw / 2 + 100, top + 74), " no encoder)", font=FS, fill=MUTE)
# lower sub-layer
y = 1040
plus(700); skip(1010, 700)
arrow(1010, 960); box(900, 60, "LayerNorm", C["ln"])
arrow(900, 860); box(770, 90, "masked multi-head", C["att"], "self-attention (causal)")
arrow(770, 729)
# upper sub-layer
plus(380); skip(684, 380)
arrow(684, 640); box(580, 60, "LayerNorm", C["ln"])
arrow(580, 540); box(450, 90, "feed-forward (MLP)", C["mlp"], "d → 4d → d, GELU")
arrow(450, 409)
# top
arrow(364, 300); box(240, 60, "final LayerNorm", C["ln"])
arrow(240, 200); box(110, 90, "linear → vocabulary logits", C["head"], "weights often tied to the embeddings")
arrow(110, 70); dr.text((cx, 40), "softmax → next-token probabilities", font=FB, fill=INK, anchor="mm")
dr.text((W - 20, 20), "pre-norm layout of GPT-2 and GPT-3", font=FS, fill=MUTE, anchor="ra")
im.save(f"{outdir}/gpt_architecture.png")

# ---------- (2) family table ----------
rows = [("", "GPT-1 (2018)", "GPT-2 XL (2019)", "GPT-3 (2020)", "gpt-oss-120b (2025)"),
        ("Parameters", "117M", "1.5B", "175B", "117B total, 5.1B active"),
        ("Layers", "12", "48", "96", "36"),
        ("Model width (d_model)", "768", "1,600", "12,288", "2,880"),
        ("Attention heads", "12", "25", "96", "64 query / 8 key-value"),
        ("Context length", "512", "1,024", "2,048", "131,072"),
        ("Vocabulary (BPE)", "40,478", "50,257", "50,257", "201,088"),
        ("Normalization", "post-norm LayerNorm", "pre-norm LayerNorm", "pre-norm LayerNorm", "pre-norm RMSNorm"),
        ("Positions", "learned", "learned", "learned", "RoPE (YaRN)"),
        ("Attention", "dense", "dense", "dense + banded sparse", "dense + 128-token window"),
        ("Feed-forward", "GELU MLP", "GELU MLP", "GELU MLP", "MoE: 128 experts, top-4, SwiGLU"),
        ("Training data", "BooksCorpus", "WebText (40 GB)", "~300B tokens", "not disclosed (trillions of tokens)")]
cw = [300, 230, 230, 250, 360]; rh = 54; W2 = sum(cw) + 40; H2 = rh * len(rows) + 70
im = Image.new("RGB", (W2, H2), "white"); dr = ImageDraw.Draw(im)
Fh, Fc, Fr = font(21, True), font(19), font(19, True)
y = 20
for r, row in enumerate(rows):
    x = 20
    for c, val in enumerate(row):
        fill = (241, 245, 249) if r == 0 else ((249, 250, 251) if r % 2 == 0 else "white")
        dr.rectangle([x, y, x + cw[c], y + rh], fill=fill, outline=GRID)
        f = Fh if r == 0 else (Fr if c == 0 else Fc)
        dr.text((x + 12, y + rh / 2), val, font=f, fill=INK, anchor="lm")
        x += cw[c]
    y += rh
dr.text((W2 - 20, H2 - 14), "Sources: GPT-1, GPT-2 and GPT-3 papers; gpt-oss model card and config", font=font(17), fill=MUTE, anchor="rd")
im.save(f"{outdir}/gpt_family_table.png")
print("ok")
