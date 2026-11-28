# Animated causal (masked) self-attention: scores -> mask future -> softmax row by row.
import sys, math, torch
from PIL import Image, ImageDraw, ImageFont
out = sys.argv[1]
def font(size, bold=False):
    for p in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        try: return ImageFont.truetype(p, size)
        except Exception: pass
    return ImageFont.load_default()
F, FB, FS, FT = font(26), font(26, True), font(22), font(30, True)
toks = ["I", "love", "attention", "maps"]
torch.manual_seed(0)
n, d = 4, 8
X = torch.randn(n, d); W = [torch.randn(d, d) / math.sqrt(d) for _ in range(3)]
Q, K = X @ W[0], X @ W[1]
S = (Q @ K.T / math.sqrt(d))
mask = torch.triu(torch.ones(n, n, dtype=torch.bool), 1)
Wt = torch.softmax(S.masked_fill(mask, float("-inf")), -1)
Wd, Ht = 1000, 720; cell = 120; x0, y0 = 300, 190
INK, MUTE, GRID = (33, 37, 41), (120, 126, 133), (210, 214, 219)
BLUE, GREY = (37, 99, 235), (236, 238, 241)
def frame(title, sub, stage, rows_done=0):
    im = Image.new("RGB", (Wd, Ht), "white"); dr = ImageDraw.Draw(im)
    dr.text((40, 30), title, font=FT, fill=INK); dr.text((40, 75), sub, font=FS, fill=MUTE)
    dr.text((x0 + 2 * cell, y0 - 70), "keys (tokens attended to)", font=FS, fill=MUTE, anchor="mm")
    for j, t in enumerate(toks): dr.text((x0 + j * cell + cell / 2, y0 - 25), t, font=FB, fill=INK, anchor="mm")
    for i, t in enumerate(toks): dr.text((x0 - 20, y0 + i * cell + cell / 2), t, font=FB, fill=INK, anchor="rm")
    dr.text((60, y0 + 2 * cell + 90), "queries", font=FS, fill=MUTE)
    for i in range(n):
        for j in range(n):
            bx, by = x0 + j * cell, y0 + i * cell; box = [bx, by, bx + cell, by + cell]
            masked = stage >= 1 and mask[i, j]
            if stage >= 2 and i < rows_done:
                w = float(Wt[i, j]); c = tuple(int(255 - (255 - b) * w) for b in BLUE)
                dr.rectangle(box, fill=GREY if masked else c, outline=GRID, width=2)
                txt = "0" if masked else f"{w:.2f}"; col = "white" if (not masked and w > 0.55) else (MUTE if masked else INK)
                dr.text((bx + cell / 2, by + cell / 2), txt, font=F, fill=col, anchor="mm")
            else:
                dr.rectangle(box, fill=GREY if masked else "white", outline=GRID, width=2)
                txt = "−∞" if masked else f"{float(S[i, j]):.2f}"
                dr.text((bx + cell / 2, by + cell / 2), txt, font=F, fill=MUTE if masked else INK, anchor="mm")
            if stage >= 2 and i == rows_done - 1:
                dr.rectangle([x0 - 4, y0 + i * cell - 4, x0 + n * cell + 4, y0 + (i + 1) * cell + 4], outline=BLUE, width=4)
    if stage >= 2 and rows_done:
        dr.text((x0 + n * cell + 30, y0 + (rows_done - 1) * cell + cell / 2), "row sums to 1", font=FS, fill=BLUE, anchor="lm")
    return im
fr, du = [], []
fr.append(frame("Step 1: scaled scores  QKᵀ / √dₖ", "Every token scores every token, including the ones after it.", 0)); du.append(2600)
fr.append(frame("Step 2: mask the future", "Scores above the diagonal are set to −∞, so later tokens can’t be seen.", 1)); du.append(2600)
for r in range(1, n + 1):
    fr.append(frame("Step 3: softmax, one row at a time", f"Row “{toks[r-1]}”: e^(−∞) = 0, so future tokens get zero weight.", 2, r)); du.append(1900)
fr.append(frame("Masked attention weights", "Each token mixes only itself and the tokens before it.", 2, n)); du[-1] = 1900; du.append(3200)
fr[0].save(out, save_all=True, append_images=fr[1:], duration=du, loop=0, optimize=True)
print(out, len(fr), "frames", [round(float(x), 2) for x in Wt.flatten()])
