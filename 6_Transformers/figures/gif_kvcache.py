# Animated KV cache: each decoding step computes q, k, v for the newest token only,
# appends k and v to the cache, and attends over everything cached so far.
import sys
from PIL import Image, ImageDraw, ImageFont
out = sys.argv[1]
def font(size, bold=False):
    for p in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        try: return ImageFont.truetype(p, size)
        except Exception: pass
    return ImageFont.load_default()
F, FB, FS, FT = font(24), font(24, True), font(21), font(30, True)
INK, MUTE, GRID = (33, 37, 41), (120, 126, 133), (210, 214, 219)
KC, VC, QC, NEW = (37, 99, 235), (22, 163, 74), (217, 119, 6), (254, 243, 199)
toks = ["The", "cat", "sat", "on", "the", "mat"]
PER_TOKEN_MB = 2 * 32 * 4096 * 2 / 2**20      # K and V, 32 layers, 4096 dims, FP16 (Llama 7B) = 0.5 MiB
W, H = 1000, 660; x0, cw = 250, 110
def frame(t, phase):
    im = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(im)
    d.text((40, 28), f"Decoding step {t}: new token “{toks[t-1]}”", font=FT, fill=INK)
    sub = {0: "Compute q, k, v for the new token only.", 1: "Append its k and v to the cache.", 2: "q attends over every cached key; the weights mix the cached values."}[phase]
    d.text((40, 74), sub, font=FS, fill=MUTE)
    d.text((40, 190), "K cache", font=FB, fill=KC); d.text((40, 440), "V cache", font=FB, fill=VC)
    d.text((40, 320), "Query", font=FB, fill=QC)
    n = t if phase >= 1 else t - 1
    for i in range(len(toks)):
        cx = x0 + i * cw
        d.text((cx + cw / 2 - 5, 135), toks[i], font=F, fill=INK if i < t else GRID, anchor="mm")
        for row, col, y in (("k", KC, 160), ("v", VC, 410)):
            box = [cx + 8, y, cx + cw - 8, y + 60]
            if i < n:
                fill = NEW if (i == t - 1 and phase == 1) else "white"
                d.rectangle(box, fill=fill, outline=col, width=3); d.text(((box[0] + box[2]) / 2, y + 30), f"{row}{i+1}", font=FB, fill=col, anchor="mm")
            else:
                d.rectangle(box, outline=GRID, width=2)
    qx = x0 + (t - 1) * cw
    d.rectangle([qx + 8, 290, qx + cw - 8, 350], fill=NEW if phase == 0 else "white", outline=QC, width=3)
    d.text((qx + cw / 2, 320), f"q{t}", font=FB, fill=QC, anchor="mm")
    if phase == 0:
        for j, (lbl, col) in enumerate((("k", KC), ("v", VC))):
            d.text((qx + cw + 10, 296 + j * 28), f"+ {lbl}{t}", font=F, fill=col)
    if phase == 2:
        for i in range(t):
            d.line([qx + cw / 2, 290, x0 + i * cw + cw / 2, 222], fill=QC, width=2)
            d.line([qx + cw / 2, 350, x0 + i * cw + cw / 2, 410], fill=VC, width=1)
    mb = n * PER_TOKEN_MB
    d.text((40, 540), f"Cache size for Llama 7B (FP16): {n} token{'s' if n != 1 else ''} × 0.5 MB = {mb:.1f} MB", font=FB, fill=INK)
    d.text((40, 580), "Without a cache, k and v for every earlier token would be recomputed at each step.", font=FS, fill=MUTE)
    return im
frames, dur = [], []
for t in range(1, len(toks) + 1):
    for ph in (0, 1, 2):
        frames.append(frame(t, ph)); dur.append(1100 if t > 2 else 1600)
end = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(end)
d.text((40, 28), "The cache grows linearly with the sequence", font=FT, fill=INK)
for k, (n, lab) in enumerate(((6, "6 tokens"), (512, "512 tokens"), (4096, "4,096 tokens"))):
    y = 150 + k * 110; w = 40 + 560 * n / 4096
    d.rectangle([250, y, 250 + w, y + 60], fill=(219, 234, 254), outline=KC, width=3)
    d.text((40, y + 30), lab, font=FB, fill=INK, anchor="lm")
    v = n * PER_TOKEN_MB; d.text((260 + w, y + 30), f"{v:.0f} MB" if v < 1024 else f"{v/1024:.0f} GB", font=FB, fill=KC, anchor="lm")
d.text((40, 500), "Per token: K and V × 32 layers × 4,096 values × 2 bytes = 0.5 MB (Llama 7B, batch of 1).", font=FS, fill=MUTE)
frames.append(end); dur.append(3500)
frames[0].save(out, save_all=True, append_images=frames[1:], duration=dur, loop=0, optimize=True)
print(out, len(frames), "frames; per-token MB", PER_TOKEN_MB, "| 4096 tokens GB", 4096 * PER_TOKEN_MB / 1024)
