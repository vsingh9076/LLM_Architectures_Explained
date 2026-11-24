# Training curve of the mini GPT from its printed log (train and validation loss every 500 steps).
import sys, re, math
from PIL import Image, ImageDraw, ImageFont
log, out = open(sys.argv[1]).read(), sys.argv[2]
pts = [(int(a), float(b), float(c)) for a, b, c in re.findall(r"step\s+(\d+): train loss ([\d.]+), val loss ([\d.]+)", log)]
def font(size, bold=False):
    try: return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf", size)
    except Exception: return ImageFont.load_default()
FT, FS, FSm = font(28, True), font(20), font(18)
INK, MUTE, GRID, BLUE, AMBER = (33, 37, 41), (110, 116, 124), (225, 228, 232), (37, 99, 235), (194, 65, 12)
W, H = 1040, 580; L, R, T, B = 110, 220, 115, 490
im = Image.new("RGB", (W, H), "white"); dr = ImageDraw.Draw(im)
dr.text((40, 24), "Mini GPT on Tiny Shakespeare: the loss falls as it learns", font=FT, fill=INK)
ymin, ymax = 1.0, 4.5
X = lambda s: L + (W - L - R) * s / 3000
Y = lambda v: B - (B - T) * (v - ymin) / (ymax - ymin)
for v in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]:
    dr.line([L, Y(v), W - R, Y(v)], fill=GRID, width=1); dr.text((L - 12, Y(v)), f"{v:.1f}", font=FSm, fill=MUTE, anchor="rm")
for s in range(0, 3001, 500): dr.text((X(s), B + 14), f"{s:,}", font=FSm, fill=MUTE, anchor="mt")
dr.text(((L + W - R) / 2, B + 44), "training step", font=FS, fill=MUTE, anchor="mt")
dr.text((40, T - 34), "loss (lower is better)", font=FS, fill=MUTE)
y0 = math.log(65); dr.line([L, Y(y0), W - R, Y(y0)], fill=MUTE, width=1)
dr.text((L + 8, Y(y0) - 6), "random guessing: ln(65) ≈ 4.17", font=FSm, fill=MUTE, anchor="lb")
for k, col, lab in [(1, BLUE, "train"), (2, AMBER, "validation")]:
    xy = [(X(p[0]), Y(p[k])) for p in pts]
    dr.line(xy, fill=col, width=4)
    for x, y in xy: dr.ellipse([x - 5, y - 5, x + 5, y + 5], fill=col)
    dr.text((W - R + 14, xy[-1][1] + (16 if k == 1 else -16)), f"{lab} {pts[-1][k]:.3f}", font=FS, fill=col, anchor="lm")
im.save(out); print(out, pts)
