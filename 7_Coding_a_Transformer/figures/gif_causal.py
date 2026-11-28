# Animated decoder mask for Part 7: torch.ones -> triu(diagonal=1) -> == 0 -> & padding mask, computed with real torch ops.
import sys, torch
from PIL import Image, ImageDraw, ImageFont
out = sys.argv[1]
def font(size, bold=False, mono=False):
    paths = ["/System/Library/Fonts/Menlo.ttc"] if mono else (["/System/Library/Fonts/Supplemental/Arial Bold.ttf"] if bold else ["/System/Library/Fonts/Supplemental/Arial.ttf"])
    for p in paths + ["/System/Library/Fonts/Helvetica.ttc"]:
        try: return ImageFont.truetype(p, size)
        except Exception: pass
    return ImageFont.load_default()
FT, FS, FB, FC, FV = font(30, True), font(22), font(21, True), font(23, mono=True), font(26, True)
toks = ["[SOS]", "—", "Che", "cosa", "?", "[PAD]", "[PAD]"]
n = len(toks)
ones = torch.ones(1, n, n).type(torch.int)
upper = torch.triu(torch.ones(1, n, n), diagonal=1).type(torch.int)
causal = (upper == 0)
pad = torch.tensor([t != "[PAD]" for t in toks]).unsqueeze(0).unsqueeze(0).int()   # (1, 1, n)
final = pad & causal                                                              # (1, n, n)
W, H = 1040, 900; cell = 78; x0, y0 = 250, 290
INK, MUTE, GRID = (33, 37, 41), (110, 116, 124), (205, 210, 216)
ON, OFF, HI, CODEBG = (209, 250, 229), (243, 244, 246), (37, 99, 235), (246, 247, 249)
GREEN_INK, AMBER = (4, 120, 87), (217, 119, 6)
def frame(title, code, note, M, shade=None, row=None, padcols=False, show=True):
    im = Image.new("RGB", (W, H), "white"); dr = ImageDraw.Draw(im)
    dr.text((40, 28), title, font=FT, fill=INK)
    dr.rounded_rectangle([40, 78, W - 40, 128], radius=8, fill=CODEBG, outline=GRID)
    dr.text((58, 103), code, font=FC, fill=INK, anchor="lm")
    import textwrap
    for k, line in enumerate(textwrap.wrap(note, 82)): dr.text((40, 145 + 28 * k), line, font=FS, fill=MUTE)
    dr.text((x0 + n * cell / 2, y0 - 62), "key positions (what a token may look at)", font=FS, fill=MUTE, anchor="mm")
    for j, t in enumerate(toks):
        dr.text((x0 + j * cell + cell / 2, y0 - 24), t, font=FB, fill=AMBER if (padcols and t == "[PAD]") else INK, anchor="mm")
    for i, t in enumerate(toks):
        dr.text((x0 - 16, y0 + i * cell + cell / 2), t, font=FB, fill=INK, anchor="rm")
    dr.text((x0 - 16, y0 + n * cell + 34), "query positions", font=FS, fill=MUTE, anchor="rm")
    for i in range(n):
        for j in range(n):
            bx, by = x0 + j * cell, y0 + i * cell
            v = int(M[0, i, j]) if show else None
            fill = ON if v == 1 else OFF
            if shade is not None and shade(i, j): fill = (254, 243, 199)
            dr.rectangle([bx, by, bx + cell, by + cell], fill=fill, outline=GRID, width=2)
            if v is not None:
                dr.text((bx + cell / 2, by + cell / 2), "1" if v else "0", font=FV, fill=GREEN_INK if v else MUTE, anchor="mm")
    if row is not None:
        dr.rectangle([x0 - 4, y0 + row * cell - 4, x0 + n * cell + 4, y0 + (row + 1) * cell + 4], outline=HI, width=4)
        seen = [toks[j] for j in range(n) if int(M[0, row, j])]
        dr.text((x0, y0 + n * cell + 34), f"“{toks[row]}” can attend to: " + ", ".join(seen), font=FB, fill=HI, anchor="lm")
    dr.text((W - 40, H - 24), "1 = may attend    0 = blocked", font=FS, fill=MUTE, anchor="rd")
    return im
fr, du = [], []
fr.append(frame("Building the decoder mask", "decoder_input = [SOS] — Che cosa ? [PAD] [PAD]   (seq_len = 7)",
                "Target “— Che cosa?” from the OpusBooks example, padded to 7 tokens.", ones, show=False)); du.append(2600)
fr.append(frame("Step 1: a square of ones", "torch.ones(1, size, size)", "Start with every position allowed to see every other position.", ones)); du.append(2200)
fr.append(frame("Step 2: keep only the part above the diagonal", "torch.triu(torch.ones(1, size, size), diagonal=1)",
                "triu(diagonal=1) zeroes the diagonal and everything below it. The 1s left are the future.", ones,
                shade=lambda i, j: j <= i)); du.append(2200)
fr.append(frame("Step 2: keep only the part above the diagonal", "torch.triu(torch.ones(1, size, size), diagonal=1)",
                "The 1s now mark exactly the future tokens: column j comes after row i.", upper)); du.append(2600)
fr.append(frame("Step 3: flip it", "return mask == 0",
                "== 0 turns 1s into False and 0s into True: each token may see itself and earlier tokens.", causal.int())); du.append(2800)
fr.append(frame("Step 4: block the padding too", "(decoder_input != pad_token).unsqueeze(0).unsqueeze(0).int()",
                "The padding mask is 0 in the [PAD] columns. Broadcast to every row, then AND it with the causal mask.", causal.int(),
                shade=lambda i, j: toks[j] == "[PAD]", padcols=True)); du.append(2800)
fr.append(frame("The decoder mask", "decoder_mask = padding_mask & causal mask",
                "Lower triangle, minus the padding columns. Wherever it is 0, attention scores become −1e9 before softmax.", final)); du.append(2800)
for r in (0, 2, 4):
    fr.append(frame("Reading one row", "decoder_mask[0, i, :]", "Row i decides which tokens position i can use to predict the next word.", final, row=r)); du.append(2000)
du[-1] = 3200
fr[0].save(out, save_all=True, append_images=fr[1:], duration=du, loop=0, optimize=True)
fr[-4].save(out.replace(".gif", "_final.png"))
print(out, len(fr), "frames"); print(final[0].tolist())
