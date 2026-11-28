# Animated walk-through of BERTDataset.random_word (Part 8): 15% selection, then 80/10/10 [MASK]/random/unchanged.
import sys, textwrap
from PIL import Image, ImageDraw, ImageFont
out = sys.argv[1]
def font(size, bold=False, mono=False):
    paths = ["/System/Library/Fonts/Menlo.ttc"] if mono else (["/System/Library/Fonts/Supplemental/Arial Bold.ttf"] if bold else ["/System/Library/Fonts/Supplemental/Arial.ttf"])
    for p in paths + ["/System/Library/Fonts/Helvetica.ttc"]:
        try: return ImageFont.truetype(p, size)
        except Exception: pass
    return ImageFont.load_default()
FT, FS, FB, FC, FM = font(30, True), font(21), font(20, True), font(21, mono=True), font(17, mono=True)
toks  = ["the", "cat", "sat", "on", "the", "mat", "because", "it", "was", "warm"]
probs = [0.62, 0.09, 0.41, 0.88, 0.23, 0.14, 0.57, 0.31, 0.76, 0.127]
sel = [i for i, p in enumerate(probs) if p < 0.15]
def branch(i):
    s = probs[i] / 0.15
    return s, ("[MASK]" if s < 0.8 else "piano" if s < 0.9 else toks[i]), ("[MASK] (80%)" if s < 0.8 else "random token (10%)" if s < 0.9 else "unchanged (10%)")
W, H = 1180, 700; cw, x0 = 108, 50
INK, MUTE, GRID = (33, 37, 41), (110, 116, 124), (205, 210, 216)
SEL, MASK, RAND, KEEP, LBL = (254, 243, 199), (219, 234, 254), (252, 231, 243), (220, 252, 231), (237, 233, 254)
CODEBG = (246, 247, 249)
def row(dr, y, label, vals, fills, bold_idx=()):
    dr.text((x0, y - 30), label, font=FB, fill=MUTE)
    for j, v in enumerate(vals):
        bx = x0 + j * cw
        dr.rectangle([bx, y, bx + cw - 6, y + 54], fill=fills[j], outline=GRID, width=2)
        dr.text((bx + (cw - 6) / 2, y + 27), v, font=FB if j in bold_idx else FS, fill=INK, anchor="mm")
def frame(title, code, note, step):
    im = Image.new("RGB", (W, H), "white"); dr = ImageDraw.Draw(im)
    dr.text((40, 26), title, font=FT, fill=INK)
    dr.rounded_rectangle([40, 74, W - 40, 118], radius=8, fill=CODEBG, outline=GRID)
    dr.text((56, 96), code, font=FC, fill=INK, anchor="lm")
    for k, line in enumerate(textwrap.wrap(note, 95)): dr.text((40, 134 + 28 * k), line, font=FS, fill=MUTE)
    y = 240
    fills = [SEL if (step >= 1 and i in sel) else "white" for i in range(len(toks))]
    row(dr, y, "sentence tokens", toks, fills)
    if step >= 1:
        for i, p in enumerate(probs):
            dr.text((x0 + i * cw + (cw - 6) / 2, y + 74), f"p={p:.2f}", font=FM, fill=(180, 83, 9) if i in sel else MUTE, anchor="mm")
    if step >= 2:
        shown = sel[:step - 1] if step <= 4 else sel
        for i in shown:
            s, _, what = branch(i)
            dr.text((x0 + i * cw + (cw - 6) / 2, y + 100), f"p/0.15={s:.2f}", font=FM, fill=INK, anchor="mm")
            dr.text((x0 + i * cw + (cw - 6) / 2, y + 124), what.split(" (")[0], font=FM, fill=INK, anchor="mm")
    if step >= 5:
        vals = [branch(i)[1] if i in sel else toks[i] for i in range(len(toks))]
        fl = [(MASK if branch(i)[1] == "[MASK]" else RAND if branch(i)[1] == "piano" else KEEP) if i in sel else "white" for i in range(len(toks))]
        row(dr, 440, "bert_input (what the model sees)", vals, fl, bold_idx=sel)
    if step >= 6:
        vals = [toks[i] if i in sel else "0" for i in range(len(toks))]
        row(dr, 560, "bert_label (what it must predict; 0 = ignored)", vals, [LBL if i in sel else "white" for i in range(len(toks))], bold_idx=sel)
    return im
fr, du = [], []
fr.append(frame("Masking one sentence", "t1_random, t1_label = self.random_word(t1)", "random_word() walks over the tokens of a sentence and decides, token by token, what BERT will see and what it has to predict.", 0)); du.append(2600)
fr.append(frame("Step 1: pick about 15% of the tokens", "prob = random.random(); if prob < 0.15: ...", "Each token draws a random number. Tokens with p < 0.15 are selected for prediction; here, 3 of 10.", 1)); du.append(2800)
for k, i in enumerate(sel, 1):
    s, new, what = branch(i)
    fr.append(frame("Step 2: decide what to show for each selected token", "prob /= 0.15  # 80% [MASK], 10% random, 10% unchanged",
                    f"“{toks[i]}”: p/0.15 = {s:.2f}, so it becomes {what}.", 1 + k)); du.append(2400)
fr.append(frame("The input BERT sees", "output.append(...)", "“cat” is hidden behind [MASK], “warm” is swapped for a random word, and “mat” is left as it is.", 5)); du.append(2600)
fr.append(frame("The labels it is trained on", "output_label.append(token_id)  # 0 everywhere else", "The loss counts only the 3 selected positions, including the unchanged “mat”, so the model can’t learn that a visible word is never the answer.", 6)); du.append(3600)
fr[0].save(out, save_all=True, append_images=fr[1:], duration=du, loop=0, optimize=True)
fr[-1].save(out.replace(".gif", "_final.png")); fr[3].save(out.replace(".gif", "_f3.png"))
print(out, len(fr), "frames", [(toks[i], round(branch(i)[0], 3), branch(i)[1]) for i in sel])
