# Animated next-token decoding with real GPT-2 probabilities (Part 9): softmax, temperature, top-k, sample, append.
import sys, json, textwrap
from PIL import Image, ImageDraw, ImageFont
data = json.load(open(sys.argv[1])); out = sys.argv[2]
def font(size, bold=False, mono=False):
    paths = ["/System/Library/Fonts/Menlo.ttc"] if mono else (["/System/Library/Fonts/Supplemental/Arial Bold.ttf"] if bold else ["/System/Library/Fonts/Supplemental/Arial.ttf"])
    for p in paths + ["/System/Library/Fonts/Helvetica.ttc"]:
        try: return ImageFont.truetype(p, size)
        except Exception: pass
    return ImageFont.load_default()
FT, FS, FB, FM, FV = font(30, True), font(21), font(22, True), font(22, mono=True), font(19)
W, H = 1120, 690
INK, MUTE, GRID = (33, 37, 41), (110, 116, 124), (205, 210, 216)
BAR, PICK, CTX, NEW = (147, 197, 253), (37, 99, 235), (246, 247, 249), (220, 252, 231)
def frame(title, note, context, new_tok, bars, pick=None, label=""):
    im = Image.new("RGB", (W, H), "white"); dr = ImageDraw.Draw(im)
    dr.text((40, 26), title, font=FT, fill=INK)
    for k, line in enumerate(textwrap.wrap(note, 92)): dr.text((40, 72 + 28 * k), line, font=FS, fill=MUTE)
    dr.text((40, 150), "context", font=FB, fill=MUTE)
    dr.rounded_rectangle([40, 180, W - 40, 236], radius=10, fill=CTX, outline=GRID)
    x = 60; y = 208
    dr.text((x, y), context, font=FM, fill=INK, anchor="lm"); x += dr.textlength(context, font=FM)
    if new_tok:
        w = dr.textlength(new_tok, font=FM)
        dr.rounded_rectangle([x - 2, 188, x + w + 4, 228], radius=6, fill=NEW); dr.text((x, y), new_tok, font=FM, fill=(4, 120, 87), anchor="lm")
    if bars:
        dr.text((40, 270), label, font=FB, fill=MUTE)
        toks, probs = bars; top = max(probs); x0, y0, bw, bh = 260, 310, 700, 40
        for i, (t, p) in enumerate(zip(toks, probs)):
            yy = y0 + i * (bh + 12)
            dr.text((x0 - 16, yy + bh / 2), repr(t)[1:-1].replace(" ", "␣") if t.strip() == "" else t.replace(" ", "␣"), font=FM, fill=INK, anchor="rm")
            dr.rectangle([x0, yy, x0 + bw, yy + bh], fill=(250, 250, 251), outline=GRID)
            dr.rectangle([x0, yy, x0 + int(bw * p / max(top, 1e-9) * 0.95), yy + bh], fill=PICK if pick == i else BAR)
            dr.text((x0 + bw + 14, yy + bh / 2), f"{p:.3f}", font=FV, fill=INK, anchor="lm")
    dr.text((W - 40, H - 18), "␣ = space    probabilities from GPT-2 (124M)", font=font(17, mono=True), fill=MUTE, anchor="rd")
    return im
fr, du = [], []
s0 = data[0]; ctx = s0["context"]
T = lambda s, t: (s["top"][t]["tokens"], s["top"][t]["probs"])
fr.append(frame("Step 1: the model scores every token", "GPT-2 reads the whole context and outputs one logit per vocabulary entry (50,257). Softmax turns them into probabilities; here are the six largest.", ctx, "", T(s0, "1.0"), label="softmax(logits), temperature 1.0")); du.append(3200)
fr.append(frame("Temperature 0.5: sharper", "Dividing the logits by a temperature below 1 makes the distribution peakier, so the top tokens win more often.", ctx, "", T(s0, "0.5"), label="softmax(logits / 0.5)")); du.append(2600)
fr.append(frame("Temperature 1.5: flatter", "A temperature above 1 flattens it, giving rarer tokens a better chance: more varied, but more mistakes.", ctx, "", T(s0, "1.5"), label="softmax(logits / 1.5)")); du.append(2600)
fr.append(frame("Step 2: keep the top k", "Top-k sampling (k = 6 here) throws away every other token and renormalizes the six that are left so they sum to 1.", ctx, "", (s0["top"]["1.0"]["tokens"], s0["renorm"]), label="top-6, renormalized")); du.append(2800)
fr.append(frame("Step 3: sample one token and append it", f"A random draw picks “{s0['picked'].strip()}”. It is added to the context, and the whole process repeats for the next position.", ctx, s0["picked"], (s0["top"]["1.0"]["tokens"], s0["renorm"]), pick=s0["pick"], label=f"sampled “{s0['picked'].strip()}” from these probabilities, then appended")); du.append(2800)
for s in data[1:]:
    fr.append(frame("Repeat: one forward pass per new token", "Each new token needs another pass through the model; the KV cache (Part 6) avoids recomputing the earlier positions.", s["context"], "", (s["top"]["1.0"]["tokens"], s["renorm"]), label="next-token probabilities after this context (top-6, renormalized)")); du.append(1700)
    fr.append(frame("Repeat: one forward pass per new token", "Each new token needs another pass through the model; the KV cache (Part 6) avoids recomputing the earlier positions.", s["context"], s["picked"], (s["top"]["1.0"]["tokens"], s["renorm"]), pick=s["pick"], label=f"sampled “{s['picked'].strip()}” from these probabilities, then appended")); du.append(1700)
last = data[-1]
fr.append(frame("Generated text", "Greedy decoding would always take the top bar; sampling with temperature, top-k or top-p trades some accuracy for variety.", last["context"] + last["picked"], "", None)); du.append(3400)
fr[0].save(out, save_all=True, append_images=fr[1:], duration=du, loop=0, optimize=True)
fr[4].save(out.replace(".gif", "_f4.png")); fr[1].save(out.replace(".gif", "_f1.png"))
print(out, len(fr), "frames")
