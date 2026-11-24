# Beginner visuals for Part 9 (GPT), drawn from real GPT-2 (124M) numbers in part9-gpt/code/vis_data.json.
# usage: python3 gif_gpt_basics.py <vis_data.json> <out_dir>
import sys, json, math, textwrap
from PIL import Image, ImageDraw, ImageFont

D = json.load(open(sys.argv[1])); OUT = sys.argv[2]
def font(size, bold=False, mono=False):
    paths = ["/System/Library/Fonts/Menlo.ttc"] if mono else (["/System/Library/Fonts/Supplemental/Arial Bold.ttf"] if bold else ["/System/Library/Fonts/Supplemental/Arial.ttf"])
    for p in paths + ["/System/Library/Fonts/Helvetica.ttc"]:
        try: return ImageFont.truetype(p, size)
        except Exception: pass
    return ImageFont.load_default()
FT, FS, FB, FM, FSm, FBig = font(30, True), font(22), font(22, True), font(21, mono=True), font(18), font(28, True)
INK, MUTE, GRID, BG2 = (33, 37, 41), (110, 116, 124), (205, 210, 216), (246, 247, 249)
BLUE, BLUE_L, GREEN, GREEN_L, AMBER, AMBER_L, RED, RED_L, GREY_L = (37, 99, 235), (219, 234, 254), (4, 120, 87), (209, 250, 229), (180, 83, 9), (254, 243, 199), (185, 28, 28), (254, 226, 226), (233, 235, 238)
W = 1040
toks = [t.replace("Ġ", " ") for t in D["tokens"]]
show = ["The", "cat", "sat", "on", "the", "mat"]          # display form (the leading space is explained in the text)

def canvas(h, title, note=None):
    im = Image.new("RGB", (W, h), "white"); dr = ImageDraw.Draw(im)
    dr.text((40, 26), title, font=FT, fill=INK)
    y = 74
    if note:
        for k, line in enumerate(textwrap.wrap(note, 86)): dr.text((40, y + 28 * k), line, font=FS, fill=MUTE)
    return im, dr
def box(dr, xy, fill, outline=None, r=10, w=2): dr.rounded_rectangle(xy, radius=r, fill=fill, outline=outline, width=w)
def save(frames, durs, name):
    frames[0].save(f"{OUT}/{name}.gif", save_all=True, append_images=frames[1:], duration=durs, loop=0, optimize=True)
    frames[-1].save(f"{OUT}/{name}_last.png"); print(name, len(frames), "frames")
def shade(v, lo=(255, 255, 255), hi=BLUE):
    return tuple(int(lo[i] + (hi[i] - lo[i]) * v) for i in range(3))

# ---------------------------------------------------------------- 1. text -> tokens -> IDs -> vectors
def tokens_gif():
    fr, du = [], []
    H = 640; cw, gap, x0 = 140, 18, 52
    xs = [x0 + i * (cw + gap) for i in range(6)]
    tv, pv = D["emb_tok"], D["emb_pos"]
    def vec(dr, x, y, vals, col, label=None):
        m = max(abs(v) for row in tv + pv for v in row)
        for j, v in enumerate(vals):
            c = shade(min(1, abs(v) / m), hi=BLUE if v >= 0 else RED)
            dr.rectangle([x + j * 16, y, x + j * 16 + 14, y + 28], fill=c, outline=GRID)
        if label: dr.text((x + 64, y + 40), label, font=FSm, fill=MUTE, anchor="mm")
    steps = [
        ("1. Start with text", "GPT never sees letters directly. First, the text is cut into pieces called tokens."),
        ("2. Split the text into tokens", "GPT-2 uses byte-pair encoding: common words are one token, rare words are split into pieces. The Ġ is how GPT-2 marks the space before a word."),
        ("3. Look up each token's ID", "Every token has a fixed number in a vocabulary of 50,257 tokens. These are GPT-2's real IDs."),
        ("4. Turn each ID into a vector", "The ID selects one row of the token embedding table: 768 learned numbers that describe the token. (Only the first 8 are drawn.)"),
        ("5. Add the position", "A second table gives each position (0, 1, 2, …) its own vector. Adding the two tells the model what the token is and where it is."),
    ]
    for s, (title, note) in enumerate(steps):
        im, dr = canvas(H, title, note)
        y = 180
        if s == 0:
            dr.text((W / 2, y + 60), "The cat sat on the mat", font=font(44, True), fill=INK, anchor="mm")
        else:
            for i in range(6):
                box(dr, [xs[i], y, xs[i] + cw, y + 60], BLUE_L, BLUE)
                dr.text((xs[i] + cw / 2, y + 30), ("Ġ" if i else "") + show[i], font=FBig, fill=INK, anchor="mm")
            dr.text((40, y + 74), "token", font=FSm, fill=MUTE)
        if s >= 2:
            for i in range(6):
                dr.line([xs[i] + cw / 2, y + 62, xs[i] + cw / 2, y + 96], fill=GRID, width=2)
                box(dr, [xs[i] + 20, y + 98, xs[i] + cw - 20, y + 140], BG2, GRID)
                dr.text((xs[i] + cw / 2, y + 119), str(D["ids"][i]), font=FM, fill=INK, anchor="mm")
            dr.text((40, y + 150), "token ID", font=FSm, fill=MUTE)
        if s >= 3:
            for i in range(6):
                vec(dr, xs[i] + 6, y + 190, tv[i], BLUE)
            dr.text((40, y + 230), "token vector", font=FSm, fill=MUTE)
        if s >= 4:
            for i in range(6):
                dr.text((xs[i] + 70, y + 262), "+", font=FBig, fill=MUTE, anchor="mm")
                vec(dr, xs[i] + 6, y + 280, pv[i], GREEN)
                dr.text((xs[i] + 70, y + 324), f"position {i}", font=FSm, fill=MUTE, anchor="mm")
                dr.text((xs[i] + 70, y + 352), "=", font=FBig, fill=MUTE, anchor="mm")
                tot = [a + b for a, b in zip(tv[i], pv[i])]
                vec(dr, xs[i] + 6, y + 370, tot, AMBER)
            dr.text((40, y + 410), "input to the first decoder block", font=FSm, fill=MUTE)
            dr.text((W - 40, y + 410), "blue = positive number, red = negative", font=FSm, fill=MUTE, anchor="ra")
        fr.append(im); du.append(2600 if s < 4 else 5000)
    save(fr, du, "gpt_tokens")

# ---------------------------------------------------------------- 2. causal (masked) self-attention, row by row
def mask_gif():
    fr, du = [], []
    H = 800; n = 6; cell = 82; x0, y0 = 330, 230
    def grid(dr, M, upto, title_head, rowfocus=None, show_vals=True):
        dr.text((x0 + n * cell / 2, y0 - 52), "keys: the tokens it may look at", font=FS, fill=MUTE, anchor="mm")
        for j in range(n): dr.text((x0 + j * cell + cell / 2, y0 - 18), show[j], font=FB, fill=INK, anchor="mm")
        dr.text((x0 - 200, y0 + n * cell + 26), "queries: the token", font=FS, fill=MUTE)
        dr.text((x0 - 200, y0 + n * cell + 52), "that is looking", font=FS, fill=MUTE)
        for i in range(n):
            dr.text((x0 - 16, y0 + i * cell + cell / 2), show[i], font=FB, fill=INK if (rowfocus is None or rowfocus == i) else MUTE, anchor="rm")
            for j in range(n):
                x, y = x0 + j * cell, y0 + i * cell
                if j > i:
                    dr.rectangle([x, y, x + cell, y + cell], fill=GREY_L, outline="white", width=2)
                    if i <= upto: dr.text((x + cell / 2, y + cell / 2), "0", font=FS, fill=(160, 164, 170), anchor="mm")
                elif i <= upto:
                    v = M[i][j]
                    dr.rectangle([x, y, x + cell, y + cell], fill=shade(v), outline="white", width=2)
                    if show_vals: dr.text((x + cell / 2, y + cell / 2), f"{v:.2f}", font=FS, fill="white" if v > 0.55 else INK, anchor="mm")
                else:
                    dr.rectangle([x, y, x + cell, y + cell], fill=(250, 250, 251), outline=(236, 238, 240), width=2)
        if rowfocus is not None:
            y = y0 + rowfocus * cell
            dr.rectangle([x0 - 2, y - 2, x0 + n * cell + 2, y + cell + 2], outline=AMBER, width=4)
        dr.text((x0 + n * cell, y0 + n * cell + 30), title_head, font=FB, fill=INK, anchor="ra")
        dr.text((x0 + n * cell, y0 + n * cell + 58), "values rounded to 2 decimals; grey = masked (exactly 0)", font=FSm, fill=MUTE, anchor="ra")
    head = D["heads"]["11.8"]; prev = [[1.0 if (j == i - 1 or (i == 0 and j == 0)) else 0.0 for j in range(n)] for i in range(n)]
    real_prev = D_prev = None
    note0 = "Each token builds a query and compares it with the keys of the tokens it is allowed to see. Grey cells are the future: the mask sets their scores to −∞, so after softmax their weight is exactly 0."
    im, dr = canvas(H, "Masked (causal) self-attention, one row at a time", note0); grid(dr, head, -1, "GPT-2, layer 12, head 9"); fr.append(im); du.append(3000)
    for i in range(n):
        seen = ", ".join(f"“{show[j]}”" for j in range(i + 1))
        top = max(range(i + 1), key=lambda j: head[i][j])
        note = f"“{show[i]}” may look at {seen}. Its weights add up to 1; the largest goes to " + ("itself" if top == i else f"“{show[top]}”") + f" ({head[i][top]:.2f})."
        im, dr = canvas(H, "Masked (causal) self-attention, one row at a time", note); grid(dr, head, i, "GPT-2, layer 12, head 9", rowfocus=i)
        fr.append(im); du.append(2600)
    im, dr = canvas(H, "The result: a lower-triangular attention matrix", "Nothing above the diagonal: no token ever sees a later one. On this sentence, this head puts most of the weight of “on” and “mat” on “sat”.")
    grid(dr, head, n, "GPT-2, layer 12, head 9"); fr.append(im); du.append(5000)
    realprev = D.get("prev_head")
    im, dr = canvas(H, "Different heads show different patterns", "Another real GPT-2 head (layer 5, head 12) puts almost all its weight on the previous token in this sentence. Different heads weigh the tokens differently.")
    grid(dr, D["heads"]["4.11"], n, "GPT-2, layer 5, head 12"); fr.append(im); du.append(5000)
    save(fr, du, "gpt_causal_attention")

# ---------------------------------------------------------------- 3. one decoder block, step by step
def block_gif():
    fr, du = [], []
    H = 900; n = 6; cw = 104; x0 = 300
    cx = [x0 + i * (cw + 14) + cw / 2 for i in range(n)]
    rows = {"in": 770, "ln1": 690, "att": 590, "add1": 500, "ln2": 430, "mlp": 350, "add2": 270}
    steps = [
        ("in", "Input", "The block receives one vector for each token: the embeddings for block 1, or the previous block's output."),
        ("ln1", "LayerNorm", "Each token's vector is normalized (shifted to average 0 and scaled to a standard spread, then a learned scale and shift) so the numbers stay well behaved through many blocks."),
        ("att", "Masked self-attention", "Each token gathers information from itself and the tokens before it (never from later ones). This is the only step where tokens talk to each other."),
        ("add1", "Add (residual)", "The attention output is added to the block's input, which reaches the + directly along the green skip path. The sub-layer only adds an adjustment, which makes deep stacks easy to train."),
        ("ln2", "LayerNorm", "Rescale again before the second sub-layer."),
        ("mlp", "Feed-forward network", "Each token now thinks on its own: a small two-layer network (768 → 3,072 → 768 numbers in GPT-2) processes every token separately, with the same weights for all of them."),
        ("add2", "Add again", "The MLP output is added to the stream again, and the result goes to the next block. GPT-2 small stacks 12 of these blocks; GPT-3 stacks 96."),
    ]
    order = [s[0] for s in steps]
    labels = {"ln1": "LayerNorm", "att": "Masked self-attention", "add1": "+", "ln2": "LayerNorm", "mlp": "Feed-forward (MLP)", "add2": "+"}
    for k, (key, title, note) in enumerate(steps):
        im, dr = canvas(H, f"One GPT block, step {k + 1} of {len(steps)}: {title}", note)
        done = order[:k + 1]
        # residual stream lines
        for i in range(n): dr.line([cx[i], rows["add2"], cx[i], rows["in"]], fill=GRID, width=3)
        for key2 in order[1:]:
            y = rows[key2]; act = key2 == key; past = key2 in done
            col_f = AMBER_L if act else (BG2 if past else "white"); col_o = AMBER if act else (GRID if past else (230, 232, 235))
            if key2 in ("add1", "add2"):
                for i in range(n):
                    dr.ellipse([cx[i] - 16, y - 16, cx[i] + 16, y + 16], fill=col_f, outline=col_o, width=2)
                    dr.text((cx[i], y), "+", font=FB, fill=INK if past else GRID, anchor="mm")
                dr.text((x0 - 24, y), "add", font=FS, fill=INK if past else GRID, anchor="rm")
            elif key2 == "att":
                box(dr, [x0 - 10, y - 42, cx[-1] + cw / 2 + 10, y + 42], col_f, col_o)
                dr.text(((x0 + cx[-1] + cw / 2) / 2, y - 22), labels[key2], font=FB, fill=INK if past else GRID, anchor="mm")
            else:
                for i in range(n):
                    box(dr, [cx[i] - cw / 2 + 4, y - 22, cx[i] + cw / 2 - 4, y + 22], col_f, col_o, r=8)
                    if key2 == "mlp": dr.text((cx[i], y), "MLP", font=FSm, fill=INK if past else GRID, anchor="mm")
                    else: dr.text((cx[i], y), "LN", font=FSm, fill=INK if past else GRID, anchor="mm")
                dr.text((x0 - 24, y), labels[key2], font=FS, fill=INK if past else GRID, anchor="rm")
        # residual (skip) paths: each sub-layer's input also goes straight to its "+"
        if k >= 1:
            for i in range(n):
                gx = cx[i] + cw / 2 + 4
                for ya, yb, kk in [(rows["in"] - 12, rows["add1"], "add1"), (rows["add1"] - 18, rows["add2"], "add2")]:
                    if kk == "add2" and "ln2" not in done: continue
                    dr.line([cx[i] + 2, ya, gx, ya], fill=GREEN, width=2); dr.line([gx, ya, gx, yb], fill=GREEN, width=2); dr.line([gx, yb, cx[i] + 17, yb], fill=GREEN, width=2)
                    dr.polygon([(cx[i] + 17, yb), (cx[i] + 25, yb - 5), (cx[i] + 25, yb + 5)], fill=GREEN)
            dr.text((cx[-1] + cw / 2 + 12, rows["add1"] + 34), "skip", font=FSm, fill=GREEN, anchor="lm")
        # attention arrows: only to earlier tokens
        if key == "att":
            y = rows["att"]
            i = n - 1
            for j in range(i):
                h = 6 + 5 * (i - j)
                dr.arc([cx[j], y + 2 - h, cx[i], y + 2 + h], 0, 180, fill=BLUE, width=3)
            dr.polygon([(cx[i], y + 2), (cx[i] - 8, y + 16), (cx[i] + 8, y + 16)], fill=BLUE)
            dr.text(((x0 + cx[-1]) / 2, y + 60), "e.g. “mat” gathers information from all earlier tokens, never from later ones", font=FSm, fill=BLUE, anchor="mm")
        # token labels and markers
        for i in range(n):
            box(dr, [cx[i] - cw / 2 + 4, rows["in"] + 40, cx[i] + cw / 2 - 4, rows["in"] + 84], BLUE_L, BLUE, r=8)
            dr.text((cx[i], rows["in"] + 62), show[i], font=FB, fill=INK, anchor="mm")
            ytop = rows[key] if key != "in" else rows["in"]
            if key != "att": dr.ellipse([cx[i] - 9, ytop - 9, cx[i] + 9, ytop + 9], fill=AMBER)
        dr.text((x0 - 24, rows["in"] + 62), "tokens", font=FS, fill=MUTE, anchor="rm")
        if key == "add2":
            dr.text(((x0 + cx[-1] + cw / 2) / 2, rows["add2"] - 40), "output → next block (GPT-2 small has 12 blocks in total)", font=FB, fill=GREEN, anchor="mm")
        fr.append(im); du.append(3400 if key != "add2" else 5000)
    save(fr, du, "gpt_block")

# ---------------------------------------------------------------- 4. next-token training on one sentence
def training_gif():
    fr, du = [], []
    H = 700; n = 5; cw, gap, x0 = 150, 22, 120
    xs = [x0 + i * (cw + gap) for i in range(n)]
    nx = D["next"][:n]
    losses = [-math.log(e["p_true"]) for e in nx]          # natural log, full-precision probabilities
    def frame(k, title, note, final=False):
        im, dr = canvas(H, title, note)
        y1, y2, y3 = 210, 330, 450
        dr.text((40, y1 + 30), "input", font=FB, fill=INK); dr.text((40, y2 + 30), "target", font=FB, fill=INK)
        dr.text((40, y3 + 10), "P(target)", font=FB, fill=INK); dr.text((40, y3 + 110), "−ln P", font=FB, fill=INK)
        for i in range(n):
            on = final or i <= k
            box(dr, [xs[i], y1, xs[i] + cw, y1 + 60], BLUE_L if on else BG2, BLUE if on else GRID)
            ctx = " ".join(show[:i + 1])
            dr.text((xs[i] + cw / 2, y1 + 30), show[i], font=FBig, fill=INK, anchor="mm")
            if on:
                dr.line([xs[i] + cw / 2, y1 + 62, xs[i] + cw / 2, y2 - 4], fill=GRID, width=2)
                box(dr, [xs[i], y2, xs[i] + cw, y2 + 60], GREEN_L, GREEN)
                dr.text((xs[i] + cw / 2, y2 + 30), show[i + 1], font=FBig, fill=INK, anchor="mm")
                p = nx[i]["p_true"]; L = losses[i]
                bh = 80 * min(1, p / 0.5)
                dr.rectangle([xs[i] + 40, y3 + 80 - bh, xs[i] + cw - 40, y3 + 80], fill=GREEN)
                dr.rectangle([xs[i] + 40, y3, xs[i] + cw - 40, y3 + 80], outline=GRID)
                dr.text((xs[i] + cw / 2, y3 + 94), f"{p:.4f}" if p < 0.01 else f"{p:.3f}", font=FM, fill=INK, anchor="mm")
                dr.text((xs[i] + cw / 2, y3 + 124), f"{L:.2f}", font=FB, fill=RED if L > 3 else GREEN, anchor="mm")
                g = nx[i]["top"][0][0]
                g = "a new line" if g == "\n" else f"“{g.strip()}”"
                dr.text((xs[i] + cw / 2, y3 + 156), f"top guess: {g}", font=FSm, fill=MUTE, anchor="mm")
        if final:
            avg = sum(losses) / n
            dr.text((W / 2, H - 50), f"average loss = ({' + '.join(f'{l:.2f}' for l in losses)}) / 5 = {avg:.2f}", font=FB, fill=INK, anchor="mm")
            dr.text((W / 2, H - 20), "natural log; average of the unrounded losses", font=FSm, fill=MUTE, anchor="mm")
        return im
    fr.append(frame(-1, "Training: predict the next token at every position", "Each target is the token that comes right after the input token above it. No human labels are needed: the text labels itself.", final=False)); du.append(3000)
    for k in range(n):
        p = nx[k]["p_true"]
        note = f"After “{' '.join(show[:k + 1])}”, GPT-2 gives the true next token “{show[k + 1]}” a probability of {p:.4f}. Loss = −ln P = {losses[k]:.2f}" + (": an easy, expected word." if p > 0.1 else ": a surprise, so a large loss.")
        fr.append(frame(k, "Training: predict the next token at every position", note)); du.append(2800)
    fr.append(frame(n, "All five predictions come from one forward pass", "The causal mask lets the model compute every position at once without peeking ahead. Training lowers this average loss over billions of such positions.", final=True)); du.append(6000)
    save(fr, du, "gpt_next_token_training")

if __name__ == "__main__":
    tokens_gif(); mask_gif(); block_gif(); training_gif()
