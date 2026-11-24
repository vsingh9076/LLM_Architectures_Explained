# Vertical timeline of the GPT family, 2018 to November 2026 (dates checked against OpenAI's pages, papers and Wikipedia).
import sys
from PIL import Image, ImageDraw, ImageFont
out = sys.argv[1]
def font(size, bold=False):
    for p in (["/System/Library/Fonts/Supplemental/Arial Bold.ttf"] if bold else ["/System/Library/Fonts/Supplemental/Arial.ttf"]):
        try: return ImageFont.truetype(p, size)
        except Exception: pass
    return ImageFont.load_default()
FT, FB, FS, FSm = font(30, True), font(22, True), font(20), font(17)
INK, MUTE, GRID = (33, 37, 41), (110, 116, 124), (205, 210, 216)
OPEN, OPEN_L, PAPER, PAPER_L, CLOSED, CLOSED_L = (4, 120, 87), (209, 250, 229), (37, 99, 235), (219, 234, 254), (107, 114, 128), (233, 235, 238)
rows = [
 ("Jun 2018", "GPT-1", "117M parameters; pre-train, then fine-tune", "open"),
 ("Feb 2019", "GPT-2", "1.5B; zero-shot tasks from a prompt", "open"),
 ("May 2020", "GPT-3", "175B; few-shot in-context learning", "paper"),
 ("Mar 2022", "InstructGPT", "RLHF: learning to follow instructions", "paper"),
 ("Nov 2022", "ChatGPT", "the chat app built on these models", "closed"),
 ("Mar 2023", "GPT-4", "images as input; no architecture details", "closed"),
 ("May 2024", "GPT-4o", "text, audio and images in one model", "closed"),
 ("Sep 2024", "o1", "OpenAI’s first reasoning model", "closed"),
 ("Aug 2025", "gpt-oss", "open-weight MoE models (120b, 20b)", "open"),
 ("Aug 2025", "GPT-5", "fast and thinking models with a router", "closed"),
 ("Apr 2026", "GPT-5.5", "", "closed"),
 ("Jul 2026", "GPT-5.6", "", "closed"),
 ("Sep 2026", "GPT-6 Astra · Sol · Luna", "new generation, from frontier to fast", "closed"),
 ("Sep–Oct 2026", "GPT-6.1 Sol · Terra", "cheaper near-Astra model; fast Terra", "closed"),
 ("Nov 2026", "GPT-6.2 Astra · Sol", "latest release (17 Nov 2026)", "closed"),
]
W, top, rh = 1040, 150, 56; H = top + rh * len(rows) + 40
im = Image.new("RGB", (W, H), "white"); dr = ImageDraw.Draw(im)
dr.text((40, 26), "The GPT family, 2018 to November 2026", font=FT, fill=INK)
lx = 40
for lab, c, cl in [("architecture and weights published", OPEN, OPEN_L), ("architecture published in a paper", PAPER, PAPER_L), ("closed: no architecture details", CLOSED, CLOSED_L)]:
    dr.rounded_rectangle([lx, 84, lx + 22, 106], radius=5, fill=cl, outline=c, width=2)
    dr.text((lx + 30, 95), lab, font=FSm, fill=MUTE, anchor="lm"); lx += 30 + dr.textlength(lab, font=FSm) + 28
xline = 230
dr.line([xline, top - 10, xline, top + rh * len(rows) - 20], fill=GRID, width=4)
for i, (d, name, note, kind) in enumerate(rows):
    y = top + i * rh + 8
    c, cl = {"open": (OPEN, OPEN_L), "paper": (PAPER, PAPER_L), "closed": (CLOSED, CLOSED_L)}[kind]
    last = i == len(rows) - 1
    dr.text((xline - 26, y + 14), d, font=FB if last else FS, fill=INK if last else MUTE, anchor="rm")
    dr.ellipse([xline - 11, y + 3, xline + 11, y + 25], fill=cl, outline=c, width=3)
    tw = dr.textlength(name, font=FB)
    dr.rounded_rectangle([xline + 30, y - 4, xline + 46 + tw, y + 32], radius=8, fill=cl, outline=c, width=3 if last else 2)
    dr.text((xline + 38, y + 14), name, font=FB, fill=INK, anchor="lm")
    if note: dr.text((xline + 64 + tw, y + 14), note, font=FS, fill=MUTE, anchor="lm")
im.save(out); print(out, im.size)
