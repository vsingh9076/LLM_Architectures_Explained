import sys, collections
from PIL import Image, ImageDraw, ImageFont
out = sys.argv[1]
def font(name, size, idx=0):
    for p in (f"/System/Library/Fonts/{name}", f"/System/Library/Fonts/Supplemental/{name}", f"/Library/Fonts/{name}"):
        try: return ImageFont.truetype(p, size, index=idx)
        except Exception: pass
    return ImageFont.load_default()
W, H = 1200, 560
BG=(250,250,247); INK=(28,30,34); MUTE=(120,124,130); ACC=(26,137,23); ACC_BG=(226,244,225); CHIP=(236,238,241); WARN=(196,70,48); WARN_BG=(250,228,222); LINE=(222,224,228)
F_T=font("Helvetica.ttc",34,1); F_S=font("Helvetica.ttc",22); F_SB=font("Helvetica.ttc",22,1); F_M=font("Menlo.ttc",24); F_MS=font("Menlo.ttc",20); F_C=font("Helvetica.ttc",17)
def tw(d,t,f): return d.textlength(t,font=f)
def chip(d,x,y,t,f=F_M,bg=CHIP,fg=INK,strike=False,pad=12,h=44):
    w=tw(d,t,f)+2*pad; d.rounded_rectangle((x,y,x+w,y+h),10,fill=bg); d.text((x+pad,y+(h-f.size)/2-2),t,font=f,fill=fg)
    if strike: d.line((x+6,y+h/2,x+w-6,y+h/2),fill=fg,width=2)
    return w
def base(title, sub):
    im=Image.new("RGB",(W,H),BG); d=ImageDraw.Draw(im)
    d.text((48,34),title,font=F_T,fill=INK); d.text((48,82),sub,font=F_S,fill=MUTE); d.line((48,124,W-48,124),fill=LINE,width=2)
    return im,d
def save(frames, durs, path):
    pal=[f.convert("P",palette=Image.ADAPTIVE,colors=64) for f in frames]
    pal[0].save(path,save_all=True,append_images=pal[1:],duration=durs,loop=0,optimize=True)

# ---------- GIF A: preprocessing pipeline (mirrors the article's own code example) ----------
steps=[("Raw text",None),("Lowercase",None),("Remove punctuation",None),("Tokenize",None),("Remove stop words",None),("Stem (Porter)",None),("Lemmatize (WordNet)",None)]
raw="Running is better than walking! Apples and oranges are different."
low=raw.lower(); nop="running is better than walking apples and oranges are different"
toks=nop.split(); stop={"is","than","and","are"}; kept=[t for t in toks if t not in stop]
stem=['run','better','walk','appl','orang','differ']; lem=['running','better','walking','apple','orange','different']
def frameA(k):
    im,d=base("Text pre-processing, step by step","The same sentence used in the code example below")
    x0=48; y=150
    for i,(name,_) in enumerate(steps):
        on = i==k; done = i<k
        d.rounded_rectangle((x0,y,x0+300,y+44),10,fill=ACC_BG if on else BG, outline=ACC if on else LINE, width=2)
        d.ellipse((x0+12,y+12,x0+32,y+32),fill=ACC if (on or done) else LINE)
        d.text((x0+44,y+10),f"{i+1}. {name}",font=F_SB if on else F_S,fill=INK if (on or done) else MUTE)
        y+=54
    px=400; d.text((px,150),"Result",font=F_SB,fill=MUTE)
    def wrap_text(t,y0,f=F_M):
        words=t.split(" "); line=""; yy=y0
        for w_ in words:
            if tw(d,line+w_+" ",f)>W-px-60: d.text((px,yy),line,font=f,fill=INK); yy+=40; line=""
            line+=w_+" "
        d.text((px,yy),line,font=f,fill=INK); return yy+40
    def chips(items,y0,strike=set(),bg=CHIP,fg=INK):
        x=px; yy=y0
        for t in items:
            w_=tw(d,t,F_M)+24
            if x+w_>W-48: x=px; yy+=56
            s=t in strike
            chip(d,x,yy,t,bg=WARN_BG if s else bg,fg=WARN if s else fg,strike=s); x+=w_+10
        return yy+56
    if k==0: wrap_text(raw,196)
    elif k==1: wrap_text(low,196); d.text((px,300),"“Running” and “running” now count as the same word.",font=F_S,fill=MUTE)
    elif k==2: wrap_text(nop,196); d.text((px,300),"“!” and “.” are gone.",font=F_S,fill=MUTE)
    elif k==3: yy=chips(toks,196); d.text((px,yy+16),f"{len(toks)} tokens",font=F_S,fill=MUTE)
    elif k==4: yy=chips(toks,196,strike=stop); d.text((px,yy+16),f"{len(kept)} tokens left after dropping is / than / and / are",font=F_S,fill=MUTE)
    elif k==5:
        yy=chips(stem,196,bg=ACC_BG); d.text((px,yy+16),"Fast, rule-based suffix stripping.",font=F_S,fill=MUTE); d.text((px,yy+46),"Not always real words: “appl”, “orang”, “differ”.",font=F_S,fill=MUTE)
    elif k==6:
        yy=chips(lem,196,bg=ACC_BG); d.text((px,yy+16),"Dictionary forms: “apples” → “apple”.",font=F_S,fill=MUTE); d.text((px,yy+46),"“running” stays as-is because the default part of speech is noun.",font=F_S,fill=MUTE)
    return im
fa=[frameA(k) for k in range(7)]
save(fa,[1800,1600,1600,1800,2200,2400,3200],f"{out}/preprocessing_pipeline.gif"); fa[4].save(f"{out}/_a4.png"); fa[6].save(f"{out}/_a6.png")

# ---------- GIF B: BPE merges (toy corpus from the Hugging Face LLM Course, ch. 6) ----------
corpus=collections.OrderedDict([("hug",10),("pug",5),("pun",12),("bun",4),("hugs",5)])
splits={w:list(w) for w in corpus}
vocab=sorted(set("".join(corpus)))
def pair_counts():
    c=collections.Counter()
    for w,f in corpus.items():
        s=splits[w]
        for a,b in zip(s,s[1:]): c[(a,b)]+=f
    return c
def frameB(step, vocab, hi=None, note="", merged=None):
    im,d=base("Byte-Pair Encoding (BPE): learning subword tokens","Repeatedly merge the most frequent adjacent pair of symbols")
    d.text((48,146),"Corpus (word × frequency)",font=F_SB,fill=MUTE)
    y=186
    for w,f in corpus.items():
        x=48; s=splits[w]; i=0
        while i<len(s):
            t=s[i]; is_hi = hi and i+1<len(s) and (s[i],s[i+1])==hi
            is_m = merged and t==merged
            if is_hi:
                w1=chip(d,x,y,s[i],bg=WARN_BG,fg=WARN); x+=w1+4; w2=chip(d,x,y,s[i+1],bg=WARN_BG,fg=WARN); x+=w2+8; i+=2; continue
            w1=chip(d,x,y,t,bg=ACC_BG if is_m else CHIP,fg=ACC if is_m else INK); x+=w1+8; i+=1
        d.text((330,y+9),f"× {f}",font=F_MS,fill=MUTE); y+=58
    px=470; d.text((px,146),f"Step {step}",font=F_SB,fill=MUTE)
    yy=186
    for line in note.split("\n"):
        d.text((px,yy),line,font=F_S,fill=INK); yy+=34
    d.text((px,360),f"Vocabulary ({len(vocab)} tokens)",font=F_SB,fill=MUTE)
    x=px; yv=400
    for t in vocab:
        w_=tw(d,t,F_MS)+20
        if x+w_>W-48: x=px; yv+=50
        is_m = merged and t==merged
        chip(d,x,yv,t,f=F_MS,bg=ACC_BG if is_m else CHIP,fg=ACC if is_m else INK,pad=10,h=38); x+=w_+8
    return im
fb=[]; durs=[]
fb.append(frameB(0,vocab,note="Start: every word is split into characters.\nThe vocabulary is just the characters seen.")); durs.append(2600)
merges=[]
for step in (1,2,3):
    c=pair_counts(); (a,b),n=max(c.items(), key=lambda kv:(kv[1],)) 
    fb.append(frameB(step,vocab,hi=(a,b),note=f"Most frequent pair:  “{a}” + “{b}”\nIt appears {n} times in the corpus.")); durs.append(2400)
    new=a+b; merges.append((a,b,n))
    for w in corpus:
        s=splits[w]; i=0; o=[]
        while i<len(s):
            if i+1<len(s) and s[i]==a and s[i+1]==b: o.append(new); i+=2
            else: o.append(s[i]); i+=1
        splits[w]=o
    vocab=vocab+[new]
    fb.append(frameB(step,vocab,merged=new,note=f"Merge them into one new token:  “{new}”\nAdd it to the vocabulary and repeat.")); durs.append(2400)
fb.append(frameB(3,vocab,note="After 3 merges, “hug” is a single token and\n“hugs” is “hug” + “s”. A new word like “bug”\nwould be split as “b” + “ug”.")); durs.append(4200)
save(fb,durs,f"{out}/bpe_merges.gif"); fb[1].save(f"{out}/_b1.png"); fb[-1].save(f"{out}/_b7.png")
print("merges:", merges, "| final splits:", dict(splits))
import os
for f in ("preprocessing_pipeline.gif","bpe_merges.gif"): print(f, os.path.getsize(f"{out}/{f}"), "bytes")