import sys, re
from PIL import Image, ImageDraw, ImageFont
out = sys.argv[1]
def font(name, size, idx=0):
    for p in (f"/System/Library/Fonts/{name}", f"/System/Library/Fonts/Supplemental/{name}"):
        try: return ImageFont.truetype(p, size, index=idx)
        except Exception: pass
    return ImageFont.load_default()
F_T=font("Helvetica.ttc",30,1); F_S=font("Helvetica.ttc",19); F_SB=font("Helvetica.ttc",19,1); F_M=font("Menlo.ttc",17); F_C=font("Menlo.ttc",14); F_CB=font("Menlo.ttc",15)
BG=(250,250,247); INK=(28,30,34); MUTE=(120,124,130); ACC=(26,137,23); ACC_BG=(226,244,225); HI=(196,70,48); HI_BG=(250,228,222); LINE=(222,224,228); CHIP=(236,238,241)
docs=["The quick brown fox jumps over the lazy dog.","The brown dog barks loudly.","The lazy cat sleeps peacefully."]
vocab="the quick brown fox jumps over lazy dog barks loudly cat sleeps peacefully".split()
toks=[re.sub(r'[^\w\s]','',d.lower()).split() for d in docs]
W_,H_=1200,800
M={a:{b:0 for b in vocab} for a in vocab}
def draw(di, ti, new_cells, note, final=False):
    im=Image.new("RGB",(W_,H_),BG); d=ImageDraw.Draw(im)
    d.text((40,26),"Building a co-occurrence matrix (window size 1)",font=F_T,fill=INK)
    d.text((40,68),"Each word counts the words immediately to its left and right.",font=F_S,fill=MUTE)
    d.line((40,102,W_-40,102),fill=LINE,width=2)
    y=118
    for k,t in enumerate(toks):
        x=40; d.text((x,y+8),f"Doc {k+1}",font=F_SB,fill=MUTE); x+=80
        for j,w in enumerate(t):
            bg,fg=CHIP,INK
            if k==di and j==ti: bg,fg=ACC_BG,ACC
            elif k==di and abs(j-ti)==1: bg,fg=HI_BG,HI
            tw=d.textlength(w,font=F_M)+18
            d.rounded_rectangle((x,y,x+tw,y+36),8,fill=bg); d.text((x+9,y+8),w,font=F_M,fill=fg); x+=tw+8
        y+=46
    d.text((40,y+6),note,font=F_SB if not final else F_SB,fill=ACC if not final else INK)
    # matrix
    x0,y0=125,y+88; cw,ch=80,27
    for c,v in enumerate(vocab):
        d.text((x0+c*cw+cw-6-d.textlength(v,font=F_C),y0-24-(16 if c%2 else 0)),v,font=F_C,fill=MUTE)
    for r,a in enumerate(vocab):
        yy=y0+r*ch
        d.text((x0-10-d.textlength(a,font=F_C),yy+5),a,font=F_C,fill=MUTE)
        for c,b in enumerate(vocab):
            xx=x0+c*cw; cell=(a,b)
            if cell in new_cells: d.rectangle((xx+2,yy+1,xx+cw-2,yy+ch-1),fill=ACC_BG)
            v=M[a][b]
            col=ACC if cell in new_cells else (INK if v else (200,203,208))
            s=str(v); d.text((xx+cw-14-d.textlength(s,font=F_CB),yy+5),s,font=F_CB,fill=col)
        d.line((x0,yy+ch,x0+len(vocab)*cw,yy+ch),fill=LINE)
    return im
frames=[]; durs=[]
frames.append(draw(-1,-1,set(),"Start: every count is 0.")); durs.append(1800)
for di,t in enumerate(toks):
    for ti,w in enumerate(t):
        cells=set()
        for j in (ti-1,ti+1):
            if 0<=j<len(t): M[w][t[j]]+=1; cells.add((w,t[j]))
        nb=", ".join(f"“{t[j]}”" for j in (ti-1,ti+1) if 0<=j<len(t))
        frames.append(draw(di,ti,cells,f"“{w}”: add 1 for {nb}")); durs.append(650)
frames.append(draw(-1,-1,{("the","lazy"),("lazy","the")},"Done. “the” and “lazy” are neighbours twice (Doc 1 and Doc 3), so that cell is 2.",final=True)); durs.append(5000)
pal=[f.convert("P",palette=Image.ADAPTIVE,colors=64) for f in frames]
pal[0].save(out,save_all=True,append_images=pal[1:],duration=durs,loop=0,optimize=True)
frames[-1].save(out.replace(".gif","_last.png")); frames[8].save(out.replace(".gif","_mid.png"))
print(len(frames),"frames")
