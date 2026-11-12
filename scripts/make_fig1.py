import sys, copy
from PIL import Image, ImageDraw
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from lxml import etree

from pathlib import Path
import argparse
ap=argparse.ArgumentParser(description='Rebuild the authored diagram after acquiring the public photograph separately.')
ap.add_argument('--out',type=Path,default=Path('generated/Figure_1.pptx'))
ap.add_argument('--assets',type=Path,default=Path('inputs/figure1'))
ap.add_argument('--patch-image',type=Path)
args=ap.parse_args()
OUT=str(args.out);args.out.parent.mkdir(parents=True,exist_ok=True)
B=str(args.assets)+'/'
PATCH_IMG=str(args.patch_image) if args.patch_image else None

GREY=RGBColor(0xEC,0xEC,0xEC); GREEN=RGBColor(0xD5,0xE8,0xCF); LINE=RGBColor(0x40,0x40,0x40)
BLACK=RGBColor(0,0,0); SEP=RGBColor(0xBF,0xBF,0xBF); NOTE=RGBColor(0x59,0x59,0x59)
FONT='Times New Roman'; FS=8.5; FT=10

# patch-grid thumbnail (schematic placeholder for the patch-token branch)
if PATCH_IMG is None:
    im=Image.open(B+'input518.png').convert('RGB').resize((518,518))
    d=ImageDraw.Draw(im); n=12
    for k in range(1,n):
        v=round(k*518/n); d.line([(v,0),(v,517)],fill=(255,255,255),width=2); d.line([(0,v),(517,v)],fill=(255,255,255),width=2)
    PATCH_IMG=B+'patch_grid.png'; im.save(PATCH_IMG)

prs=Presentation(); prs.slide_width=Emu(6120000); prs.slide_height=Emu(4080000)
s=prs.slides.add_slide(prs.slide_layouts[6])
I=Inches

def text(x,y,w,h,lines,size=FS,bold=False,italic=False,align=PP_ALIGN.LEFT,color=BLACK,anchor=MSO_ANCHOR.TOP):
    tb=s.shapes.add_textbox(I(x),I(y),I(w),I(h)); tf=tb.text_frame; tf.word_wrap=True
    tf.margin_left=tf.margin_right=tf.margin_top=tf.margin_bottom=0; tf.vertical_anchor=anchor
    if isinstance(lines,str): lines=[lines]
    for i,ln in enumerate(lines):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph(); p.alignment=align
        segs=ln if isinstance(ln,list) else [(ln,bold,italic)]
        for t,b,it in segs:
            r=p.add_run(); r.text=t; f=r.font; f.name=FONT; f.size=Pt(size); f.bold=b; f.italic=it; f.color.rgb=color
    return tb

def box(x,y,w,h,lines,fill=None,size=FS):
    sh=s.shapes.add_shape(MSO_SHAPE.RECTANGLE,I(x),I(y),I(w),I(h))
    if fill is None: sh.fill.background()
    else: sh.fill.solid(); sh.fill.fore_color.rgb=fill
    sh.line.color.rgb=LINE; sh.line.width=Pt(0.75); sh.shadow.inherit=False
    tf=sh.text_frame; tf.word_wrap=True; tf.margin_left=tf.margin_right=I(0.03); tf.margin_top=tf.margin_bottom=0
    tf.vertical_anchor=MSO_ANCHOR.MIDDLE
    if isinstance(lines,str): lines=[lines]
    for i,ln in enumerate(lines):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph(); p.alignment=PP_ALIGN.CENTER
        r=p.add_run(); r.text=ln; f=r.font; f.name=FONT; f.size=Pt(size); f.color.rgb=BLACK
    return sh

def line(x1,y1,x2,y2,arrow=False,color=LINE,width=0.75,dash=False):
    c=s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,I(x1),I(y1),I(x2),I(y2))
    c.line.color.rgb=color; c.line.width=Pt(width)
    ln=c.line._get_or_add_ln()
    if dash:
        pd=etree.SubElement(ln,qn('a:prstDash')); pd.set('val','dash')
    if arrow:
        te=etree.SubElement(ln,qn('a:tailEnd')); te.set('type','triangle'); te.set('w','sm'); te.set('len','sm')
    return c

def circle_plus(cx,cy,r):
    sh=s.shapes.add_shape(MSO_SHAPE.OVAL,I(cx-r),I(cy-r),I(2*r),I(2*r))
    sh.fill.solid(); sh.fill.fore_color.rgb=RGBColor(255,255,255); sh.line.color.rgb=LINE; sh.line.width=Pt(0.75); sh.shadow.inherit=False
    line(cx-r*0.55,cy,cx+r*0.55,cy); line(cx,cy-r*0.55,cx,cy+r*0.55)

# ---------------- (a) ----------------
text(0.04,0.0,3.0,0.2,'(a) Single backbone',size=FT)
# legend in title row
for x,fill,lab in [(4.80,GREY,'Frozen'),(5.62,GREEN,'Trained')]:
    sh=s.shapes.add_shape(MSO_SHAPE.RECTANGLE,I(x),I(0.06),I(0.17),I(0.12)); sh.fill.solid(); sh.fill.fore_color.rgb=fill
    sh.line.color.rgb=LINE; sh.line.width=Pt(0.75); sh.shadow.inherit=False
    text(x+0.22,0.03,0.6,0.18,lab)
s.shapes.add_picture(B+'input518.png',I(0.10),I(0.58),I(0.74),I(0.74))
text(0.0,1.36,0.94,0.18,'Input image',align=PP_ALIGN.CENTER)
line(0.84,0.95,0.98,0.95,arrow=True)
box(0.98,0.73,0.84,0.44,['DINOv2-L','(frozen)'],GREY)
line(1.82,0.95,2.0,0.95)
line(2.0,0.42,2.0,1.69)
line(2.0,0.42,2.62,0.42,arrow=True); text(2.03,0.22,0.58,0.18,'Class token')
line(2.0,1.69,2.10,1.69)
s.shapes.add_picture(PATCH_IMG,I(2.10),I(1.49),I(0.40),I(0.40))
text(2.03,1.27,1.3,0.18,'Patch tokens (37 × 37)')
line(2.50,1.69,2.62,1.69,arrow=True)
box(2.62,0.23,1.12,0.38,['Class-token probe','(fitted, then frozen)'],GREY)
box(2.62,1.50,1.12,0.38,['Attention pooling','+ output projection'],GREEN)
line(3.74,0.42,4.75,0.42,arrow=True); text(3.80,0.22,0.5,0.18,'Logits')
line(4.25,0.42,4.25,1.59,arrow=True)
line(3.74,1.69,4.15,1.69,arrow=True); text(3.67,1.49,0.56,0.16,'Residual',italic=True,align=PP_ALIGN.CENTER)
circle_plus(4.25,1.69,0.10)
line(4.35,1.69,4.75,1.69,arrow=True)
box(4.75,0.29,0.62,0.27,'Softmax'); line(5.37,0.42,5.52,0.42,arrow=True)
text(5.55,0.27,1.1,0.32,['Class-token','prediction'])
box(4.75,1.56,0.62,0.27,'Softmax'); line(5.37,1.69,5.52,1.69,arrow=True)
text(5.55,1.54,1.1,0.32,['Attentive','prediction'])

line(0.08,2.12,6.62,2.12,color=SEP)

# ---------------- (b) ----------------
text(0.04,2.20,3.6,0.2,'(b) Three-backbone fusion',size=FT)
rows=[2.62,3.04,3.46]; h=0.32; xg=1.50; wg=1.36
box(0.10,rows[0],1.12,h,['DINOv2-L','class-token probe'],GREY)
line(1.22,rows[0]+h/2,xg,rows[0]+h/2,arrow=True); text(1.19,rows[0]+h/2-0.17,0.34,0.16,'swap',italic=True,align=PP_ALIGN.CENTER)
box(xg,rows[0],wg,h,['DINOv2-L','attentive probe'],GREEN)
box(xg,rows[1],wg,h,['DINOv3-L','class-token probe'],GREY)
box(xg,rows[2],wg,h,['CLIP-L','image-embedding probe'],GREY)
text(0.10,2.44,1.12,0.16,'Three-probe fusion',italic=True,align=PP_ALIGN.CENTER)
text(xg,2.44,wg,0.16,'Attentive swap',italic=True,align=PP_ALIGN.CENTER)
text(0.10,3.06,1.30,0.60,['Backbones and the','fusion rule stay fixed;','only the DINOv2-L','classifier changes.'],size=8,italic=True,color=NOTE)
bx=xg+wg+0.13
for y in rows: line(xg+wg,y+h/2,bx,y+h/2)
line(bx,rows[0]+h/2,bx,rows[2]+h/2)
mx=bx+0.12; my=rows[1]+h/2
line(bx,my,mx,my,arrow=True)
box(mx,my-0.28,0.80,0.56,['Mean of class','probabilities'])
line(mx+0.40,my+0.28,mx+0.40,my+0.46,arrow=True)
text(mx-0.09,my+0.48,0.98,0.18,'Prediction',align=PP_ALIGN.CENTER)

line(4.02,2.22,4.02,4.40,color=SEP)

# ---------------- (c) ----------------
text(4.10,2.20,2.55,0.2,'(c) An error corrected by the attentive probe',size=FT)
s.shapes.add_picture(B+'input518.png',I(4.14),I(2.50),I(1.0),I(1.0))
s.shapes.add_picture(B+'attn_spot.png',I(5.32),I(2.50),I(1.0),I(1.0))
text(4.04,3.52,1.2,0.16,'Input image',align=PP_ALIGN.CENTER)
text(5.12,3.52,1.4,0.16,'Attention, five-seed mean',align=PP_ALIGN.CENTER)
text(4.14,3.76,2.5,0.6,[[('True: ',False,False),('tomato early blight',False,False)],
                        [('Class-token probe: ',False,False),('tomato bacterial spot',False,False)],
                        [('Attentive probe: ',False,False),('tomato early blight',True,False),(' (5 of 5 seeds)',False,False)]])
# remove theme styles (they add a soft shadow) and set an explicit empty effect list on every shape
from pptx.oxml.ns import qn as _qn
for shp in s.shapes:
    el=shp._element
    for st in el.findall(_qn('p:style')):
        el.remove(st)
    spPr=el.find(_qn('p:spPr'))
    if spPr is not None:
        for e in spPr.findall(_qn('a:effectLst')):
            spPr.remove(e)
        eff=etree.SubElement(spPr,_qn('a:effectLst'))
        # effectLst must follow ln if present; move it to the end of spPr children order
        ln=spPr.find(_qn('a:ln'))
        if ln is not None:
            spPr.remove(eff); ln.addnext(eff)
prs.save(OUT); print('saved',OUT)
