"""Attention overlay for Figure 1(c).

Dims patches with low attention weight and keeps high-weight patches in colour.
Input: the 518x518 model-input crop and the 37x37 attention array averaged over
five seeds (softmax weights). Output: a 518x518 PNG.

Usage: python make_fig1_attention.py --image input518.png --attn attn_mean5.npy --out fig1_attention_overlay.png
"""
import argparse
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('--image', required=True)
ap.add_argument('--attn', required=True)
ap.add_argument('--out', required=True)
args = ap.parse_args()

a = np.load(args.attn).astype(float)                      # 37 x 37, sums to 1
im = np.asarray(Image.open(args.image).convert('RGB')).astype(float) / 255
t = np.asarray(Image.fromarray((a / a.max() * 255).astype(np.uint8))
               .resize(im.shape[1::-1], Image.BILINEAR)).astype(float) / 255
w = np.clip(t, 0, 1) ** 0.6                                # weight per pixel
gray = im.mean(2, keepdims=True)
base = 0.35 * gray + 0.10                                  # dimmed grey background
out = base * (1 - w[..., None]) + im * w[..., None]
Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).save(args.out)
