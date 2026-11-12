import hashlib
import numpy as np
from PIL import Image

def geometry(width, height):
    if width <= 0 or height <= 0:
        raise ValueError('Invalid image dimensions')
    # torchvision/HF shortest-edge integer truncation and center-crop floor.
    if width <= height:
        rw, rh = 592, int(592 * height / width)
    else:
        rw, rh = int(592 * width / height), 592
    left, top = (rw - 518) // 2, (rh - 518) // 2
    return {'original_width': width, 'original_height': height,
            'resize_width': rw, 'resize_height': rh,
            'crop_left': left, 'crop_top': top, 'crop_size': 518}

def transform(image, geom, mask=False):
    resampling = Image.Resampling.NEAREST if mask else Image.Resampling.BICUBIC
    im = image.resize((geom['resize_width'], geom['resize_height']), resampling)
    x, y = geom['crop_left'], geom['crop_top']
    return im.crop((x, y, x + 518, y + 518))

def binary_mask(raw):
    raw = np.asarray(raw)
    if raw.ndim != 2 or not np.all((raw >= 0) & (raw <= 115)):
        raise ValueError('Frozen raw mask gate violated')
    return ((raw >= 1) & (raw <= 115)).astype(np.uint8)

def original_box(binary):
    y, x = np.nonzero(binary)
    out = np.zeros_like(binary, dtype=np.uint8)
    if len(x) == 0:
        return out, None
    bbox = [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1]
    out[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
    return out, bbox

def patch_weights(binary):
    if binary.shape != (518, 518) or not np.all((binary == 0) | (binary == 1)):
        raise ValueError('Expected transformed binary 518x518 mask')
    return binary.reshape(37, 14, 37, 14).mean(axis=(1, 3), dtype=np.float64)

def random_shift(binary, image_id):
    digest = hashlib.sha256(b'1' + image_id.encode('utf-8')).digest()
    dx, dy = int.from_bytes(digest[:8], 'big') % 37, int.from_bytes(digest[8:16], 'big') % 37
    if dx == 0 and dy == 0:
        dx = 1
    shifted = np.roll(binary, (dy * 14, dx * 14), axis=(0, 1))
    a, b = patch_weights(binary), patch_weights(shifted)
    if binary.sum() != shifted.sum() or not np.array_equal(np.sort(a.ravel()), np.sort(b.ravel())):
        raise AssertionError('R1 area/weight distribution changed')
    if not np.array_equal(b, np.roll(a, (dy, dx), axis=(0, 1))):
        raise AssertionError('R1 patch alignment changed')
    return shifted, {'shift_dx_grid': dx, 'shift_dy_grid': dy,
                     'shift_dx_pixels': dx * 14, 'shift_dy_pixels': dy * 14}

def pool(tokens, mask_weights, box_weights, random_weights):
    if tokens.shape != (1370, 768):
        raise ValueError(f'Unexpected DINO token shape {tokens.shape}')
    cls, patches = tokens[0], tokens[1:]
    global_mean = patches.mean(axis=0, dtype=np.float32)
    def weighted(w):
        w = np.asarray(w, np.float32).ravel()
        return global_mean.copy() if w.sum() == 0 else (w @ patches / w.sum()).astype(np.float32)
    box, mask, random = weighted(box_weights), weighted(mask_weights), weighted(random_weights)
    return dict(G1=cls.copy(), G2=global_mean, B1=box, M1=mask,
                B2=np.concatenate([cls, box]), M2=np.concatenate([cls, mask]), R1=random)
