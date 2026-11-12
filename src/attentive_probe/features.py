"""Batch adapters preserving the original preprocessing and token extraction.

Models and (for CLIP) its official processor are supplied by the caller. Models
must be frozen, in evaluation mode and on CUDA. No model is loaded at import.
"""
import numpy as np
import torch
from PIL import Image
from .geometry import geometry, transform

def extract_batch(images, model, backbone, processor=None, dinov3_patch_dtype="float16"):
    """Return global, patch and mean-patch arrays for caller-provided PIL images.

    dinov3_patch_dtype is float16 for PlantWild and float32 for PlantSeg.
    CLIP globals are projected 768-D; patches are pre-post-layernorm 1024-D.
    DINOv2 mean patches are computed from the stored float16 patch values.
    """
    if backbone not in {"dinov2", "dinov3", "clip"}:
        raise ValueError(backbone)
    views=[]
    for image in images:
        im=image.convert("RGB")
        view=transform(im,geometry(*im.size),mask=False)
        views.append(view.resize((512,512),Image.Resampling.BICUBIC) if backbone=="dinov3" else view)
    if backbone=="clip":
        if processor is None:
            raise ValueError("CLIP requires its official image processor")
        pixels=processor(images=views,return_tensors="pt")["pixel_values"].cuda()
    else:
        pix=np.stack([(np.asarray(v,np.float32)/255-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32) for v in views]).transpose(0,3,1,2)
        pixels=torch.from_numpy(pix.copy()).cuda()
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        out=model(pixel_values=pixels)
    tokens=out.last_hidden_state.float().cpu().numpy()
    if backbone=="dinov2":
        assert tokens.shape[1:]==(1370,1024)
        patches=tokens[:,1:].astype(np.float16)
        return tokens[:,0].astype(np.float16),patches,patches.astype(np.float32).mean(1)
    if backbone=="dinov3":
        assert tokens.shape[1:]==(1029,1024)
        assert model.config.num_register_tokens==4
        assert torch.equal(out.pooler_output,out.last_hidden_state[:,0,:])
        return out.pooler_output.float().cpu().numpy(),tokens[:,5:].astype(dinov3_patch_dtype),None
    assert tokens.shape[1:]==(257,1024)
    return out.image_embeds.float().cpu().numpy(),tokens[:,1:].astype(np.float16),None
