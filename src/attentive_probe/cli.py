"""Portable orchestration for the PlantWild core protocol.

This adapter calls the preserved scientific functions. It has been statically
checked only; no training or backbone inference was performed for packaging.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def read_manifest(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"image_id", "label", "split"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Manifest needs image_id,label,split columns")
    if len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError("Manifest image_id values must be unique")
    if not all(r["split"] in {"train", "val", "test", "exclude"} for r in rows):
        raise ValueError("Unknown split")
    return rows


def extract(args):
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModel, CLIPConfig, CLIPImageProcessor, CLIPVisionModelWithProjection
    from .features import extract_batch
    from .training import configure
    rows = read_manifest(args.manifest)
    if not {"relative_path", "sha256"}.issubset(rows[0]):
        raise ValueError("Extraction also requires relative_path and sha256")
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    configure(); torch.manual_seed(1)
    processor = None
    if args.backbone == "clip":
        conf = CLIPConfig.from_pretrained(args.model, local_files_only=True)
        vision = conf.vision_config; vision.projection_dim = conf.projection_dim
        model = CLIPVisionModelWithProjection.from_pretrained(args.model, config=vision, local_files_only=True)
        processor = CLIPImageProcessor.from_pretrained(args.model, local_files_only=True)
    else:
        model = AutoModel.from_pretrained(args.model, local_files_only=True)
    model = model.cuda().eval().requires_grad_(False)
    batch = 32 if args.backbone == "clip" else 8
    arrays = {}
    root = Path(args.images).resolve()
    for first in range(0, len(rows), batch):
        images = []
        for row in rows[first:first+batch]:
            path = (root / row["relative_path"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Image path must remain below the image root")
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError("Image checksum mismatch: " + row["image_id"])
            with Image.open(path) as im:
                images.append(im.convert("RGB"))
        values = extract_batch(images, model, args.backbone, processor, args.dinov3_patch_dtype)
        for name, value in zip(["global", "patch", "mean_patch"], values):
            if value is None:
                continue
            if name not in arrays:
                arrays[name] = np.lib.format.open_memmap(out/(name+".npy"), mode="w+", dtype=value.dtype, shape=(len(rows), *value.shape[1:]))
            arrays[name][first:first+len(images)] = value
            arrays[name].flush()
    # Preserve only public IDs, labels and roles, not local image locations.
    with (out/"index.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image_id", "label", "split"])
        writer.writeheader(); writer.writerows({k:r[k] for k in writer.fieldnames} for r in rows)


def train(args):
    import numpy as np
    import torch
    from scipy.special import softmax
    from . import training as tr
    from .io import Progress
    from .residual_mlp import train_residual, CLSResidual
    from .fusion import feature_blocks
    rows = read_manifest(args.manifest)
    y = np.array([int(r["label"]) for r in rows], dtype=np.int64)
    ix = {role:np.array([i for i,r in enumerate(rows) if r["split"]==role], dtype=np.int64) for role in ["train","val","test"]}
    if any(len(v)==0 for v in ix.values()):
        raise ValueError("Nonempty train, val and test roles are required")
    classes = int(y.max()+1)
    if not np.array_equal(np.unique(y[ix["train"]]), np.arange(classes)):
        raise ValueError("Training labels must cover contiguous class IDs")
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    tr.configure(); rt = Progress()
    blocks = [np.load(p, mmap_mode="r") for p in args.features]
    if any(len(x)!=len(rows) for x in blocks):
        raise ValueError("Features must match manifest row order exactly")
    x = feature_blocks(blocks) if args.head == "feature-fusion" else np.asarray(blocks[0], np.float32)
    if len(blocks)>1 and args.head!="feature-fusion":
        raise ValueError("Multiple feature blocks require feature-fusion")
    raw = json.loads(Path(args.config).read_text()) if args.config else {}
    candidates = raw if isinstance(raw,list) else raw.get("candidates", [raw.get("chosen",raw.get("selected",raw.get("configuration",raw)))])
    models=[]
    def predict(folder, which):
        ids=ix[which]
        if args.head=="feature-fusion":
            m,_=tr.load_global(folder/"HEAD.pt")
            z=tr.predict_global(m,x[ids],rt)
        elif args.head=="mlp":
            m=CLSResidual().cuda()
            m.load_state_dict(torch.load(folder/"HEAD.pt",map_location="cpu",weights_only=True))
            m.eval();zs=[]
            with torch.inference_mode():
                for start in range(0,len(ids),16):
                    ii=torch.as_tensor(ids[start:start+16],device="cuda")
                    zs.append(m(xx[ii].float(),g[ii]).cpu().numpy())
            z=np.concatenate(zs)
        else:
            m,_=tr.load_mil(folder/"HEAD.pt")
            z=tr.predict_mil(m,patches,g,ids,rt,variant)
        del m;torch.cuda.empty_cache()
        return z
    if args.head!="feature-fusion":
        grid=[float(c) for c in args.c_grid.split(",")]
        probe_candidates=[]
        for j,C in enumerate(grid):
            f=tr.lr_fit(x,y,ix["train"],C,out/f"probe_{j}")
            score=tr.metrics(y[ix["val"]],tr.lr_logits(f,x[ix["val"]]))
            probe_candidates.append((score["Accuracy_pct"],score["MacroF1_pct"],-C,C,f))
        selected=max(probe_candidates,key=lambda q:q[:3]);fit=selected[-1]
        (out/"probe_selection.json").write_text(json.dumps({"C":selected[-2]}))
        if args.head=="probe":
            z=tr.lr_logits(fit,x[ix["test"]]);models=[("deterministic",z)]
        else:
            g=torch.from_numpy(tr.lr_logits(fit,x)).cuda()
            if args.head=="mlp":
                if classes!=89:raise ValueError("This residual MLP recipe is specific to PlantWild")
                xx=torch.from_numpy(x).cuda()
            else:
                if not args.patches:raise ValueError("Attentive/uniform heads require --patches")
                cache=np.load(args.patches,mmap_mode="r")
                if len(cache)!=len(rows):raise ValueError("Patch rows do not match manifest")
                class PatchBatches:
                    def __getitem__(self, index):
                        ids=index.detach().cpu().numpy()
                        return torch.from_numpy(np.array(cache[ids],copy=True)).cuda()
                patches=PatchBatches()
                variant="UNIFORM" if args.head=="uniform" else "FULL"
    if args.head!="probe":
        if not args.config:raise ValueError("Neural fitting requires --config")
        def run(c,seed,folder):
            if args.head=="feature-fusion":
                tr.train_global(folder,c,seed,x,y,ix["train"],ix["val"],rt,classes)
            elif args.head=="mlp":
                folder.mkdir()
                m,log=train_residual(c,seed,xx,g,y,ix["train"])
                torch.save({k:v.detach().cpu() for k,v in m.state_dict().items()},folder/"HEAD.pt")
                (folder/"training.json").write_text(json.dumps(log));del m;torch.cuda.empty_cache()
            else:
                c=dict(c,batch=16,MIL_hidden=128,classification_loss="CE")
                tr.train_mil(folder,c,seed,patches,g,y,ix["train"],rt,classes,variant)
        if args.fixed_config:
            if len(candidates)!=1:
                raise ValueError("--fixed-config requires one selected configuration, not a candidate search")
            chosen=candidates[0];first_folder=out/"seed101"
            run(chosen,101,first_folder)
        else:
            search=[]
            for j,c in enumerate(candidates):
                folder=out/f"candidate_{j}_seed101";run(c,101,folder)
                score=tr.metrics(y[ix["val"]],predict(folder,"val"))
                search.append((score["Accuracy_pct"],score["MacroF1_pct"],-j,c,folder))
            winner=max(search,key=lambda q:q[:3]);chosen=winner[3];first_folder=winner[4]
        (out/"selection.json").write_text(json.dumps(chosen,indent=2))
        folders=[(101,first_folder)]
        for seed in [202,303,404,505]:
            folder=out/f"seed{seed}";run(chosen,seed,folder);folders.append((seed,folder))
        models=[(seed,predict(folder,"test")) for seed,folder in folders]
    for seed,z in models:
        np.save(out/f"probabilities_{seed}.npy",softmax(z.astype(np.float64),axis=1))
        with (out/f"predictions_{seed}.csv").open("w",newline="",encoding="utf-8") as stream:
            writer=csv.writer(stream);writer.writerow(["image_id","label","prediction"])
            writer.writerows((rows[i]["image_id"],int(y[i]),int(z[j].argmax())) for j,i in enumerate(ix["test"]))


def fuse(args):
    import numpy as np
    from .fusion import combine
    arrays=[np.load(p) for p in args.probabilities]
    if len(arrays)!=3 or any(a.shape!=arrays[0].shape for a in arrays):
        raise ValueError("Three identically ordered probability arrays are required")
    np.save(args.output,combine(arrays))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("extract")
    for name in ["manifest","images","model","output"]:p.add_argument("--"+name,required=True)
    p.add_argument("--backbone",choices=["dinov2","dinov3","clip"],required=True)
    p.add_argument("--dinov3-patch-dtype",choices=["float16","float32"],default="float16")
    p.set_defaults(func=extract)
    p=sub.add_parser("train")
    for name in ["manifest","output"]:p.add_argument("--"+name,required=True)
    p.add_argument("--features",nargs="+",required=True)
    p.add_argument("--head",choices=["probe","attentive","uniform","mlp","feature-fusion"],required=True)
    p.add_argument("--patches");p.add_argument("--config")
    p.add_argument("--fixed-config",action="store_true",help="Use one official-selected neural configuration without candidate re-selection")
    p.add_argument("--c-grid",default="0.001,0.01,0.1,1,10,100,1000,10000")
    p.set_defaults(func=train)
    p=sub.add_parser("fuse")
    p.add_argument("--probabilities",nargs=3,required=True);p.add_argument("--output",required=True);p.set_defaults(func=fuse)
    args=parser.parse_args();args.func(args)


if __name__=="__main__":main()
