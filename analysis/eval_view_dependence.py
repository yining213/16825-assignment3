"""Q4.1: compare NeRFs trained with and without view dependence on the materials scene.

Prints train/test PSNR for both models and saves a cropped test view (ground truth | without | with),
using the test view where view dependence helps most.

Usage:
    python analysis/eval_view_dependence.py --vd_ckpt CKPT --novd_ckpt CKPT --out DIR
"""
import argparse
import os

import imageio.v2 as imageio
import numpy as np

from common import load_config, load_model, render, psnr, view_indices, vrm

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="nerf_materials_512")
parser.add_argument("--vd_ckpt", required=True, help="checkpoint trained with view dependence")
parser.add_argument("--novd_ckpt", required=True, help="checkpoint trained without view dependence")
parser.add_argument("--n_views", type=int, default=40)
parser.add_argument("--out", required=True)
args = parser.parse_args()

cfg = load_config(args.config, **{"renderer.chunk_size": 8192})
W, H = cfg.data.image_size
train_ds, _, test_ds = vrm.get_nerf_datasets(dataset_name=cfg.data.dataset_name, image_size=[H, W])

models = {}
for name, vd, ckpt in (("without", False, args.novd_ckpt), ("with", True, args.vd_ckpt)):
    cfg.implicit_function.view_dependence = vd
    models[name] = load_model(cfg, ckpt)[0]


def evaluate(dataset, keep=False):
    scores, renders = {k: [] for k in models}, []
    for i in view_indices(dataset, args.n_views):
        item = dataset[int(i)]
        gt, cam = item["image"].cuda(), item["camera"].cuda()
        preds = {k: render(m, cam, (W, H)) for k, m in models.items()}
        for k, p in preds.items():
            scores[k].append(psnr(p, gt))
        if keep:
            renders.append((int(i), item["image"].numpy(), {k: p.clamp(0, 1).cpu().numpy() for k, p in preds.items()}))
    return {k: np.array(v) for k, v in scores.items()}, renders


train_psnr, _ = evaluate(train_ds)
test_psnr, renders = evaluate(test_ds, keep=True)

os.makedirs(args.out, exist_ok=True)
lines = ["| Model | Train PSNR | Test PSNR | Train - test |", "|---|---|---|---|"]
for k in models:
    tr, te = train_psnr[k].mean(), test_psnr[k].mean()
    lines.append(f"| {k} view dependence | {tr:.2f} | {te:.2f} | {tr - te:.2f} |")
gain = test_psnr["with"] - test_psnr["without"]
best = int(np.argmax(gain))
lines.append(f"\nPer-view test gain from view dependence: mean {gain.mean():.2f} dB, max {gain[best]:.2f} dB "
             f"(test view {renders[best][0]}), improved on {np.mean(gain > 0) * 100:.0f}% of views")
print("\n".join(lines))
with open(os.path.join(args.out, "view_dependence_psnr.md"), "w") as f:
    f.write("\n".join(lines) + "\n")

# Crop the best test view to the objects
idx, gt, preds = renders[best]
ys, xs = np.nonzero(gt.max(-1) > 0.05)
pad = H // 32
y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, H)
x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, W)
strip = np.concatenate([gt[y0:y1, x0:x1], preds["without"][y0:y1, x0:x1], preds["with"][y0:y1, x0:x1]], axis=1)
imageio.imwrite(os.path.join(args.out, "q4_1_test_view.png"), (np.clip(strip, 0, 1) * 255).astype(np.uint8))
