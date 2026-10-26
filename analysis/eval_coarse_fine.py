"""Q4.2: compare the 512x512 lego NeRFs with and without coarse/fine sampling.

Prints parameters, test PSNR, training-step time and render time for both, and saves one test view
(ground truth | without | with).

Usage:
    python analysis/eval_coarse_fine.py --base_ckpt CKPT --cf_ckpt CKPT --out DIR
"""
import argparse
import os
import time

import imageio.v2 as imageio
import numpy as np
import torch

from common import load_config, load_model, render, psnr, view_indices, vrm

parser = argparse.ArgumentParser()
parser.add_argument("--base_ckpt", required=True, help="checkpoint of nerf_lego_512")
parser.add_argument("--cf_ckpt", required=True, help="checkpoint of nerf_lego_512_coarse_fine")
parser.add_argument("--n_test", type=int, default=20)
parser.add_argument("--timed_steps", type=int, default=50)
parser.add_argument("--out", required=True)
args = parser.parse_args()

RUNS = {
    "without coarse/fine": ("nerf_lego_512", args.base_ckpt),
    "with coarse/fine": ("nerf_lego_512_coarse_fine", args.cf_ckpt),
}
_, _, test_ds = vrm.get_nerf_datasets(dataset_name="lego", image_size=[512, 512])
test_idx = view_indices(test_ds, args.n_test)


def now():
    torch.cuda.synchronize()
    return time.time()


rows, renders = [], {}
for name, (cfg_name, ckpt) in RUNS.items():
    cfg = load_config(cfg_name, **{"renderer.chunk_size": 8192})
    model, _ = load_model(cfg, ckpt)

    # Training-step time: forward + backward on 1024 random rays
    model.train()
    cam = test_ds[0]["camera"].cuda()
    for i in range(args.timed_steps + 5):
        if i == 5:
            t0 = now()
        rays = vrm.get_rays_from_pixels(vrm.get_random_pixels_from_image(1024, (512, 512), cam), (512, 512), cam)
        out = model(rays)
        loss = out["feature"].mean() + (out["feature_coarse"].mean() if "feature_coarse" in out else 0)
        model.zero_grad()
        loss.backward()
    step_ms = (now() - t0) / args.timed_steps * 1000

    model.eval()
    scores, times = [], []
    for k, i in enumerate(test_idx):
        item = test_ds[int(i)]
        t0 = now()
        pred = render(model, item["camera"].cuda(), (512, 512))
        times.append(now() - t0)
        scores.append(psnr(pred, item["image"].cuda()))
        if k == 0:
            renders[name] = pred.clamp(0, 1).cpu().numpy()
            renders["ground truth"] = item["image"].numpy()
    params = sum(p.numel() for p in model.parameters())
    rows.append(f"| {name} | {params / 1e3:.0f}k | {np.mean(scores):.2f} | {step_ms:.1f} | {np.mean(times[1:]):.2f} |")

os.makedirs(args.out, exist_ok=True)
table = "\n".join(["| Model | Parameters | Test PSNR | Training step (ms) | Render 512x512 (s) |",
                   "|---|---|---|---|---|", *rows])
print(table)
with open(os.path.join(args.out, "coarse_fine.md"), "w") as f:
    f.write(table + "\n")
strip = np.concatenate([renders["ground truth"], renders["without coarse/fine"], renders["with coarse/fine"]], 1)
imageio.imwrite(os.path.join(args.out, "q4_2_test_view.png"), (strip * 255).astype(np.uint8))
