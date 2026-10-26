"""Q4.3: compare VolSDF runs with different alpha / beta / initialization.

For each run: test PSNR, mean SDF gradient norm, and the connected pieces of the marching-cubes mesh.
Also saves the same frame of every --grid run's color and geometry GIFs side by side.

Usage:
    python analysis/eval_volsdf.py --out DIR \
        --run NAME ALPHA BETA SPHERE_RADIUS CKPT IMAGE_DIR [--run ...] [--grid NAME ...]
    (SPHERE_RADIUS = none for runs without sphere initialization)
"""
import argparse
import os

import imageio.v2 as imageio
import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from common import load_config, load_model, render, psnr, view_indices, vrm
from render_functions import implicit_to_mesh

parser = argparse.ArgumentParser()
parser.add_argument("--run", nargs=6, action="append", required=True,
                    metavar=("NAME", "ALPHA", "BETA", "SPHERE", "CKPT", "IMAGE_DIR"))
parser.add_argument("--grid", nargs="*", default=None, help="runs to show in the comparison image (default: all)")
parser.add_argument("--frame", type=int, default=3)
parser.add_argument("--n_test", type=int, default=20)
parser.add_argument("--out", required=True)
args = parser.parse_args()

_, _, test_ds = vrm.get_nerf_datasets(dataset_name="lego", image_size=[128, 128])
torch.manual_seed(0)
probe = torch.rand(20000, 3, device="cuda") * 8 - 4          # the eikonal loss's sampling bounds

rows, frames = [], {}
for name, alpha, beta, sphere, ckpt, image_dir in args.run:
    cfg = load_config("volsdf_surface", **{
        "renderer.alpha": float(alpha), "renderer.beta": float(beta),
        "implicit_function.sphere_init_radius": None if sphere.lower() == "none" else float(sphere),
    })
    model, _ = load_model(cfg, ckpt)

    scores = []
    for i in view_indices(test_ds, args.n_test):
        item = test_ds[int(i)]
        scores.append(psnr(render(model, item["camera"].cuda(), (128, 128)), item["image"].cuda()))
    _, grad = model.implicit_fn.get_distance_and_gradient(probe)

    try:
        mesh = implicit_to_mesh(model.implicit_fn, scale=3, device="cuda", thresh=0)
        faces, n_verts = mesh.faces_packed().cpu().numpy(), mesh.verts_packed().shape[0]
    except ValueError:                                       # the SDF never crosses 0: empty mesh
        faces, n_verts = np.zeros((0, 3), int), 0
    if len(faces):
        edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]]])
        adj = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n_verts, n_verts))
        n_pieces, labels = connected_components(adj, directed=False)
        largest = np.bincount(labels[faces[:, 0]]).max() / len(faces) * 100
    else:
        n_pieces, largest = 0, 0.0
    rows.append(f"| {name} | {alpha} | {beta} | {sphere} | {np.mean(scores):.2f} | "
                f"{grad.norm(dim=-1).mean():.3f} | {n_pieces} ({largest:.1f}%) |")

    color = imageio.mimread(os.path.join(image_dir, "part_4_3.gif"))[args.frame][..., :3]
    geometry = imageio.mimread(os.path.join(image_dir, "part_4_3_geometry.gif"))[args.frame][..., :3]
    frames[name] = (color, geometry)

os.makedirs(args.out, exist_ok=True)
table = "\n".join(["| Run | alpha | beta | Sphere init | Test PSNR | Mean SDF gradient norm | "
                   "Mesh pieces (% of faces in largest piece) |", "|---|---|---|---|---|---|---|", *rows])
print(table)
with open(os.path.join(args.out, "volsdf_runs.md"), "w") as f:
    f.write(table + "\n")

grid = args.grid or [r[0] for r in args.run]
image = np.concatenate([np.concatenate([frames[n][0] for n in grid], 1),
                        np.concatenate([frames[n][1] for n in grid], 1)], 0)
imageio.imwrite(os.path.join(args.out, "q4_3_sweep.png"), image.repeat(2, 0).repeat(2, 1))
