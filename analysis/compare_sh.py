"""Q7.1: side-by-side RGB comparison of the same render.py frame without and with spherical harmonics.

Each render.py frame is [color | depth | mask]; the color panel is the left third.

Usage:
    python analysis/compare_sh.py --no_sh_frames DIR --sh_frames DIR --views 15 --out DIR
"""
import argparse
import os

import imageio.v2 as imageio
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--no_sh_frames", required=True, help="output/part_5_render/ of render.py --no_sh")
parser.add_argument("--sh_frames", required=True, help="output/part_5_render/ of render.py")
parser.add_argument("--views", type=int, nargs="+", default=[15])
parser.add_argument("--out", required=True)
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)
for v in args.views:
    no_sh = imageio.imread(os.path.join(args.no_sh_frames, f"{v:03d}.png"))[..., :3]
    sh = imageio.imread(os.path.join(args.sh_frames, f"{v:03d}.png"))[..., :3]
    w = no_sh.shape[1] // 3
    gap = np.full((no_sh.shape[0], 8, 3), 255, np.uint8)
    imageio.imwrite(os.path.join(args.out, f"q7_1_cmp_view{v:02d}.png"), np.concatenate([no_sh[:, :w], gap, sh[:, :w]], 1))
    print(f"saved q7_1_cmp_view{v:02d}.png (left: without SH, right: with SH)")
