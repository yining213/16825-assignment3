"""Helpers shared by the analysis scripts: import the NeRF code and render / score full images."""
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

warnings.filterwarnings("ignore")
NERF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nerf")
sys.path.insert(0, NERF_DIR)

import volume_rendering_main as vrm  # noqa: E402


def load_config(name, **overrides):
    """Load nerf/configs/<name>.yaml and apply dotted-key overrides, e.g. {"renderer.alpha": 20}."""
    cfg = OmegaConf.load(os.path.join(NERF_DIR, "configs", f"{name}.yaml"))
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value, force_add=True)
    return cfg


def load_model(cfg, checkpoint):
    model = vrm.Model(cfg).cuda().eval()
    ckpt = torch.load(checkpoint, map_location="cuda")
    model.load_state_dict(ckpt["model"])
    return model, ckpt["epoch"]


@torch.no_grad()
def render(model, camera, image_size):
    W, H = image_size
    rays = vrm.get_rays_from_pixels(vrm.get_pixels_from_image((W, H), camera), (W, H), camera)
    return model(rays)["feature"].view(H, W, 3)


def psnr(pred, gt):
    return -10 * torch.log10(F.mse_loss(pred, gt)).item()


def view_indices(dataset, n):
    """n evenly spaced indices into a dataset."""
    return np.linspace(0, len(dataset) - 1, n).astype(int)
