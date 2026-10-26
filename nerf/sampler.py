import math
from typing import List

import torch
from ray_utils import RayBundle
from pytorch3d.renderer.cameras import CamerasBase


# Sampler which implements stratified (uniform) point sampling along rays
class StratifiedRaysampler(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth

    def forward(
        self,
        ray_bundle,
    ):
        # TODO (Q1.4): Compute z values for self.n_pts_per_ray points uniformly sampled between [near, far]
        z_vals = torch.linspace(
            self.min_depth, self.max_depth, self.n_pts_per_ray, 
            device=ray_bundle.origins.device
        ).view(1, -1, 1) # (1, n_pts_per_ray, 1)

        # TODO (Q1.4): Sample points from z values
        sample_points = (
            ray_bundle.origins.unsqueeze(1)               # (R, 3) -> (R, 1, 3)
            + z_vals * ray_bundle.directions.unsqueeze(1) # (1, n_pts_per_ray, 1) * (R, 1, 3)
        )                                                 # (R, n_pts_per_ray, 3)

        # Return
        return ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths=z_vals * torch.ones_like(sample_points[..., :1]),
        )


# Q4.2: inverse-transform sampling of depths from the piecewise-constant pdf
# that the coarse weights define over the bins between coarse samples
def sample_pdf(
    bins,                   # (R, M + 1) bin edges along each ray
    weights,                # (R, M) weight of each bin
    n_samples,
    deterministic=False,
):
    weights = weights + 1e-5                                         # keep the pdf non-zero everywhere
    pdf = weights / weights.sum(dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)   # (R, M + 1)

    # Evenly spaced quantiles when rendering, random ones when training
    if deterministic:
        u = torch.linspace(0.0, 1.0, n_samples, device=bins.device).expand(cdf.shape[0], n_samples)
    else:
        u = torch.rand(cdf.shape[0], n_samples, device=bins.device)
    u = u.contiguous()

    # Find the bin each quantile falls in, then interpolate linearly inside it
    idx = torch.searchsorted(cdf, u, right=True)
    below = (idx - 1).clamp(min=0)
    above = idx.clamp(max=cdf.shape[-1] - 1)
    cdf_below, cdf_above = torch.gather(cdf, -1, below), torch.gather(cdf, -1, above)
    bins_below, bins_above = torch.gather(bins, -1, below), torch.gather(bins, -1, above)

    denom = cdf_above - cdf_below
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_below) / denom

    return bins_below + t * (bins_above - bins_below)               # (R, n_samples)


sampler_dict = {
    'stratified': StratifiedRaysampler
}