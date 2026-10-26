import torch

from typing import List, Optional, Tuple
from pytorch3d.renderer.cameras import CamerasBase

from ray_utils import RayBundle
from sampler import sample_pdf


# Volume renderer which integrates color and density along rays
# according to the equations defined in [Mildenhall et al. 2020]
class VolumeRenderer(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self._chunk_size = cfg.chunk_size
        self._white_background = cfg.white_background if 'white_background' in cfg else False

    def _compute_weights(
        self,
        deltas,
        rays_density: torch.Tensor,
        eps: float = 1e-10
    ):
        # TODO (1.5): Compute transmittance using the equation described in the README
        alpha = 1 - torch.exp(-rays_density * deltas) # (R, n_pts_per_ray, 1)
        T = torch.cumprod(1 - alpha + eps, dim=1) # (R, n_pts_per_ray, 1)
        
        # shift: T_1 = 1, T_i = ∏_{j<i}, transmittance of the last one doesn't matter
        T = torch.cat([torch.ones_like(T[:, :1]), T[:, :-1]], dim=1) 

        # TODO (1.5): Compute weight used for rendering from transmittance and alpha
        weights = T * alpha
        
        return weights
    
    def _aggregate(
        self,
        weights: torch.Tensor,
        rays_feature: torch.Tensor
    ):
        # TODO (1.5): Aggregate (weighted sum of) features using weights
        feature = torch.sum(weights * rays_feature, dim=1) # (R, n_pts, 1) * (R, n_pts, C) -> (R, C)

        return feature

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
    ):
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]

            # Sample points along the ray
            cur_ray_bundle = sampler(cur_ray_bundle)
            n_pts = cur_ray_bundle.sample_shape[1]

            # Call implicit function with sample points
            implicit_output = implicit_fn(cur_ray_bundle)
            density = implicit_output['density']
            feature = implicit_output['feature']

            # Compute length of each ray segment
            depth_values = cur_ray_bundle.sample_lengths[..., 0]
            deltas = torch.cat(
                (
                    depth_values[..., 1:] - depth_values[..., :-1],
                    1e10 * torch.ones_like(depth_values[..., :1]),
                ),
                dim=-1,
            )[..., None]

            # Compute aggregation weights
            weights = self._compute_weights(
                deltas.view(-1, n_pts, 1),
                density.view(-1, n_pts, 1)
            ) 

            # TODO (1.5): Render (color) features using weights
            feature = self._aggregate(weights, feature.view(-1, n_pts, feature.shape[-1]))

            # TODO (1.5): Render depth map
            depth = self._aggregate(weights, depth_values.view(-1, n_pts, 1))

            # Return
            cur_out = {
                'feature': feature,
                'depth': depth,
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


def sdf_to_density(signed_distance, alpha, beta):
    # TODO (Q4.3): Convert signed distance to density with alpha, beta parameters
    # VolSDF Eq. 2-3: sigma = alpha * Psi_beta(-d), Psi_beta = Laplace CDF.
    # Each branch is clamped so the branch torch.where discards can't overflow (inf * 0 = NaN grads).
    d = signed_distance
    return alpha * torch.where(
        d > 0,
        0.5 * torch.exp(-d.clamp(min=0) / beta),        # outside: decays to 0
        1 - 0.5 * torch.exp(d.clamp(max=0) / beta),     # inside: rises to 1
    )

class VolumeSDFRenderer(VolumeRenderer):
    def __init__(
        self,
        cfg
    ):
        super().__init__(cfg)

        self._chunk_size = cfg.chunk_size
        self._white_background = cfg.white_background if 'white_background' in cfg else False
        self.alpha = cfg.alpha
        self.beta = cfg.beta

        self.cfg = cfg

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
        light_dir=None
    ):
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]

            # Sample points along the ray
            cur_ray_bundle = sampler(cur_ray_bundle)
            n_pts = cur_ray_bundle.sample_shape[1]

            # Call implicit function with sample points
            distance, color = implicit_fn.get_distance_color(cur_ray_bundle.sample_points)
            density = sdf_to_density(distance, self.alpha, self.beta) # TODO (Q4.3): convert SDF to density

            # Compute length of each ray segment
            depth_values = cur_ray_bundle.sample_lengths[..., 0]
            deltas = torch.cat(
                (
                    depth_values[..., 1:] - depth_values[..., :-1],
                    1e10 * torch.ones_like(depth_values[..., :1]),
                ),
                dim=-1,
            )[..., None]

            # Compute aggregation weights
            weights = self._compute_weights(
                deltas.view(-1, n_pts, 1),
                density.view(-1, n_pts, 1)
            ) 

            geometry_color = torch.zeros_like(color)

            # Compute color
            color = self._aggregate(
                weights,
                color.view(-1, n_pts, color.shape[-1])
            )

            # Return
            cur_out = {
                'feature': color,
                "geometry": geometry_color
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


# Q4.2: hierarchical sampling. A coarse network renders the stratified samples;
# its weights decide where a fine network gets extra samples along each ray.
class VolumeCoarseFineRenderer(VolumeRenderer):
    def __init__(
        self,
        cfg
    ):
        super().__init__(cfg)

        self._n_pts_fine = cfg.n_pts_fine

    def _render(self, implicit_fn, ray_bundle):
        n_pts = ray_bundle.sample_shape[1]
        implicit_output = implicit_fn(ray_bundle)
        feature = implicit_output['feature']

        depth_values = ray_bundle.sample_lengths[..., 0]
        deltas = torch.cat(
            (
                depth_values[..., 1:] - depth_values[..., :-1],
                1e10 * torch.ones_like(depth_values[..., :1]),
            ),
            dim=-1,
        )[..., None]

        weights = self._compute_weights(
            deltas.view(-1, n_pts, 1),
            implicit_output['density'].view(-1, n_pts, 1)
        )
        feature = self._aggregate(weights, feature.view(-1, n_pts, feature.shape[-1]))
        depth = self._aggregate(weights, depth_values.view(-1, n_pts, 1))

        return feature, depth, weights

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
    ):
        implicit_fn_coarse, implicit_fn_fine = implicit_fn
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]

            # Coarse pass: stratified samples through the coarse network
            cur_ray_bundle = sampler(cur_ray_bundle)
            feature_coarse, _, weights_coarse = self._render(implicit_fn_coarse, cur_ray_bundle)

            # Fine pass: draw extra depths where the coarse weights are high,
            # merge them with the coarse depths, and render with the fine network
            z_coarse = cur_ray_bundle.sample_lengths[..., 0]                 # (R, n_coarse)
            with torch.no_grad():
                z_mid = 0.5 * (z_coarse[..., 1:] + z_coarse[..., :-1])       # bin edges between coarse samples
                z_fine = sample_pdf(
                    z_mid,
                    weights_coarse[:, 1:-1, 0],
                    self._n_pts_fine,
                    deterministic=not (self.training and torch.is_grad_enabled()),
                )
            z_all, _ = torch.sort(torch.cat([z_coarse, z_fine], dim=-1), dim=-1)  # (R, n_coarse + n_fine)

            fine_ray_bundle = RayBundle(
                cur_ray_bundle.origins,
                cur_ray_bundle.directions,
                cur_ray_bundle.origins[:, None, :] + z_all[..., None] * cur_ray_bundle.directions[:, None, :],
                z_all[..., None],
            )
            feature, depth, _ = self._render(implicit_fn_fine, fine_ray_bundle)

            # Return
            cur_out = {
                'feature': feature,
                'depth': depth,
                'feature_coarse': feature_coarse,
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


renderer_dict = {
    'volume': VolumeRenderer,
    'volume_sdf': VolumeSDFRenderer,
    'volume_coarse_fine': VolumeCoarseFineRenderer,
}
