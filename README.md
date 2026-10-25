# 16-825 Assignment 3: Volume Rendering, NeRF, and 3D Gaussian Splatting

The report is `report/index.html`.

## Setup and data

See the Setup section of [handin.md](handin.md).

## Part A: NeRF (run from `nerf/`, outputs in `nerf/images/`)

```bash
mkdir -p images
python volume_rendering_main.py --config-name=box               # Q1
python volume_rendering_main.py --config-name=train_box         # Q2
python volume_rendering_main.py --config-name=nerf_lego         # Q3
python volume_rendering_main.py --config-name=nerf_materials_512                                        # Q4.1
python volume_rendering_main.py --config-name=nerf_materials_512 implicit_function.view_dependence=False  # Q4.1 baseline
python volume_rendering_main.py --config-name=nerf_lego_512                 # Q4.2 without coarse/fine
python volume_rendering_main.py --config-name=nerf_lego_512_coarse_fine     # Q4.2 with coarse/fine
python -m volume_rendering_main --config-name=volsdf_surface renderer.alpha=20   # Q4.3
```

Training resumes from the config's `checkpoint_path` if it exists. `nerf_lego` and `nerf_materials` share
`./checkpoints`, so delete it before switching scenes.

## Part B: Gaussian splatting (run from `GS/`, outputs in `GS/output/`)

```bash
python unit_test_gaussians.py   # Q5.1-5.2 checks
python render.py                # Q5 / Q7.1
python train.py                 # Q6
python train_harder_scene.py    # Q7.2
```

`render.py` uses the spherical harmonics in `chair.ply` (Q7.1). The Q5.5 result uses only the DC colour, from
before spherical harmonics were enabled in `Scene.render`.

If GPU memory runs out, pass e.g. `--gaussians_per_splat 2048`.
