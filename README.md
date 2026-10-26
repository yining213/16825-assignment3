# 16-825 Assignment 3: Volume Rendering, NeRF, and 3D Gaussian Splatting

**Name:** Yi-Ning Huang
**Andrew ID:** yiningh3

Report webpage: http://www.andrew.cmu.edu/course/16-825/projects/yiningh3/proj3/ or https://yining213.github.io/16825-assignment3/

Completed: Q1–Q3, Q4.1 (view dependence), Q4.2 (coarse/fine sampling), Q4.3 (VolSDF), Q5, Q6, Q7.1 (spherical
harmonics) and Q7.2 (harder scene).

## Setup and data

See the Setup section of [handin.md](handin.md) for the environment and the datasets (`nerf/data`, `GS/data` and the
materials scene). The datasets are not included in this folder.

The results were produced with Python 3.10, PyTorch 2.14 (CUDA 13.0), torchvision 0.29 and PyTorch3D 0.7.8 on one
RTX 4090.

## Running the code

`main.py` runs every question and collects all outputs in `results/`.

```bash
python main.py                  # run everything
python main.py --list           # list the available questions
python main.py -q 1 4.2 7.1     # run only the listed questions
python main.py --dry_run        # print the commands main.py would run, without running them
python main.py --retrain        # retrain NeRF models even if their checkpoint exists
```

Each NeRF run saves its checkpoint in `nerf/ckpt/` and resumes from it if it exists. Approximate run times on one
RTX 4090: Q1–Q2 1 min each, Q3 10 min, Q4.1 15 min, Q4.2 30 min, Q4.3 50 min (five runs), Q5 5 min, Q6 5 min,
Q7.1 5 min, Q7.2 25 min. If GPU memory runs out in Part B, add `--gaussians_per_splat 2048` to the GS commands.

The scripts can also be run directly, from `nerf/` or `GS/`, e.g.
`python volume_rendering_main.py --config-name=nerf_lego` or `python render.py`.

## Outputs

| Question | Output in `results/` |
|---|---|
| 1 Volume rendering | `q1/grid.png`, `q1/rays.png`, `q1/sample_points.png`, `q1/part_1.gif`, `q1/depth.png` |
| 2 Optimizing a box | `q2/part_2.gif`; box center and side lengths at the end of `q2/log.txt` |
| 3 NeRF | `q3/part_3.gif` |
| 4.1 View dependence | `q4_1/view_dependence.gif`, `q4_1/no_view_dependence.gif`, `q4_1/view_dependence_psnr.md`, `q4_1/q4_1_test_view.png` |
| 4.2 Coarse/fine | `q4_2/with_coarse_fine.gif`, `q4_2/without_coarse_fine.gif`, `q4_2/coarse_fine.md`, `q4_2/q4_2_test_view.png` |
| 4.3 VolSDF | `q4_3/chosen/part_4_3.gif`, `q4_3/chosen/part_4_3_geometry.gif`, `q4_3/volsdf_runs.md`, `q4_3/q4_3_sweep.png`; one folder per run |
| 5 Gaussian rasterization | `q5/part_5_render.gif`, `q5/unit_tests.txt` |
| 6 Training on the truck | `q6/part_6_training_progress.gif`, `q6/part_6_training_final_renders.gif`; PSNR and SSIM at the end of `q6/log.txt` |
| 7.1 Spherical harmonics | `q7_1/part_7_1_render.gif`, `q7_1/q7_1_cmp_view15.png`, `q7_1/q7_1_cmp_view03.png` (left: without, right: with) |
| 7.2 Harder scene | `q7_2/baseline/` and `q7_2/improved/`: both GIFs each; PSNR and SSIM at the end of each `log.txt` |

## Code

- `nerf/`: the Part A TODOs, plus
  - `sampler.py` `sample_pdf` and `renderer.py` `VolumeCoarseFineRenderer` (Q4.2);
  - view dependence in `NeuralRadianceField` (`view_dependence` config key, Q4.1);
  - sphere initialization in `NeuralSurface` (`sphere_init_radius` config key, Q4.3);
  - configs `nerf_materials_512.yaml`, `nerf_lego_512.yaml`, `nerf_lego_512_coarse_fine.yaml`.
- `GS/`: the Part B TODOs; `render.py --no_sh` renders the DC colour only (Q5) instead of spherical harmonics (Q7.1);
  `train_harder_scene.py --baseline` uses the Q6 setup, without it the improved method (Q7.2).
- `analysis/`: scripts for the report's comparison tables and figures (Q4.1, Q4.2, Q4.3, Q7.1); `main.py` runs them.
