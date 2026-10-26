"""
16-825 Assignment 3: Volume Rendering, NeRF, and 3D Gaussian Splatting
Yi-Ning Huang (yiningh3)

Runs every question in the assignment and collects all results in results/.
Each question calls the assignment's own scripts (nerf/volume_rendering_main.py, GS/render.py, GS/train.py,
GS/train_harder_scene.py) with the settings used for the report, then the scripts in analysis/ for the report's
tables and comparison figures.

Usage:
    python main.py                 # run everything
    python main.py -q 1 4.2 7.1    # run only the listed questions
    python main.py --list          # show available questions
    python main.py --dry_run       # print the commands without running them
    python main.py --retrain       # retrain NeRF models even if their checkpoint exists (they resume otherwise)
"""
import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
NERF = os.path.join(ROOT, "nerf")
GS = os.path.join(ROOT, "GS")
RESULTS = os.path.join(ROOT, "results")


def run(cmd, cwd, opts, log=None):
    """Run a python script in cwd, echo its output and append it to log."""
    print(f"$ (cd {os.path.relpath(cwd, ROOT)} && {' '.join(cmd)})", flush=True)
    if opts.dry_run:
        return
    if log:
        os.makedirs(os.path.dirname(log), exist_ok=True)
    proc = subprocess.Popen([sys.executable] + cmd[1:] if cmd[0] == "python" else cmd, cwd=cwd,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    with open(log, "a") if log else open(os.devnull, "w") as f:
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
    if proc.wait() != 0:
        sys.exit(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def collect(src, dst, opts):
    """Copy a file or folder produced by a script into results/."""
    if opts.dry_run:
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy(src, dst)


def out(*parts):
    return os.path.join(RESULTS, *parts)


# --------------------------------------------------------------------------
# Part A: NeRF (run from nerf/, the scripts write to nerf/images/)
# --------------------------------------------------------------------------
def nerf_train(config, ckpt, opts, log, *overrides):
    """Train with its own checkpoint file so different questions never resume from each other."""
    ckpt_path = os.path.join(NERF, "ckpt", ckpt)
    if opts.retrain and os.path.exists(ckpt_path) and not opts.dry_run:
        os.remove(ckpt_path)
    run(["python", "volume_rendering_main.py", f"--config-name={config}",
         f"training.checkpoint_path=./ckpt/{ckpt}", *overrides], NERF, opts, log)
    return ckpt_path


def images(name):
    return os.path.join(NERF, "images", name)


def q1(opts):
    """Differentiable volume rendering (box): grid, rays, sample points, color and depth."""
    os.makedirs(images(""), exist_ok=True)
    run(["python", "volume_rendering_main.py", "--config-name=box"], NERF, opts, out("q1", "log.txt"))
    for name in ("grid.png", "rays.png", "sample_points.png", "depth.png", "part_1.gif"):
        collect(images(name), out("q1", name), opts)


def q2(opts):
    """Optimizing a box: center and side lengths are printed at the end of log.txt."""
    os.makedirs(images(""), exist_ok=True)
    run(["python", "volume_rendering_main.py", "--config-name=train_box"], NERF, opts, out("q2", "log.txt"))
    collect(images("part_2.gif"), out("q2", "part_2.gif"), opts)


def q3(opts):
    """NeRF on lego (128x128)."""
    os.makedirs(images(""), exist_ok=True)
    nerf_train("nerf_lego", "q3_lego.pth", opts, out("q3", "log.txt"))
    collect(images("part_3.gif"), out("q3", "part_3.gif"), opts)


def q4_1(opts):
    """View dependence on materials (512x512), with and without view dependence."""
    os.makedirs(images(""), exist_ok=True)
    vd = nerf_train("nerf_materials_512", "q4_1_vd.pth", opts, out("q4_1", "log_vd.txt"))
    collect(images("part_3.gif"), out("q4_1", "view_dependence.gif"), opts)
    novd = nerf_train("nerf_materials_512", "q4_1_no_vd.pth", opts, out("q4_1", "log_no_vd.txt"),
                      "implicit_function.view_dependence=False")
    collect(images("part_3.gif"), out("q4_1", "no_view_dependence.gif"), opts)
    run(["python", "analysis/eval_view_dependence.py", "--vd_ckpt", vd, "--novd_ckpt", novd,
         "--out", out("q4_1")], ROOT, opts)


def q4_2(opts):
    """Coarse/fine sampling on lego (512x512), with and without."""
    os.makedirs(images(""), exist_ok=True)
    base = nerf_train("nerf_lego_512", "q4_2_base.pth", opts, out("q4_2", "log_base.txt"))
    collect(images("part_3.gif"), out("q4_2", "without_coarse_fine.gif"), opts)
    cf = nerf_train("nerf_lego_512_coarse_fine", "q4_2_coarse_fine.pth", opts, out("q4_2", "log_coarse_fine.txt"))
    collect(images("part_3.gif"), out("q4_2", "with_coarse_fine.gif"), opts)
    run(["python", "analysis/eval_coarse_fine.py", "--base_ckpt", base, "--cf_ckpt", cf, "--out", out("q4_2")],
        ROOT, opts)


VOLSDF_RUNS = [  # name, alpha, beta, sphere-init radius ("null" = off)
    ("chosen", 20, 0.05, "1.0"),
    ("default", 10, 0.05, "1.0"),
    ("low_beta", 10, 0.01, "1.0"),
    ("high_beta", 10, 0.2, "1.0"),
    ("no_sphere_init", 10, 0.05, "null"),
]


def q4_3(opts):
    """VolSDF on lego: the chosen setting plus the alpha / beta / initialization comparison."""
    os.makedirs(images(""), exist_ok=True)
    eval_args = []
    for name, alpha, beta, sphere in VOLSDF_RUNS:
        ckpt = nerf_train("volsdf_surface", f"q4_3_{name}.pth", opts, out("q4_3", name, "log.txt"),
                          f"renderer.alpha={alpha}", f"renderer.beta={beta}",
                          f"implicit_function.sphere_init_radius={sphere}")
        for gif in ("part_4_3.gif", "part_4_3_geometry.gif"):
            collect(images(gif), out("q4_3", name, gif), opts)
        eval_args += ["--run", name, str(alpha), str(beta), "none" if sphere == "null" else sphere, ckpt,
                      out("q4_3", name)]
    run(["python", "analysis/eval_volsdf.py", *eval_args, "--grid", "default", "low_beta", "high_beta", "chosen",
         "--out", out("q4_3")], ROOT, opts)


# --------------------------------------------------------------------------
# Part B: Gaussian splatting (run from GS/, the scripts write to GS/output/<question>/)
# --------------------------------------------------------------------------
def q5(opts):
    """Render the pre-trained chair with the DC colour (also runs the unit tests)."""
    run(["python", "unit_test_gaussians.py"], GS, opts, out("q5", "unit_tests.txt"))
    run(["python", "render.py", "--no_sh", "--out_path", "output/q5"], GS, opts, out("q5", "log.txt"))
    collect(os.path.join(GS, "output", "q5", "part_5_render.gif"), out("q5", "part_5_render.gif"), opts)
    collect(os.path.join(GS, "output", "q5", "part_5_render"), out("q5", "frames"), opts)


def q6(opts):
    """Train isotropic Gaussians on the truck; PSNR / SSIM are at the end of log.txt."""
    run(["python", "train.py", "--out_path", "output/q6"], GS, opts, out("q6", "log.txt"))
    for gif in ("part_6_training_progress.gif", "part_6_training_final_renders.gif"):
        collect(os.path.join(GS, "output", "q6", gif), out("q6", gif), opts)


def q7_1(opts):
    """Render the chair with spherical harmonics and compare with Q5."""
    if not opts.dry_run and not os.path.isdir(out("q5", "frames")):
        q5(opts)
    run(["python", "render.py", "--out_path", "output/q7_1"], GS, opts, out("q7_1", "log.txt"))
    collect(os.path.join(GS, "output", "q7_1", "part_5_render.gif"), out("q7_1", "part_7_1_render.gif"), opts)
    collect(os.path.join(GS, "output", "q7_1", "part_5_render"), out("q7_1", "frames"), opts)
    run(["python", "analysis/compare_sh.py", "--no_sh_frames", out("q5", "frames"), "--sh_frames",
         out("q7_1", "frames"), "--views", "15", "3", "--out", out("q7_1")], ROOT, opts)


def q7_2(opts):
    """Materials with random initialization: the Q6 setup (baseline) and the improved method."""
    for name, flags in (("baseline", ["--baseline"]), ("improved", [])):
        run(["python", "train_harder_scene.py", *flags, "--out_path", f"output/q7_2_{name}"], GS, opts,
            out("q7_2", name, "log.txt"))
        for gif in ("part_7_2_training_progress.gif", "part_7_2_training_final_renders.gif"):
            collect(os.path.join(GS, "output", f"q7_2_{name}", gif), out("q7_2", name, gif), opts)


QUESTIONS = {
    "1": q1, "2": q2, "3": q3, "4.1": q4_1, "4.2": q4_2, "4.3": q4_3,
    "5": q5, "6": q6, "7.1": q7_1, "7.2": q7_2,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-q", "--questions", nargs="+", default=list(QUESTIONS), help="questions to run")
    parser.add_argument("--list", action="store_true", help="list the questions and exit")
    parser.add_argument("--dry_run", action="store_true", help="print the commands without running them")
    parser.add_argument("--retrain", action="store_true", help="delete NeRF checkpoints before training")
    opts = parser.parse_args()

    if opts.list:
        for q, fn in QUESTIONS.items():
            print(f"{q:>4}  {fn.__doc__}")
        return
    unknown = [q for q in opts.questions if q not in QUESTIONS]
    if unknown:
        parser.error(f"unknown question(s) {unknown}; choose from {list(QUESTIONS)}")
    for q in opts.questions:
        print(f"\n===== Question {q}: {QUESTIONS[q].__doc__}", flush=True)
        QUESTIONS[q](opts)
    print(f"\nDone. Results are in {os.path.relpath(RESULTS, os.getcwd())}/")


if __name__ == "__main__":
    main()
