"""
capture_figure2.py
==================
Standalone script that boots the AffordGrasp PyBullet simulation, spawns a
mug clutter scene, captures multi-view RGB-D images and a full GUI overview,
then assembles a composite figure suitable for a B81EZ portfolio (Figure 2).

Usage:
    python capture_figure2.py

Outputs (all written to portfolio_figures/):
    figure2_view_top.png           – top-down RGB view
    figure2_view_front.png         – front-view RGB (if present)
    figure2_view_side.png          – side-view RGB (if present)
    figure2_scene_overview.png     – 1920×1080 PyBullet GUI screenshot
    figure2_composite.png          – assembled portfolio figure
"""

import os
import sys
import time
import numpy as np

# ── Output directory ───────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "portfolio_figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Ensure repo root is on the path ───────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import SimulationConfig, PipelineConfig
from sim_env import SimulationEnvironment

import pybullet as p
from PIL import Image
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe with PyBullet GUI
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Helpers ────────────────────────────────────────────────────────────────────

def save_rgb(rgb_array: np.ndarray, path: str) -> None:
    """Save a (H, W, 3) uint8 numpy array as a PNG."""
    img = Image.fromarray(rgb_array.astype(np.uint8))
    img.save(path)
    print(f"  Saved: {path}")


def capture_gui_screenshot(width: int = 1920, height: int = 1080) -> np.ndarray:
    """
    Render a 3/4-perspective overview of the scene via p.getCameraImage().

    Uses a viewpoint above and to the left of the robot so the HSR body,
    table, and all objects are visible in a natural isometric-style view.
    Returns an (H, W, 3) uint8 RGB array.
    """
    # 3/4-perspective: camera placed high and to the side, looking at table centre
    eye_pos    = [-0.30, -0.80, 1.10]   # behind-left, elevated
    target_pos = [ 0.50,  0.05, 0.05]   # table surface centre
    up_vec     = [ 0.00,  0.00, 1.00]

    view_mat = p.computeViewMatrix(eye_pos, target_pos, up_vec)
    proj_mat = p.computeProjectionMatrixFOV(
        fov=55.0, aspect=width / height, nearVal=0.05, farVal=6.0
    )

    # ER_BULLET_HARDWARE_OPENGL is GPU-accelerated and fast in GUI mode.
    # Fall back to ER_TINY_RENDERER (software) only when running headless.
    renderer = p.ER_BULLET_HARDWARE_OPENGL
    _, _, rgba, _, _ = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_mat,
        projectionMatrix=proj_mat,
        renderer=renderer,
        shadow=1,
        lightDirection=[1, 1, 3],
        flags=p.ER_NO_SEGMENTATION_MASK,  # skip seg mask — faster
    )

    rgb = np.array(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
    return rgb


def build_composite(
    overview_rgb: np.ndarray,
    view_pairs: list,          # [(name, np.ndarray), ...]
    out_path: str,
) -> None:
    """
    Compose a portfolio figure:
      TOP ROW    – full-width scene overview
      BOTTOM ROW – multi-view captures side by side, labelled

    Saves to out_path.
    """
    n_views = len(view_pairs)

    fig = plt.figure(figsize=(16, 10), facecolor="#1a1a2e")

    gs = gridspec.GridSpec(
        2, n_views,
        figure=fig,
        height_ratios=[1.6, 1.0],
        hspace=0.08,
        wspace=0.04,
        left=0.02, right=0.98,
        top=0.91,  bottom=0.03,
    )

    # ── Top row: overview spanning all columns ─────────────────────────────────
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.imshow(overview_rgb)
    ax_top.set_title(
        "Scene overview — 3/4 perspective (1920 × 1080)",
        color="white", fontsize=11, pad=6,
    )
    ax_top.axis("off")

    # ── Bottom row: per-view captures ──────────────────────────────────────────
    label_map = {
        "top":   "Top-down view",
        "front": "Front view",
        "side":  "Side view",
    }

    for col, (view_name, rgb) in enumerate(view_pairs):
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(rgb)
        ax.set_title(
            label_map.get(view_name, view_name.replace("_", " ").title()),
            color="white", fontsize=10, pad=4,
        )
        ax.axis("off")
        # Thin white border
        for spine in ax.spines.values():
            spine.set_edgecolor("white")
            spine.set_linewidth(0.6)
            spine.set_visible(True)

    # ── Super-title ────────────────────────────────────────────────────────────
    fig.suptitle(
        "PyBullet simulation: Toyota HSR with YCB objects (multi-view capture)",
        color="white", fontsize=13, fontweight="bold", y=0.975,
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n=== AffordGrasp — Figure 2 capture script ===\n")

    # 1. Build config (uses project defaults)
    cfg = PipelineConfig()
    sim_cfg = cfg.simulation

    # 2. Create and set up the simulation environment (GUI enabled)
    print("[1/6] Starting PyBullet simulation (GUI) …")
    env = SimulationEnvironment(sim_cfg, gui=True)
    env.setup()

    # Give the GUI window a moment to render on Windows
    time.sleep(0.5)

    # 3. Spawn clutter scene
    print("[2/6] Spawning clutter scene (target=mug, distractors=5) …")
    target, distractors = env.spawn_clutter_scene(
        target_category="mug",
        num_distractors=5,
    )
    print(f"      target  : {target.name}")
    print(f"      distractors: {[d.name for d in distractors]}")

    # 4. Settle physics
    print("[3/6] Stepping physics (200 steps) …")
    for _ in range(200):
        p.stepSimulation()

    # 5. Capture multi-view RGB-D
    print("[4/6] Capturing multi-view RGB-D …")
    views = env.capture_multiview_rgbd()   # [(name, CapturedImage), ...]
    print(f"      Captured {len(views)} view(s): {[v[0] for v in views]}")

    saved_paths = []

    # View name → output filename mapping
    view_filename_map = {
        "top":   "figure2_view_top.png",
        "front": "figure2_view_front.png",
        "side":  "figure2_view_side.png",
    }
    # Fallback pattern for any unlisted view names
    def view_filename(name: str) -> str:
        return view_filename_map.get(name, f"figure2_view_{name}.png")

    view_rgb_pairs = []
    for view_name, img_data in views:
        fname = view_filename(view_name)
        out_path = os.path.join(OUTPUT_DIR, fname)
        save_rgb(img_data.rgb, out_path)
        saved_paths.append(out_path)
        view_rgb_pairs.append((view_name, img_data.rgb))

    # 6. Capture high-resolution GUI overview
    # Using ER_BULLET_HARDWARE_OPENGL (GPU) at 1920×1080 for fast capture.
    print("[5/6] Capturing 1920×1080 scene overview …")
    overview_rgb = capture_gui_screenshot(width=1920, height=1080)
    overview_path = os.path.join(OUTPUT_DIR, "figure2_scene_overview.png")
    save_rgb(overview_rgb, overview_path)
    saved_paths.append(overview_path)

    # 7. Build composite figure
    print("[6/6] Building composite figure …")
    composite_path = os.path.join(OUTPUT_DIR, "figure2_composite.png")
    build_composite(overview_rgb, view_rgb_pairs, composite_path)
    saved_paths.append(composite_path)

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n=== All files saved ===")
    for p_str in saved_paths:
        print(f"  {p_str}")

    # Keep the GUI open briefly so you can inspect it, then disconnect
    print("\nClosing simulation …")
    time.sleep(2.0)
    env.close() if hasattr(env, "close") else p.disconnect()
    print("Done.\n")


if __name__ == "__main__":
    main()
