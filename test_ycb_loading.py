"""
test_ycb_loading.py
===================
Smoke-test for the local elpis-lab/YCB_Dataset integration.

Spawns one object per category defined in SimulationConfig.category_to_ycb_id,
captures an overhead RGB image, and saves it to test_ycb_scene.png.
Also prints a per-category load report so you can see which URDFs were found.

Usage:
    python test_ycb_loading.py            # headless (DIRECT mode)
    python test_ycb_loading.py --gui      # with PyBullet GUI window
"""

import argparse
import os
import sys

import numpy as np
import pybullet as p
import pybullet_data
import cv2

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
from config import SimulationConfig, PipelineConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="Open PyBullet GUI window")
    parser.add_argument("--out", default="test_ycb_scene.png", help="Output image path")
    return parser.parse_args()


def main():
    args = parse_args()

    # Build config — __post_init__ resolves ycb_data_path automatically
    cfg = PipelineConfig()
    sim_cfg = cfg.simulation

    print(f"[Test] YCB data path: {sim_cfg.ycb_data_path}")
    print(f"[Test] Path exists: {os.path.isdir(sim_cfg.ycb_data_path)}")

    # Connect to PyBullet
    if args.gui:
        physics_client = p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    else:
        physics_client = p.connect(p.DIRECT)

    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")

    # ---------------------------------------------------------------------------
    # Attempt to load one object per category
    # ---------------------------------------------------------------------------
    categories = list(sim_cfg.category_to_ycb_id.keys())
    results = {}  # category -> "ok" | "missing" | "error: <msg>"

    cols = 4
    spacing = 0.20  # metres between objects
    loaded_ids = []

    for idx, category in enumerate(categories):
        ycb_id = sim_cfg.category_to_ycb_id[category]
        urdf_path = os.path.join(sim_cfg.ycb_data_path, f"{ycb_id}.urdf")

        row = idx // cols
        col = idx % cols
        pos = [col * spacing + 0.15, row * spacing + 0.15, 0.05]

        if not os.path.isfile(urdf_path):
            results[category] = f"missing URDF: {ycb_id}.urdf"
            continue

        try:
            body_id = p.loadURDF(
                urdf_path, pos, [0, 0, 0, 1],
                useFixedBase=False, globalScaling=1.0
            )

            # Texture fallback
            vis_data = p.getVisualShapeData(body_id)
            tex_status = "texture_ok"
            if vis_data and vis_data[0][8] == -1:
                tex_path = os.path.join(sim_cfg.ycb_data_path, ycb_id, "texture_map.png")
                if os.path.isfile(tex_path):
                    try:
                        tex_id = p.loadTexture(tex_path)
                        p.changeVisualShape(body_id, -1, textureUniqueId=tex_id)
                        tex_status = "texture_fallback_ok"
                    except Exception as te:
                        tex_status = f"texture_err: {te}"
                else:
                    tex_status = "no_texture"

            loaded_ids.append(body_id)
            results[category] = f"ok ({tex_status})"

        except Exception as e:
            results[category] = f"error: {e}"

    # Let objects settle a bit
    for _ in range(30):
        p.stepSimulation()

    # ---------------------------------------------------------------------------
    # Capture overhead image
    # ---------------------------------------------------------------------------
    num_cols_used = min(cols, len(categories))
    num_rows_used = (len(categories) + cols - 1) // cols

    cam_x = (num_cols_used * spacing) / 2
    cam_y = (num_rows_used * spacing) / 2
    cam_z = max(num_cols_used, num_rows_used) * spacing * 1.2

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[cam_x, cam_y, cam_z],
        cameraTargetPosition=[cam_x, cam_y, 0],
        cameraUpVector=[0, 1, 0]
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60, aspect=4.0 / 3.0, nearVal=0.01, farVal=10.0
    )

    renderer = p.ER_BULLET_HARDWARE_OPENGL if args.gui else p.ER_TINY_RENDERER
    _, _, rgb_pixels, _, _ = p.getCameraImage(
        width=960, height=720,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=renderer
    )

    rgb = np.array(rgb_pixels, dtype=np.uint8).reshape(720, 960, 4)[:, :, :3]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.out, bgr)
    print(f"[Test] Scene image saved to: {os.path.abspath(args.out)}")

    p.disconnect()

    # ---------------------------------------------------------------------------
    # Print report
    # ---------------------------------------------------------------------------
    print("\n--- YCB Load Report ---")
    ok_count = 0
    for category, status in results.items():
        ycb_id = sim_cfg.category_to_ycb_id[category]
        tag = "[OK]" if status.startswith("ok") else "[FAIL]"
        if status.startswith("ok"):
            ok_count += 1
        print(f"  {tag:6s} {category:15s} -> {ycb_id:25s}  {status}")

    print(f"\n{ok_count}/{len(categories)} categories loaded successfully.")
    return 0 if ok_count == len(categories) else 1


if __name__ == "__main__":
    sys.exit(main())
