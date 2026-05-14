"""
AffordGrasp HSR Demo Video Generator
======================================
Produces a 4-shot demonstration video for the Toyota HSR:

  Shot 1 -- Prompt display    : wide scene view with instruction text (4 s)
  Shot 2 -- Object detection  : grounding_visualisation.png as a static hold (3 s)
  Shot 3 -- Wide establishing : robot + full tabletop, wider FOV (2 s)
  Shot 4 -- Grasp execution   : side-angled camera, continuous recording
  Shot 5 -- Success frame     : wide shot of robot holding object (2 s)

All text uses ASCII-only HERSHEY fonts (no Unicode em-dashes or arrows).

Usage:
    python make_demo_video.py
    python make_demo_video.py --instruction "Pick up the mug by its handle" --target mug
    python make_demo_video.py --output results/hsr_demo.mp4 --no-gui
"""

import os
import argparse
import tempfile
import numpy as np


# ── pipeline modules ──────────────────────────────────────────────────────────
from config import PipelineConfig
from sim_env import SimulationEnvironment
from affordance_reasoning import AffordanceReasoner
from visual_grounding import VisualAffordanceGrounder
from grasp_generation import AffordanceGraspGenerator


# ── helpers ───────────────────────────────────────────────────────────────────

def _draw_text(frame: np.ndarray, lines: list) -> np.ndarray:
    """
    Return a copy of *frame* with a dark banner and white text lines.
    Uses HERSHEY_SIMPLEX (ASCII-only) -- no Unicode characters in *lines*.
    """
    try:
        import cv2
    except ImportError:
        return frame.copy()

    out = frame.copy()
    _, w = out.shape[:2]
    line_h = 38
    banner_h = line_h * len(lines) + 16
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.72, out, 0.28, 0, out)

    font = cv2.FONT_HERSHEY_SIMPLEX
    y = line_h - 2
    for line in lines:
        # Drop-shadow for legibility on any background
        cv2.putText(out, line, (14, y + 2), font, 0.70,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, line, (14, y), font, 0.70,
                    (255, 255, 255), 2, cv2.LINE_AA)
        y += line_h
    return out


def _hold(frames: list, rgb: np.ndarray, n: int) -> None:
    """Append *n* copies of *rgb* to *frames*."""
    frames.extend([rgb] * n)


def _load_png_rgb(path: str) -> np.ndarray:
    """Load a PNG from disk and return it as an RGB uint8 array."""
    import cv2
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _resize_to(frame: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize *frame* to (h, w) using area interpolation."""
    import cv2
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)


def _save_video(frames: list, path: str, fps: int = 15) -> None:
    try:
        import cv2
    except ImportError:
        print("[Demo] OpenCV not available -- cannot save video.")
        return
    if not frames:
        print("[Demo] No frames collected.")
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"[Demo] Saved {len(frames)} frames @ {fps} fps -> {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def make_demo_video(
    instruction: str = "Pick up the mug by its handle",
    target: str = "mug",
    num_distractors: int = 4,
    output_path: str = "results/hsr_demo.mp4",
    gui: bool = True,
    fps: int = 15,
) -> None:

    # ── init ──────────────────────────────────────────────────────────────────
    config    = PipelineConfig()
    sim       = SimulationEnvironment(config.simulation, gui=gui)
    reasoner  = AffordanceReasoner(config.vlm)
    grounder  = VisualAffordanceGrounder(config.visual_grounding)
    grasp_gen = AffordanceGraspGenerator(config.grasp)

    sim.setup()
    frames: list = []

    # Temp directory used for the grounding visualisation PNG (Shot 2)
    tmp_dir = tempfile.mkdtemp(prefix="affordgrasp_demo_")

    try:
        # ── scene ─────────────────────────────────────────────────────────────
        print("\n[Demo] Spawning clutter scene...")
        target_obj, _ = sim.spawn_clutter_scene(
            target_category=target,
            num_distractors=num_distractors,
        )
        import pybullet as p
        for _ in range(240):
            p.stepSimulation()

        # ── PIPELINE STAGE 1: affordance reasoning ────────────────────────────
        print("[Demo] Running affordance reasoning...")
        all_views   = sim.capture_multiview_rgbd()
        primary_img = all_views[0][1]

        reasoning = reasoner.reason(
            instruction=instruction,
            scene_image=primary_img.rgb,
            verbose=True,
            target_hint=target,
        )
        obj_name  = reasoning.object_identification.target_object
        part_name = reasoning.affordance_reasoning.optimal_grasp_part
        approach  = reasoning.affordance_reasoning.grasp_approach
        print(f"[Demo] Reasoning -> object: {obj_name}, part: {part_name}")

        # ── PIPELINE STAGE 2: visual grounding ────────────────────────────────
        print("[Demo] Running visual grounding...")
        keywords = obj_name.lower().split()
        semantic_body_id = target_obj.body_id
        best_score = 0
        for obj in sim.objects:
            score = sum(1 for kw in keywords if kw in obj.name.lower())
            if score > best_score:
                best_score = score
                semantic_body_id = obj.body_id

        grounding_result, grounding_image, best_view = grounder.ground_multiview(
            views=all_views,
            target_object=obj_name,
            target_part=part_name,
            simulation_segmask=primary_img.segmentation,
            target_body_id=semantic_body_id,
        )
        print(f"[Demo] Grounding best view: '{best_view}'")

        # Save the grounding visualisation to disk (used as Shot 2 frame)
        grounding_vis_path = os.path.join(tmp_dir, "grounding_vis.png")
        grounder.visualize_grounding(
            grounding_image.rgb, grounding_result, save_path=grounding_vis_path
        )

        # ── PIPELINE STAGE 3: grasp generation ───────────────────────────────
        print("[Demo] Generating grasps...")
        grounder.offload_to_cpu()

        aff_center_3d = grounder.compute_affordance_center_3d_from_mask(
            grounding_result.affordance_mask,
            grounding_image.depth,
            grounding_image.camera_intrinsics,
            grounding_image.camera_extrinsics,
        )
        if aff_center_3d is None:
            aff_center_3d = grounding_result.affordance_center_3d

        affordance_pts = sim.depth_to_pointcloud(
            grounding_image, mask=grounding_result.affordance_mask
        )
        full_cloud = sim.depth_to_pointcloud(grounding_image)

        if aff_center_3d is None and len(affordance_pts) > 0:
            aff_center_3d = affordance_pts.mean(axis=0)

        if aff_center_3d is None:
            print("[Demo] ERROR: No 3D affordance centre -- aborting.")
            return

        grasps = grasp_gen.generate(
            point_cloud=full_cloud,
            affordance_center_3d=aff_center_3d,
            affordance_points=affordance_pts,
            table_height=config.simulation.table_position[2],
            depth_image=grounding_image.depth,
            camera_intrinsics=grounding_image.camera_intrinsics,
            camera_extrinsics=grounding_image.camera_extrinsics,
            segmentation_mask=grounding_result.affordance_mask,
        )
        print(f"[Demo] Generated {len(grasps)} grasps.")

        # Reference frame size from the wide view (all shots use same resolution)
        ref_wide = sim.capture_wide_view()
        vid_h, vid_w = ref_wide.rgb.shape[:2]

        # =====================================================================
        #  VIDEO ASSEMBLY
        # =====================================================================

        # -- SHOT 1: Prompt display -------------------------------------------
        # Wide scene view (full robot + table), instruction text banner, 4 s.
        print("\n[Demo] Shot 1 -- prompt display...")
        shot1 = _draw_text(ref_wide.rgb, [
            f'Instruction: "{instruction}"',
            f"Target: {obj_name}",
            f"Grasp part: {part_name}  ({approach})",
        ])
        _hold(frames, shot1, n=fps * 4)

        # -- SHOT 2: Object detection / grounding -----------------------------
        # Use the grounding_visualisation.png (LangSAM mask overlay on the best
        # RGB view) as a static held frame -- much clearer than any sim render.
        print("[Demo] Shot 2 -- object detection...")
        try:
            det_rgb = _load_png_rgb(grounding_vis_path)
            # Resize to match video dimensions if the grounding image differs
            if det_rgb.shape[:2] != (vid_h, vid_w):
                det_rgb = _resize_to(det_rgb, vid_h, vid_w)
        except Exception as e:
            print(f"  [warn] Could not load grounding vis ({e}); using sim render.")
            det_rgb = grounding_image.rgb
            if det_rgb.shape[:2] != (vid_h, vid_w):
                det_rgb = _resize_to(det_rgb, vid_h, vid_w)

        shot2 = _draw_text(det_rgb, [
            "Stage 2: Visual Affordance Grounding",
            f"Detected '{part_name}' on '{obj_name}'",
            f"Mask: {int(grounding_result.affordance_mask.sum())} px  "
            f"Confidence: {grounding_result.confidence:.2f}",
        ])
        _hold(frames, shot2, n=fps * 3)

        # -- SHOT 3: Wide establishing shot -----------------------------------
        # Capture again (robot may have settled) with wide FOV, 2 s hold.
        print("[Demo] Shot 3 -- wide establishing shot...")
        wide2 = sim.capture_wide_view()
        shot3 = _draw_text(wide2.rgb, [
            "Stage 3: Spatial Context",
            "Toyota HSR  +  cluttered tabletop scene",
        ])
        _hold(frames, shot3, n=fps * 2)

        # -- SHOT 4: Grasp execution ------------------------------------------
        # Side-angled camera slightly above table height so the gripper
        # approach, contact, closing, and lift are all clearly visible.
        if grasps:
            print("[Demo] Shot 4 -- grasp execution...")
            sim.start_recording(
                every_n=8,                      # ~30 fps capture at 240 Hz
                eye=sim.EXEC_EYE,
                target=sim.EXEC_TARGET,
                up=sim.EXEC_UP,
                fov=sim.EXEC_FOV,
            )

            success = sim.execute_grasp(
                grasp_position=grasps[0].position,
                grasp_orientation=grasps[0].orientation,
                pre_grasp_height=0.15,
                target_body_id=semantic_body_id,
            )

            sim.stop_recording()

            # Annotate each execution frame (ASCII only -- no em-dashes)
            for f in sim._video_frames:
                frames.append(_draw_text(f, [
                    "Stage 4: Grasp Execution",
                    f"Grasping '{obj_name}' at '{part_name}'",
                ]))
            sim._video_frames.clear()

            # -- SHOT 5: Success frame ----------------------------------------
            # Capture the wide view while the robot is still holding the object.
            print("[Demo] Shot 5 -- success frame...")
            wide_end = sim.capture_wide_view()
            if success:
                label = f"Grasp Successful -- {target} lifted by {part_name}"
                color_overlay = None   # no tint; green banner is enough
            else:
                label = "Grasp attempted -- check IK / affordance region"
                color_overlay = None

            shot5 = _draw_text(wide_end.rgb, [
                "Stage 5: Result",
                label,
            ])

            # Draw a green border on success to make it pop
            if success:
                import cv2
                h5, w5 = shot5.shape[:2]
                thick = max(6, h5 // 60)
                cv2.rectangle(shot5, (thick, thick), (w5 - thick, h5 - thick),
                              (50, 220, 80), thick)

            _hold(frames, shot5, n=fps * 3)

        else:
            print("[Demo] No grasps generated -- skipping execution shots.")

        # -- save -------------------------------------------------------------
        _save_video(frames, output_path, fps=fps)

    finally:
        sim.close()
        # Clean up temp dir
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print("[Demo] Done.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AffordGrasp HSR demo video")
    parser.add_argument("--instruction", default="Pick up the mug by its handle",
                        help="Task instruction shown in Shot 1")
    parser.add_argument("--target", default="mug",
                        help="Target object category (e.g. mug, hammer)")
    parser.add_argument("--distractors", type=int, default=4,
                        help="Number of distractor objects on the table")
    parser.add_argument("--output", default="results/hsr_demo.mp4",
                        help="Output video path")
    parser.add_argument("--no-gui", action="store_true",
                        help="Run headless (no PyBullet GUI window)")
    parser.add_argument("--fps", type=int, default=15,
                        help="Output video frame rate")
    args = parser.parse_args()

    make_demo_video(
        instruction=args.instruction,
        target=args.target,
        num_distractors=args.distractors,
        output_path=args.output,
        gui=not args.no_gui,
        fps=args.fps,
    )
