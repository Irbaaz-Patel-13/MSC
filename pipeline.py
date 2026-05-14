"""
AffordGrasp Pipeline
====================
End-to-end pipeline integrating all modules:
  1. Simulation Environment  →  scene + RGB-D capture
  2. Affordance Reasoning    →  task analysis + object ID + part affordance
  3. Visual Grounding        →  pixel-level affordance masks
  4. Grasp Generation        →  task-oriented 6-DoF grasp poses
  5. Grasp Execution         →  robot picks up the object

This is the main entry point for running the AffordGrasp reproduction.

Author: Irbaaz Patel (MSc Robotics, Heriot-Watt University)
"""

import os
import json
import time
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import asdict

from config import PipelineConfig
from sim_env import SimulationEnvironment, CapturedImage
from affordance_reasoning import AffordanceReasoner, FullReasoningResult
from visual_grounding import VisualAffordanceGrounder, GroundingResult
from grasp_generation import AffordanceGraspGenerator, GraspPose


class AffordGraspPipeline:
    """
    Complete AffordGrasp pipeline for open-vocabulary task-oriented grasping.
    
    Reproduces the full pipeline from:
        Tang et al., "AffordGrasp: In-Context Affordance Reasoning for
        Open-Vocabulary Task-Oriented Grasping in Clutter", IROS 2025
    
    Usage:
        config = PipelineConfig()
        pipeline = AffordGraspPipeline(config)
        pipeline.setup()
        
        result = pipeline.run(
            instruction="I want to drink some coffee",
            target_category="mug",
            num_distractors=5
        )
        
        pipeline.cleanup()
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.sim_env = None
        self.reasoner = None
        self.grounder = None
        self.grasp_gen = None
    
    def setup(self, gui: bool = True) -> None:
        """Initialise all pipeline components."""
        print("=" * 70)
        print("  AffordGrasp Pipeline - Initialisation")
        print("=" * 70)
        
        # 1. Simulation Environment
        print("\n[1/4] Setting up simulation environment...")
        self.sim_env = SimulationEnvironment(self.config.simulation, gui=gui)
        self.sim_env.setup()
        
        # 2. VLM Affordance Reasoner
        print("[2/4] Initialising affordance reasoner...")
        self.reasoner = AffordanceReasoner(self.config.vlm)
        
        # 3. Visual Grounder
        print("[3/4] Initialising visual grounder...")
        self.grounder = VisualAffordanceGrounder(self.config.visual_grounding)
        
        # 4. Grasp Generator
        print("[4/4] Initialising grasp generator...")
        self.grasp_gen = AffordanceGraspGenerator(self.config.grasp)
        
        print("\n[OK] Pipeline ready.\n")
    
    def run(
        self,
        instruction: str,
        target_category: Optional[str] = None,
        num_distractors: int = 5,
        execute_grasp: bool = True,
        save_results: bool = True,
        record_video: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute the full AffordGrasp pipeline.

        Args:
            instruction: User's natural language instruction
            target_category: Category to guarantee in the scene (e.g. "mug").
                If None, a random category from config is chosen — the VLM will
                identify the task-relevant object purely from the instruction.
            num_distractors: Number of distractor objects
            execute_grasp: Whether to execute the grasp on the robot
            save_results: Whether to save visualisations and logs

        Returns:
            Dictionary containing all pipeline results
        """
        # When no target is specified, pick a random scene category so the scene
        # is always diverse and the VLM identifies whatever the instruction requires.
        if target_category is None:
            target_category = np.random.choice(
                self.config.simulation.object_categories
            )
            print(f"[Pipeline] No --target given; scene seeded with '{target_category}'")

        results = {
            "instruction": instruction,
            "target_category": target_category,
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "success": False
        }

        trial_dir = os.path.join(
            self.config.output_dir,
            f"trial_{results['timestamp']}_{target_category}"
        )
        os.makedirs(trial_dir, exist_ok=True)

        # =====================================================================
        #  STAGE 1: Scene Setup & RGB-D Capture
        # =====================================================================
        print(f"\n{'='*60}")
        print(f"  STAGE 1: Scene Setup & RGB-D Capture")
        print(f"{'='*60}")

        target_obj, distractors = self.sim_env.spawn_clutter_scene(
            target_category=target_category,
            num_distractors=num_distractors
        )
        results["num_objects"] = 1 + len(distractors)
        
        # Wait for physics to settle
        for _ in range(200):
            import pybullet as p
            p.stepSimulation()
        
        # Capture RGB-D from all configured camera views
        all_views = self.sim_env.capture_multiview_rgbd()
        # Primary (top-down) view used for VLM scene image and saved to disk
        image_data = all_views[0][1]

        if save_results:
            for view_name, view_img in all_views:
                prefix = f"scene_{view_name}"
                self.sim_env.save_rgbd(view_img, trial_dir, prefix=prefix)

        print(f"  Captured {len(all_views)} view(s): {[n for n, _ in all_views]}")
        print(f"  Primary RGB-D shape: {image_data.rgb.shape}")
        
        # =====================================================================
        #  STAGE 2: In-Context Affordance Reasoning (VLM)
        # =====================================================================
        print(f"\n{'='*60}")
        print(f"  STAGE 2: In-Context Affordance Reasoning")
        print(f"{'='*60}")
        
        reasoning_result = self.reasoner.reason(
            instruction=instruction,
            scene_image=image_data.rgb,
            verbose=True,
            target_hint=target_category
        )
        
        results["reasoning"] = {
            "task_goal": reasoning_result.task_analysis.task_goal,
            "target_object": reasoning_result.object_identification.target_object,
            "optimal_part": reasoning_result.affordance_reasoning.optimal_grasp_part,
            "approach": reasoning_result.affordance_reasoning.grasp_approach,
        }
        
        if save_results:
            reasoning_path = os.path.join(trial_dir, "reasoning_result.json")
            with open(reasoning_path, 'w') as f:
                json.dump(results["reasoning"], f, indent=2)
        
        # =====================================================================
        #  STAGE 3: Visual Affordance Grounding
        # =====================================================================
        print(f"\n{'='*60}")
        print(f"  STAGE 3: Visual Affordance Grounding")
        print(f"{'='*60}")

        target_object_name = reasoning_result.object_identification.target_object
        target_part_name = reasoning_result.affordance_reasoning.optimal_grasp_part

        # Semantic object lookup: find the scene object whose PyBullet name
        # contains the VLM's target keywords (e.g. "lid" → ycb_skillet_lid).
        # This gives us the correct body_id independent of visual grounding.
        semantic_body_id: Optional[int] = None
        keywords = target_object_name.lower().split()
        best_kw_score = 0
        for obj in self.sim_env.objects:
            score = sum(1 for kw in keywords if kw in obj.name.lower())
            if score > best_kw_score:
                best_kw_score = score
                semantic_body_id = obj.body_id
        if semantic_body_id is not None and best_kw_score > 0:
            sem_name = next(o.name for o in self.sim_env.objects if o.body_id == semantic_body_id)
            print(f"  Semantic match: '{target_object_name}' → {sem_name} (id={semantic_body_id})")
        else:
            # No keyword match — fall back to the spawned target object
            semantic_body_id = target_obj.body_id
            print(f"  No semantic match for '{target_object_name}'; using spawn target.")

        grounding_result, grounding_image, best_view = self.grounder.ground_multiview(
            views=all_views,
            target_object=target_object_name,
            target_part=target_part_name,
            simulation_segmask=image_data.segmentation,
            target_body_id=semantic_body_id,
        )
        print(f"  Best grounding view: '{best_view}'")
        
        results["grounding"] = {
            "affordance_center_px": list(grounding_result.affordance_center),
            "affordance_center_3d": (
                grounding_result.affordance_center_3d.tolist()
                if grounding_result.affordance_center_3d is not None else None
            ),
            "confidence": grounding_result.confidence,
            "mask_area_px": int(grounding_result.affordance_mask.sum()),
        }
        
        # Use the semantic body_id (from Stage 2 keyword match) as the authoritative
        # grasp target. This is more reliable than pixel-level segmentation matching
        # because it's grounded in the VLM's explicit text output and object names.
        grasp_target_body_id: Optional[int] = semantic_body_id

        print(f"  Object: {target_object_name}")
        print(f"  Part: {target_part_name}")
        print(f"  Affordance centre (px): {grounding_result.affordance_center}")
        print(f"  Affordance centre (3D): {grounding_result.affordance_center_3d}")
        print(f"  Mask area: {grounding_result.affordance_mask.sum()} pixels")

        if save_results:
            vis_path = os.path.join(trial_dir, "grounding_visualisation.png")
            self.grounder.visualize_grounding(
                grounding_image.rgb, grounding_result, save_path=vis_path
            )

        # Free GPU memory before CGN (LangSAM + SAM2 occupy most of VRAM)
        self.grounder.offload_to_cpu()

        # =====================================================================
        #  STAGE 4: Task-Oriented Grasp Generation
        # =====================================================================
        print(f"\n{'='*60}")
        print(f"  STAGE 4: Task-Oriented Grasp Generation")
        print(f"{'='*60}")

        # Use the best-grounding view's depth + camera parameters for CGN
        affordance_points = self.sim_env.depth_to_pointcloud(
            grounding_image, mask=grounding_result.affordance_mask
        )
        full_cloud = self.sim_env.depth_to_pointcloud(grounding_image)

        print(f"  Affordance point cloud: {len(affordance_points)} points")
        print(f"  Full point cloud: {len(full_cloud)} points")

        # Compute 3D affordance centre as the mean of all masked 3D points
        # (more robust than single-pixel back-projection)
        aff_center_3d = self.grounder.compute_affordance_center_3d_from_mask(
            grounding_result.affordance_mask,
            grounding_image.depth,
            grounding_image.camera_intrinsics,
            grounding_image.camera_extrinsics,
        )
        # Fallback: use single-pixel centre or point cloud mean
        if aff_center_3d is None:
            aff_center_3d = grounding_result.affordance_center_3d
        if aff_center_3d is None and len(affordance_points) > 0:
            aff_center_3d = affordance_points.mean(axis=0)
        
        if aff_center_3d is None:
            print("  ERROR: Cannot determine 3D affordance centre.")
            results["grasps"] = []
            return results
        
        print(f"  3D affordance centre: {aff_center_3d}")
        
        # Generate grasps — passes depth/intrinsics for Contact-GraspNet,
        # falls back to antipodal sampling if CGN unavailable
        grasps = self.grasp_gen.generate(
            point_cloud=full_cloud,
            affordance_center_3d=aff_center_3d,
            affordance_points=affordance_points,
            table_height=self.config.simulation.table_position[2],
            depth_image=grounding_image.depth,
            camera_intrinsics=grounding_image.camera_intrinsics,
            camera_extrinsics=grounding_image.camera_extrinsics,
            segmentation_mask=grounding_result.affordance_mask,
        )
        
        results["grasps"] = [
            {
                "position": g.position.tolist(),
                "orientation": g.orientation.tolist(),
                "width": g.width,
                "quality_score": g.quality_score,
                "affordance_score": g.affordance_score,
                "combined_score": g.combined_score,
            }
            for g in grasps
        ]
        
        if save_results and len(grasps) > 0:
            grasp_vis_path = os.path.join(trial_dir, "grasp_visualisation.png")
            self.grasp_gen.visualize_grasps(
                full_cloud, grasps, aff_center_3d, save_path=grasp_vis_path
            )

        # Print detailed grasp table for analysis
        if len(grasps) > 0:
            from scipy.spatial.transform import Rotation as R_util
            print(f"\n  {'#'*56}")
            print(f"  Top-{len(grasps)} task-oriented grasps (AffordGrasp ranking):")
            print(f"  {'#'*56}")
            hdr = (f"  {'#':>3}  {'pos_x':>7} {'pos_y':>7} {'pos_z':>7}  "
                   f"{'roll':>7} {'pitch':>7} {'yaw':>7}  "
                   f"{'width':>6}  {'q_score':>7} {'dist_m':>7} {'ranking':>8}")
            print(hdr)
            print("  " + "-" * 74)
            for idx, g in enumerate(grasps):
                euler = R_util.from_quat(g.orientation).as_euler('xyz', degrees=True)
                print(f"  {idx+1:>3}  "
                      f"{g.position[0]:>7.4f} {g.position[1]:>7.4f} {g.position[2]:>7.4f}  "
                      f"{euler[0]:>7.1f} {euler[1]:>7.1f} {euler[2]:>7.1f}  "
                      f"{g.width:>6.4f}  "
                      f"{g.quality_score:>7.4f} {g.affordance_score:>7.4f} {g.combined_score:>8.4f}")
            print()

        # =====================================================================
        #  STAGE 5: Grasp Execution
        # =====================================================================
        if execute_grasp and len(grasps) > 0:
            print(f"\n{'='*60}")
            print(f"  STAGE 5: Grasp Execution")
            print(f"{'='*60}")

            if record_video:
                # --- Phase 1: Prompt display (wide view + instruction text) ---
                scene_wide = self.sim_env.capture_wide_view()
                self.sim_env.add_annotated_frames(
                    scene_wide.rgb,
                    text_lines=[
                        f"Task:  \"{instruction}\"",
                        f"Object:  {results['reasoning']['target_object']}",
                        f"Grasp:  {results['reasoning']['optimal_part']}",
                    ],
                    hold_frames=60,   # 4 s at 15 fps
                )

                # --- Phase 2: Detection / grounding overlay (45 frames = 3 s) ---
                try:
                    import cv2
                    grounding_vis_path = os.path.join(trial_dir, "grounding_visualisation.png")
                    if os.path.isfile(grounding_vis_path):
                        det_img = cv2.cvtColor(
                            cv2.imread(grounding_vis_path), cv2.COLOR_BGR2RGB
                        )
                    else:
                        det_img = grounding_image.rgb
                    self.sim_env.add_annotated_frames(
                        det_img,
                        text_lines=[
                            "Stage 2: Visual Affordance Grounding",
                            f"Detected '{results['reasoning']['optimal_part']}' "
                            f"on '{results['reasoning']['target_object']}'",
                        ],
                        hold_frames=45,   # 3 s at 15 fps
                    )
                except Exception:
                    pass  # grounding image unavailable — skip this phase

                # --- Phase 3: Live execution (wide-angle, robot + object in frame) ---
                self.sim_env.start_recording()

            best_grasp = grasps[0]
            from scipy.spatial.transform import Rotation as R_util
            euler_best = R_util.from_quat(best_grasp.orientation).as_euler('xyz', degrees=True)
            print(f"  Executing best grasp:")
            print(f"    Position (world): [{best_grasp.position[0]:.4f}, "
                  f"{best_grasp.position[1]:.4f}, {best_grasp.position[2]:.4f}]")
            print(f"    Orientation (RPY°): [{euler_best[0]:.1f}, "
                  f"{euler_best[1]:.1f}, {euler_best[2]:.1f}]")
            print(f"    Quaternion (xyzw): [{best_grasp.orientation[0]:.4f}, "
                  f"{best_grasp.orientation[1]:.4f}, {best_grasp.orientation[2]:.4f}, "
                  f"{best_grasp.orientation[3]:.4f}]")
            print(f"    Gripper width:  {best_grasp.width:.4f} m")
            print(f"    Quality score:  {best_grasp.quality_score:.4f}")
            print(f"    Dist to aff:    {best_grasp.affordance_score:.4f} m")
            print(f"    Ranking score:  {best_grasp.combined_score:.4f}")
            
            success = self.sim_env.execute_grasp(
                grasp_position=best_grasp.position,
                grasp_orientation=best_grasp.orientation,
                pre_grasp_height=0.15,
                target_body_id=grasp_target_body_id,
            )
            
            if record_video:
                self.sim_env.stop_recording()
                video_path = os.path.join(trial_dir, "demo_video.mp4")
                self.sim_env.save_video(video_path)
                print(f"  Demo video saved: {video_path}")

            results["success"] = success
            results["grasp_executed"] = {
                "position": best_grasp.position.tolist(),
                "combined_score": best_grasp.combined_score,
            }
        
        # Save full results
        if save_results:
            results_path = os.path.join(trial_dir, "pipeline_results.json")
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n  Results saved to: {trial_dir}")
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"  PIPELINE SUMMARY")
        print(f"{'='*60}")
        print(f"  Instruction: \"{instruction}\"")
        print(f"  Target: {target_object_name} -> Grasp: {target_part_name}")
        print(f"  Grasps generated: {len(grasps)}")
        print(f"  Grasp success: {'OK' if results['success'] else 'FAIL'}")
        print(f"{'='*60}\n")
        
        return results
    
    def run_benchmark(
        self,
        scenarios: Optional[List[Dict]] = None,
        num_trials_per_scenario: int = 5
    ) -> Dict[str, Any]:
        """
        Run the full benchmark matching the paper's evaluation protocol.
        
        The paper evaluates on these scenarios (Table I, II):
        - Single object grasping: GSR (Grasp Success Rate) and TSR (Task Success Rate)
        - Cluttered grasping: Same metrics with 5-7 distractor objects
        """
        if scenarios is None:
            scenarios = self._default_scenarios()
        
        all_results = []
        
        for scenario in scenarios:
            print(f"\n{'#'*60}")
            print(f"  Scenario: {scenario['instruction']}")
            print(f"{'#'*60}")
            
            scenario_results = {
                "scenario": scenario,
                "trials": [],
                "gsr": 0.0,  # Grasp Success Rate
            }
            
            successes = 0
            for trial in range(num_trials_per_scenario):
                print(f"\n  --- Trial {trial + 1}/{num_trials_per_scenario} ---")
                
                result = self.run(
                    instruction=scenario["instruction"],
                    target_category=scenario["target"],
                    num_distractors=scenario.get("num_distractors", 5),
                    execute_grasp=True,
                    save_results=True
                )
                
                scenario_results["trials"].append(result)
                if result["success"]:
                    successes += 1
                
                self.sim_env.reset()
            
            scenario_results["gsr"] = successes / num_trials_per_scenario
            all_results.append(scenario_results)
            print(f"\n  GSR for '{scenario['instruction']}': "
                  f"{scenario_results['gsr']:.2%}")
        
        # Compute overall metrics
        overall_gsr = np.mean([r["gsr"] for r in all_results])
        
        benchmark_results = {
            "scenarios": all_results,
            "overall_gsr": overall_gsr,
            "num_scenarios": len(scenarios),
            "trials_per_scenario": num_trials_per_scenario,
        }
        
        # Save benchmark results
        bench_path = os.path.join(self.config.output_dir, "benchmark_results.json")
        with open(bench_path, 'w') as f:
            json.dump(benchmark_results, f, indent=2, default=str)
        
        print(f"\n{'='*60}")
        print(f"  BENCHMARK COMPLETE")
        print(f"  Overall GSR: {overall_gsr:.2%}")
        print(f"  Results saved to: {bench_path}")
        print(f"{'='*60}")
        
        return benchmark_results
    
    def _default_scenarios(self) -> List[Dict]:
        """
        Default evaluation scenarios matching the AffordGrasp paper.
        Table II from the paper: 7 object categories with implicit instructions.
        """
        return [
            {"instruction": "I want to drink water.",         "target": "cup",         "num_distractors": 5},
            {"instruction": "I want to drink soup.",          "target": "spoon",       "num_distractors": 5},
            {"instruction": "I want to hammer nails.",        "target": "hammer",      "num_distractors": 5},
            {"instruction": "I want to drink some wine.",     "target": "bowl",        "num_distractors": 5},
            {"instruction": "I want to tighten screws.",      "target": "screwdriver", "num_distractors": 5},
            {"instruction": "I need to hold my food.",        "target": "bowl",        "num_distractors": 5},
            {"instruction": "I need to cut something.",       "target": "scissors",    "num_distractors": 5},
        ]
    
    def cleanup(self) -> None:
        """Shut down all components."""
        if self.sim_env:
            self.sim_env.close()
        print("[Pipeline] Cleaned up.")


# =========================================================================
#  STANDALONE EXECUTION
# =========================================================================

def main():
    """Run a single AffordGrasp trial."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AffordGrasp Pipeline")
    parser.add_argument(
        "--instruction", type=str,
        default="I want to drink some coffee.",
        help="User instruction for the robot"
    )
    parser.add_argument(
        "--target", type=str, default=None,
        help=(
            "Target object category to guarantee in the scene (e.g. 'mug', 'pan'). "
            "If omitted, a random category is chosen and the VLM identifies the "
            "task-relevant object purely from the instruction."
        )
    )
    parser.add_argument(
        "--distractors", type=int, default=5,
        help="Number of distractor objects"
    )
    parser.add_argument(
        "--no-gui", action="store_true",
        help="Run without GUI (headless)"
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run full benchmark evaluation"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--record-video", action="store_true",
        help="Record grasp execution as grasp_video.mp4 in the trial output directory"
    )
    parser.add_argument(
        "--use-sim-segmentation", action="store_true",
        help=(
            "Fall back to PyBullet segmentation masks instead of LangSAM. "
            "Affordance masks will be heuristic, NOT visually grounded. "
            "Use only when LangSAM cannot be installed."
        )
    )
    args = parser.parse_args()

    # Create configuration
    config = PipelineConfig(output_dir=args.output_dir)
    if args.use_sim_segmentation:
        config.visual_grounding.force_sim_fallback = True
    
    # Create and setup pipeline
    pipeline = AffordGraspPipeline(config)
    pipeline.setup(gui=not args.no_gui)
    
    try:
        if args.benchmark:
            pipeline.run_benchmark()
        else:
            result = pipeline.run(
                instruction=args.instruction,
                target_category=args.target,
                num_distractors=args.distractors,
                execute_grasp=True,
                save_results=True,
                record_video=args.record_video,
            )
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    main()