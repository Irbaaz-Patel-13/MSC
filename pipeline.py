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

Author: Irbaaz Patel (MSc Robotics & Embedded Systems, Heriot-Watt University)
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
        
        print("\n✓ Pipeline ready.\n")
    
    def run(
        self,
        instruction: str,
        target_category: str,
        num_distractors: int = 5,
        execute_grasp: bool = True,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Execute the full AffordGrasp pipeline.
        
        Args:
            instruction: User's natural language instruction
            target_category: Category of the target object (for scene setup)
            num_distractors: Number of distractor objects
            execute_grasp: Whether to execute the grasp on the robot
            save_results: Whether to save visualisations and logs
        
        Returns:
            Dictionary containing all pipeline results
        """
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
        
        # Capture RGB-D
        image_data = self.sim_env.capture_rgbd()
        
        if save_results:
            self.sim_env.save_rgbd(image_data, trial_dir, prefix="scene")
        
        print(f"  Captured RGB-D: {image_data.rgb.shape}")
        
        # =====================================================================
        #  STAGE 2: In-Context Affordance Reasoning (VLM)
        # =====================================================================
        print(f"\n{'='*60}")
        print(f"  STAGE 2: In-Context Affordance Reasoning")
        print(f"{'='*60}")
        
        reasoning_result = self.reasoner.reason(
            instruction=instruction,
            scene_image=image_data.rgb,
            verbose=True
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
        
        grounding_result = self.grounder.ground(
            rgb_image=image_data.rgb,
            depth_image=image_data.depth,
            target_object=target_object_name,
            target_part=target_part_name,
            camera_intrinsics=image_data.camera_intrinsics,
            camera_extrinsics=image_data.camera_extrinsics,
            simulation_segmask=image_data.segmentation,
            target_body_id=target_obj.body_id
        )
        
        results["grounding"] = {
            "affordance_center_px": list(grounding_result.affordance_center),
            "affordance_center_3d": (
                grounding_result.affordance_center_3d.tolist()
                if grounding_result.affordance_center_3d is not None else None
            ),
            "confidence": grounding_result.confidence,
            "mask_area_px": int(grounding_result.affordance_mask.sum()),
        }
        
        print(f"  Object: {target_object_name}")
        print(f"  Part: {target_part_name}")
        print(f"  Affordance centre (px): {grounding_result.affordance_center}")
        print(f"  Affordance centre (3D): {grounding_result.affordance_center_3d}")
        print(f"  Mask area: {grounding_result.affordance_mask.sum()} pixels")
        
        if save_results:
            vis_path = os.path.join(trial_dir, "grounding_visualisation.png")
            self.grounder.visualize_grounding(
                image_data.rgb, grounding_result, save_path=vis_path
            )
        
        # =====================================================================
        #  STAGE 4: Task-Oriented Grasp Generation
        # =====================================================================
        print(f"\n{'='*60}")
        print(f"  STAGE 4: Task-Oriented Grasp Generation")
        print(f"{'='*60}")
        
        # Generate point cloud from affordance region
        affordance_points = self.sim_env.depth_to_pointcloud(
            image_data, mask=grounding_result.affordance_mask
        )
        full_cloud = self.sim_env.depth_to_pointcloud(image_data)
        
        print(f"  Affordance point cloud: {len(affordance_points)} points")
        print(f"  Full point cloud: {len(full_cloud)} points")
        
        # Use 3D affordance center, or estimate from point cloud
        aff_center_3d = grounding_result.affordance_center_3d
        if aff_center_3d is None and len(affordance_points) > 0:
            aff_center_3d = affordance_points.mean(axis=0)
        
        if aff_center_3d is None:
            print("  ERROR: Cannot determine 3D affordance centre.")
            results["grasps"] = []
            return results
        
        # Generate grasps
        grasps = self.grasp_gen.generate(
            point_cloud=full_cloud,
            affordance_center_3d=aff_center_3d,
            affordance_points=affordance_points,
            table_height=self.config.simulation.table_position[2]
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
        
        # =====================================================================
        #  STAGE 5: Grasp Execution
        # =====================================================================
        if execute_grasp and len(grasps) > 0:
            print(f"\n{'='*60}")
            print(f"  STAGE 5: Grasp Execution")
            print(f"{'='*60}")
            
            best_grasp = grasps[0]
            print(f"  Executing best grasp:")
            print(f"    Position: {best_grasp.position}")
            print(f"    Score: {best_grasp.combined_score:.3f}")
            
            success = self.sim_env.execute_grasp(
                grasp_position=best_grasp.position,
                grasp_orientation=best_grasp.orientation,
                pre_grasp_height=0.15
            )
            
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
        print(f"  Target: {target_object_name} → Grasp: {target_part_name}")
        print(f"  Grasps generated: {len(grasps)}")
        print(f"  Grasp success: {'✓' if results['success'] else '✗'}")
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
        "--target", type=str, default="mug",
        help="Target object category"
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
    args = parser.parse_args()
    
    # Create configuration
    config = PipelineConfig(output_dir=args.output_dir)
    
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
                save_results=True
            )
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    main()
