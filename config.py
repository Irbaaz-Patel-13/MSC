"""
AffordGrasp Configuration
=========================
Central configuration for the AffordGrasp pipeline reproduction.
Covers simulation, VLM reasoning, visual grounding, and grasp generation parameters.

Author: Irbaaz Patel (MSc Robotics & Embedded Systems, Heriot-Watt University)
Reference: Tang et al., "AffordGrasp: In-Context Affordance Reasoning for
           Open-Vocabulary Task-Oriented Grasping in Clutter", IROS 2025
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class SimulationConfig:
    """PyBullet simulation environment parameters."""
    # Physics
    time_step: float = 1.0 / 240.0
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)
    
    # UR5 Robot
    robot_urdf: str = "ur5/ur5.urdf"  # PyBullet data path
    robot_base_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    robot_base_orientation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    end_effector_index: int = 7  # UR5 EE link index
    
    # Gripper
    gripper_type: str = "robotiq_85"  # or "simple_gripper"
    gripper_open_width: float = 0.085  # metres
    gripper_close_width: float = 0.0
    
    # Workspace / Table
    table_position: Tuple[float, float, float] = (0.5, 0.0, 0.0)
    table_size: Tuple[float, float, float] = (0.6, 0.8, 0.02)
    workspace_bounds: dict = field(default_factory=lambda: {
        "x": (0.25, 0.75),
        "y": (-0.3, 0.3),
        "z": (0.02, 0.30),
    })
    
    # Camera (eye-to-hand, top-down view matching paper)
    camera_position: Tuple[float, float, float] = (0.5, 0.0, 0.8)
    camera_target: Tuple[float, float, float] = (0.5, 0.0, 0.0)
    camera_up_vector: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    image_width: int = 640
    image_height: int = 480
    camera_fov: float = 60.0
    camera_near: float = 0.01
    camera_far: float = 2.0
    
    # Clutter settings
    num_distractor_objects: int = 5
    object_categories: List[str] = field(default_factory=lambda: [
        "cup", "spoon", "hammer", "bowl", "screwdriver",
        "scissors", "wine_glass", "knife", "fork", "bottle",
        "mug", "pan", "spatula", "kettle", "racket"
    ])


@dataclass
class VLMConfig:
    """Vision-Language Model configuration for affordance reasoning."""
    model_name: str = "gpt-4o"
    api_key: str = ""  # Set via environment variable OPENAI_API_KEY
    max_tokens: int = 1024
    temperature: float = 0.2
    
    # Prompt templates for three-step reasoning
    system_prompt: str = """You are an expert robotic manipulation assistant. 
You help robots understand how to grasp objects for specific tasks by reasoning 
about object affordances. You analyze scenes and determine which object to grasp 
and which part of that object is optimal for the given task.

Always respond in the exact JSON format requested."""
    
    step1_task_analysis_template: str = """Analyze the following user instruction and extract:
1. The implicit task goal
2. The functional requirements for completing this task
3. The type of tool/object needed

User instruction: "{instruction}"

Respond in JSON format:
{{
    "task_goal": "<what the user wants to accomplish>",
    "functional_requirements": ["<requirement1>", "<requirement2>", ...],
    "required_object_type": "<type of object needed>",
    "required_properties": ["<property1>", "<property2>", ...]
}}"""
    
    step2_object_identification_template: str = """Given the following scene image and task analysis, 
identify the most suitable object for the task.

Task Analysis:
- Task Goal: {task_goal}
- Functional Requirements: {functional_requirements}
- Required Object Type: {required_object_type}

Look at the scene image and identify the best object for this task.

Respond in JSON format:
{{
    "target_object": "<name of the identified object>",
    "object_description": "<brief description of why this object is suitable>",
    "confidence": <0.0-1.0>,
    "approximate_location": "<description of where in the image>"
}}"""
    
    step3_affordance_reasoning_template: str = """Given the identified target object and the task, 
decompose the object into its functional parts and determine the optimal graspable part.

Target Object: {target_object}
Task Goal: {task_goal}
Functional Requirements: {functional_requirements}

Analyze the object's parts and their affordances for the given task.

Respond in JSON format:
{{
    "object_parts": [
        {{
            "part_name": "<name>",
            "function": "<what this part does>",
            "graspable": <true/false>,
            "affordance_for_task": "<how this part relates to the task>"
        }}
    ],
    "optimal_grasp_part": "<name of the best part to grasp>",
    "grasp_reasoning": "<why this part is optimal for grasping given the task>",
    "grasp_approach": "<suggested approach direction: top/side/angled>"
}}"""


@dataclass
class VisualGroundingConfig:
    """LangSAM visual grounding module configuration."""
    # SAM model
    sam_model_type: str = "vit_h"
    sam_checkpoint: str = "sam_vit_h_4b8939.pth"
    
    # Grounding DINO
    grounding_dino_model: str = "groundingdino_swinb_cogcoor"
    box_threshold: float = 0.3
    text_threshold: float = 0.25
    
    # Segmentation refinement
    mask_threshold: float = 0.5
    min_mask_area: int = 100  # minimum pixel area for valid mask
    
    # Image processing
    input_size: Tuple[int, int] = (224, 224)  # Paper uses 224x224 for segmentation
    
    # Device
    device: str = "cuda"  # or "cpu"


@dataclass
class GraspConfig:
    """Grasp generation configuration."""
    # AnyGrasp parameters (or alternative grasp sampler)
    method: str = "antipodal_sampling"  # "anygrasp" requires license
    
    # Antipodal grasp sampling parameters
    num_grasp_samples: int = 200
    friction_coefficient: float = 0.5
    grasp_width_range: Tuple[float, float] = (0.02, 0.085)
    grasp_depth: float = 0.02
    
    # Affordance-guided filtering
    affordance_weight: float = 0.7  # weight for affordance center proximity
    quality_weight: float = 0.3     # weight for grasp quality score
    max_distance_from_affordance_center: float = 0.05  # metres
    
    # Grasp pose parameters
    approach_angle_range: Tuple[float, float] = (-30.0, 30.0)  # degrees from vertical
    num_approach_angles: int = 12
    
    # Post-processing
    collision_check: bool = True
    top_k_grasps: int = 5


@dataclass
class PipelineConfig:
    """Master configuration combining all sub-configs."""
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    visual_grounding: VisualGroundingConfig = field(default_factory=VisualGroundingConfig)
    grasp: GraspConfig = field(default_factory=GraspConfig)
    
    # Output / logging
    output_dir: str = "results"
    save_visualizations: bool = True
    verbose: bool = True
    
    # Experiment
    experiment_name: str = "affordgrasp_reproduction"
    seed: int = 42

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        if not self.vlm.api_key:
            self.vlm.api_key = os.environ.get("OPENAI_API_KEY", "")
