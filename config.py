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
    
    # UR5 Robot — use pybullet_ur5_robotiq URDF for UR5 + Robotiq-85
    robot_urdf: str = "ur5/ur5.urdf"  # PyBullet data path (fallback)
    robot_base_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    robot_base_orientation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    end_effector_index: int = 7  # UR5 EE link (verify against your URDF)
    # Home joint configuration for UR5
    home_joint_positions: Tuple = (-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0)
    
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
    
    # Camera (eye-to-hand, matching paper's RealSense L515 setup)
    camera_position: Tuple[float, float, float] = (0.5, 0.0, 1.2)
    camera_target: Tuple[float, float, float] = (0.5, 0.0, 0.0)
    camera_up_vector: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    image_width: int = 640
    image_height: int = 480
    camera_fov: float = 60.0
    camera_near: float = 0.01
    camera_far: float = 5.0  # extended for wider depth range
    
    # Object loading: use YCB meshes when available, primitives as fallback
    use_ycb_objects: bool = True
    ycb_data_path: str = ""  # auto-detected from pybullet_object_models if installed
    
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
    model_name: str = "gpt-4o-2024-08-06"  # supports Structured Outputs
    api_key: str = ""  # Set via environment variable OPENAI_API_KEY
    max_tokens: int = 512
    temperature: float = 0.3
    
    # Structured Outputs — uses Pydantic models for guaranteed JSON schema
    use_structured_outputs: bool = True
    
    # Robust consensus querying — query VLM n times, take majority vote
    n_consensus_samples: int = 3
    
    # Image detail level: "high" (~850 tokens, better accuracy) or
    # "low" (85 tokens, ~$0.001/call vs ~$0.004/call)
    image_detail: str = "high"
    
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
    """LangSAM visual grounding module configuration (v0.2.x API)."""
    # LangSAM model variant — sam2.1 variants: "sam2.1_hiera_tiny" (4GB),
    # "sam2.1_hiera_small" (default, ~6GB), "sam2.1_hiera_large" (~10GB)
    sam_variant: str = "sam2.1_hiera_small"
    
    # GroundingDINO detection thresholds
    box_threshold: float = 0.3
    text_threshold: float = 0.25
    
    # Two-pass grounding: crop padding around detected object bbox (pixels)
    crop_padding: int = 20
    
    # Segmentation refinement
    min_mask_area: int = 100  # minimum pixel area for valid mask
    
    # Image processing — paper uses 224x224 for affordance segmentation,
    # but LangSAM works best at native resolution (no resize needed)
    resize_for_segmentation: bool = False
    
    # Device
    device: str = "cuda"  # or "cpu"


@dataclass
class GraspConfig:
    """Grasp generation configuration."""
    # Method: "contact_graspnet" (recommended open-source) or "antipodal" (fallback)
    method: str = "contact_graspnet"
    
    # Contact-GraspNet configuration (PyTorch port by elchun)
    cgn_checkpoint_dir: str = "checkpoints/contact_graspnet"
    cgn_forward_passes: int = 5   # number of stochastic forward passes
    cgn_z_range: Tuple[float, float] = (0.2, 1.8)  # depth range filter (metres)
    cgn_local_regions: bool = True  # per-segment grasp generation
    cgn_filter_grasps: bool = True  # collision filtering
    
    # Antipodal sampling fallback parameters
    num_grasp_samples: int = 200
    friction_coefficient: float = 0.5
    grasp_width_range: Tuple[float, float] = (0.02, 0.085)
    grasp_depth: float = 0.02
    approach_angle_range: Tuple[float, float] = (-30.0, 30.0)  # degrees from vertical
    num_approach_angles: int = 12
    
    # Affordance-guided ranking — the paper's core formula:
    #   ranking = grasp_score / distance_to_affordance_centre
    # This replaces the weighted-sum approach with the actual AffordGrasp formula.
    min_distance_epsilon: float = 1e-6  # avoids division by zero
    max_distance_from_affordance_center: float = 0.10  # metres, hard cutoff
    
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