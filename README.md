# AffordGrasp: Reproduction & Execution Strategy

**MSc Robotics — Heriot-Watt University**  
**Author:** Irbaaz Patel  
**Reference:** Tang et al., "AffordGrasp: In-Context Affordance Reasoning for Open-Vocabulary Task-Oriented Grasping in Clutter", IROS 2025  
**Paper:** [arXiv:2503.00778](https://arxiv.org/abs/2503.00778) | **Project:** [eqcy.github.io/affordgrasp](https://eqcy.github.io/affordgrasp/)

---

## Overview

This repository contains a full reproduction of the AffordGrasp pipeline for open-vocabulary task-oriented grasping in cluttered scenes. The system takes a natural language instruction (e.g., "I want to drink coffee") and an RGB-D image of a cluttered scene, then reasons about which object to grasp and where to grasp it based on affordances.

### Pipeline Architecture

```
User Instruction ──┐
                    ├──► [Stage 1] VLM Affordance Reasoning (GPT-4o)
Scene RGB-D ───────┘         │
                             ├── Step 1: Task Analysis
                             ├── Step 2: Object Identification  
                             └── Step 3: Part & Affordance Reasoning
                                         │
                                         ▼
                        [Stage 2] Visual Affordance Grounding (LangSAM)
                              │
                              ├── Object Segmentation (GroundingDINO + SAM)
                              └── Part-Level Affordance Mask
                                         │
                                         ▼
                        [Stage 3] Task-Oriented Grasp Generation (AnyGrasp*)
                              │
                              ├── Candidate Grasp Sampling on Point Cloud
                              ├── Affordance-Guided Grasp Ranking
                              └── Optimal 6-DoF Grasp Pose Selection
                                         │
                                         ▼
                        [Stage 4] Grasp Execution (UR5 + Gripper)
```

*\*AnyGrasp requires a commercial license. This implementation provides an antipodal sampling alternative.*

---

## Project Structure

```
affordgrasp_project/
├── pipeline.py                    # Main entry point (end-to-end pipeline)
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── modules/
│   ├── __init__.py
│   ├── config.py                  # All configuration parameters
│   ├── sim_env.py                 # PyBullet simulation environment
│   ├── affordance_reasoning.py    # VLM three-step reasoning (GPT-4o)
│   ├── visual_grounding.py        # LangSAM segmentation module
│   └── grasp_generation.py        # Affordance-guided grasp generation
├── results/                       # Output directory for trial results
├── assets/                        # Object meshes and URDFs (optional)
└── docs/                          # Documentation
```

---

## Installation

### Step 1: System Requirements

- Ubuntu 20.04+ or macOS
- Python 3.9+
- NVIDIA GPU with CUDA 11.7+ (for LangSAM; CPU mode available)
- At least 8 GB RAM (16 GB recommended)

### Step 2: Create Environment

```bash
# Create and activate virtual environment
python -m venv affordgrasp_env
source affordgrasp_env/bin/activate

# Install core dependencies
pip install pybullet numpy scipy opencv-python matplotlib Pillow open3d
pip install openai torch torchvision transforms3d trimesh pyyaml tqdm
```

### Step 3: Install LangSAM (Optional - for full visual grounding)

```bash
# Requires PyTorch with CUDA
pip install lang-sam
```

If LangSAM is not installed, the pipeline automatically falls back to PyBullet's segmentation masks (perfect in simulation).

### Step 4: Set OpenAI API Key (Optional - for real VLM reasoning)

```bash
export OPENAI_API_KEY="sk-your-key-here"
```

Without an API key, the reasoner runs in **mock mode** with a built-in affordance knowledge base — useful for testing the full pipeline.

---

## Usage

### Quick Start: Single Trial

```bash
python pipeline.py \
    --instruction "I want to drink some coffee." \
    --target mug \
    --distractors 5
```

### Headless Mode (No GUI)

```bash
python pipeline.py \
    --instruction "I want to scoop something." \
    --target spoon \
    --no-gui
```

### Full Benchmark

```bash
python pipeline.py --benchmark
```

### Python API

```python
from modules.config import PipelineConfig
from pipeline import AffordGraspPipeline

config = PipelineConfig()
pipeline = AffordGraspPipeline(config)
pipeline.setup(gui=True)

result = pipeline.run(
    instruction="I want to hammer nails.",
    target_category="hammer",
    num_distractors=5
)

print(f"Success: {result['success']}")
print(f"Grasp Part: {result['reasoning']['optimal_part']}")

pipeline.cleanup()
```

---

## Module Documentation

### 1. Simulation Environment (`sim_env.py`)

Reproduces the PyBullet simulation from the paper: UR5 robot arm on a tabletop with cluttered objects and eye-to-hand RGB-D camera.

Key features:
- Randomised clutter scene generation with collision-free placement
- RGB-D capture with intrinsic/extrinsic calibration matrices
- Point cloud generation from depth images
- IK-based robot motion and simplified grasp execution

### 2. Affordance Reasoning (`affordance_reasoning.py`)

Implements the three-step in-context affordance reasoning using GPT-4o:

- **Step 1 — Task Analysis:** Extracts implicit task goal from user instruction
- **Step 2 — Object Identification:** Identifies the most task-relevant object in the scene
- **Step 3 — Part & Affordance Reasoning:** Decomposes the object into functional parts and selects the optimal graspable part

Includes a comprehensive mock mode with affordance knowledge for common household objects.

### 3. Visual Grounding (`visual_grounding.py`)

Grounds VLM reasoning into pixel-level masks using:

- **LangSAM** (primary): GroundingDINO + SAM for open-vocabulary segmentation
- **Simulation fallback**: PyBullet's segmentation mask + heuristic part estimation
- Computes 3D affordance centre from depth + camera parameters

### 4. Grasp Generation (`grasp_generation.py`)

Generates task-oriented 6-DoF grasp poses:

- Antipodal grasp sampling on the affordance point cloud
- Force-closure quality estimation
- **Affordance-guided ranking**: grasps near the affordance centre score higher
- Combined scoring: `affordance_weight × proximity + quality_weight × quality`

---

## Evaluation Metrics

Following the paper, the key metrics are:

| Metric | Description |
|--------|-------------|
| **GSR** (Grasp Success Rate) | Object lifted ≥ 5cm above table |
| **TSR** (Task Success Rate) | Grasped at correct affordance region |

The paper reports (Table II, Grasping in Clutter):

| Method | Cup | Spoon | Hammer | Bowl | Screwdriver | Scissors | Wine Glass | Avg |
|--------|-----|-------|--------|------|-------------|----------|------------|-----|
| ThinkGrasp | 0.70 | 0.88 | 0.76 | 0.96 | 0.16 | 0.30 | 0.00 | 0.54 |
| **AffordGrasp** | **0.84** | **0.92** | **0.90** | **0.94** | **0.76** | **0.50** | **0.52** | **0.77** |

---

## Key Differences from Original Paper

| Aspect | Original Paper | This Reproduction |
|--------|---------------|-------------------|
| Grasp Generation | AnyGrasp (licensed) | Antipodal sampling |
| Visual Grounding | LangSAM | LangSAM or sim fallback |
| VLM | GPT-4o (API) | GPT-4o or mock mode |
| Objects | YCB dataset meshes | PyBullet primitives |
| Robot | UR5 + RS-485 gripper | UR5 + simplified gripper |
| Camera | Intel RealSense L515 | Simulated pinhole camera |

---

## Extending This Work

**For the Toyota HSR (dissertation):**
- Replace UR5 URDF with HSR URDF in `sim_env.py`
- Adjust end-effector index and gripper parameters
- Add intention-aware reasoning to the VLM prompts
- Integrate with ROS for real-world deployment

**For better grasps:**
- Apply for AnyGrasp license: [graspnet.net](https://graspnet.net/anygrasp.html)
- Use GraspNet-1Billion for training data
- Integrate Contact-GraspNet as an alternative

---

## Citation

```bibtex
@article{tang2025affordgrasp,
  title={AffordGrasp: In-Context Affordance Reasoning for Open-Vocabulary 
         Task-Oriented Grasping in Clutter},
  author={Tang, Yingbo and Zhang, Shuaike and Hao, Xiaoshuai and Wang, Pengwei 
          and Wu, Jianlong and Wang, Zhongyuan and Zhang, Shanghang},
  journal={arXiv preprint arXiv:2503.00778},
  year={2025}
}
```
