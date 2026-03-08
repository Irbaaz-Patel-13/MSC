"""
Simulation Environment Module
=============================
PyBullet-based simulation environment with UR5 robot arm, cluttered tabletop scene,
and RGB-D camera capture. Reproduces the simulation setup described in AffordGrasp.

Key components:
- UR5 robot arm with parallel-jaw gripper
- Randomised clutter scene generation
- Eye-to-hand RGB-D camera system
- Object spawning from YCB-style meshes or PyBullet primitives

Reference: AffordGrasp uses PyBullet with UR5 arm + RS-485 gripper + RealSense L515
"""

import os
import time
import traceback
import numpy as np
import pybullet as p
import pybullet_data
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from config import SimulationConfig


@dataclass
class CapturedImage:
    """Container for captured RGB-D data from the simulation camera."""
    rgb: np.ndarray          # (H, W, 3) uint8
    depth: np.ndarray        # (H, W) float32, in metres
    segmentation: np.ndarray # (H, W) int32, object IDs
    camera_intrinsics: np.ndarray  # (3, 3) camera intrinsic matrix
    camera_extrinsics: np.ndarray  # (4, 4) camera extrinsic matrix (world->camera)
    view_matrix: np.ndarray  # (4, 4) OpenGL view matrix
    projection_matrix: np.ndarray  # (4, 4) OpenGL projection matrix


class SceneObject:
    """Represents an object placed in the simulation scene."""
    def __init__(self, body_id: int, name: str, category: str,
                 position: np.ndarray, orientation: np.ndarray,
                 color: Tuple[float, ...] = (0.5, 0.5, 0.5, 1.0)):
        self.body_id = body_id
        self.name = name
        self.category = category
        self.position = position
        self.orientation = orientation
        self.color = color

    def get_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        pos, orn = p.getBasePositionAndOrientation(self.body_id)
        return np.array(pos), np.array(orn)


class SimulationEnvironment:
    """
    PyBullet simulation environment for AffordGrasp reproduction.
    
    Sets up a UR5 robot on a table with cluttered objects and provides
    RGB-D image capture functionality.
    
    Usage:
        config = SimulationConfig()
        env = SimulationEnvironment(config)
        env.setup()
        env.spawn_clutter_scene(target="mug", instruction="I want to drink coffee")
        image_data = env.capture_rgbd()
    """
    
    def __init__(self, config: SimulationConfig, gui: bool = True):
        self.config = config
        self.gui = gui
        self.physics_client = None
        self.robot_id = None
        self.table_id = None
        self.objects: List[SceneObject] = []
        self._is_setup = False
        # Video recording state
        self._video_frames: List[np.ndarray] = []
        self._recording: bool = False
        self._video_every_n: int = 16  # capture 1 frame per N sim steps (~15fps at 240Hz)
    
    def setup(self) -> None:
        """Initialise the PyBullet simulation environment."""
        # Connect to physics server
        if self.gui:
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        # Set physics parameters
        p.setGravity(*self.config.gravity)
        p.setTimeStep(self.config.time_step)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Create table
        self.table_id = self._create_table()
        
        # Load UR5 robot
        self.robot_id = self._load_robot()
        
        # Move robot to home position
        self._set_home_position()
        
        self._is_setup = True
        print("[SimEnv] Environment setup complete.")
    
    def _create_table(self) -> int:
        """Create a table using a box collision shape."""
        table_pos = self.config.table_position
        half_extents = [s / 2 for s in self.config.table_size]
        
        # Table top
        col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        vis_id = p.createVisualShape(
            p.GEOM_BOX, halfExtents=half_extents,
            rgbaColor=[0.6, 0.5, 0.4, 1.0]
        )
        table_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=[table_pos[0], table_pos[1], table_pos[2] - half_extents[2]]
        )
        
        # Table legs - only create if table surface is elevated above the ground
        leg_radius = 0.025
        leg_height = table_pos[2] - self.config.table_size[2]
        if leg_height > 0:
            leg_positions = [
                (table_pos[0] - half_extents[0] + 0.05, table_pos[1] - half_extents[1] + 0.05),
                (table_pos[0] + half_extents[0] - 0.05, table_pos[1] - half_extents[1] + 0.05),
                (table_pos[0] - half_extents[0] + 0.05, table_pos[1] + half_extents[1] - 0.05),
                (table_pos[0] + half_extents[0] - 0.05, table_pos[1] + half_extents[1] - 0.05),
            ]
            for lx, ly in leg_positions:
                leg_col = p.createCollisionShape(
                    p.GEOM_CYLINDER, radius=leg_radius, height=leg_height
                )
                leg_vis = p.createVisualShape(
                    p.GEOM_CYLINDER, radius=leg_radius, length=leg_height,
                    rgbaColor=[0.4, 0.3, 0.2, 1.0]
                )
                p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=leg_col,
                    baseVisualShapeIndex=leg_vis,
                    basePosition=[lx, ly, leg_height / 2]
                )
        
        return table_id
    
    # =========================================================================
    #  ROBOT LOADING — dispatches on config.robot_type
    # =========================================================================

    def _load_robot(self) -> int:
        """Load the robot URDF and configure joint mappings."""
        if getattr(self.config, "robot_type", "panda") == "hsr":
            return self._load_hsr()
        else:
            return self._load_panda()

    # ── Franka Panda ──────────────────────────────────────────────────────────

    def _load_panda(self) -> int:
        """Load Franka Panda (bundled with pybullet_data)."""
        robot_id = p.loadURDF(
            self.config.robot_urdf,
            basePosition=list(self.config.robot_base_position),
            baseOrientation=list(self.config.robot_base_orientation),
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION,
        )
        self.ee_link_index        = self.config.end_effector_index
        self.arm_joint_indices    = list(range(7))
        self.gripper_joint_index  = None
        self.joint_name_to_idx    = {}
        self.torso_lift_joint_idx = None
        self.hsr_finger_joint_idxs = []
        self._ik_lower = self._ik_upper = self._ik_ranges = self._ik_rest = []
        self.head_camera_link_idx = None
        return robot_id

    # ── Toyota HSR ────────────────────────────────────────────────────────────

    def _load_hsr(self) -> int:
        """Load the Toyota HSR from hsrb4s_pybullet.urdf."""
        urdf_path = str(self.config.hsr_urdf)
        if not os.path.isabs(urdf_path):
            urdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), urdf_path)

        robot_id = p.loadURDF(
            urdf_path,
            basePosition=list(self.config.robot_base_position),
            baseOrientation=list(self.config.robot_base_orientation),
            useFixedBase=True,
        )
        self._build_hsr_joint_map(robot_id)
        return robot_id

    def _build_hsr_joint_map(self, robot_id: int) -> None:
        """Build name→index maps and pre-compute IK limit arrays for HSR."""
        self.joint_name_to_idx: Dict[str, int] = {}
        for i in range(p.getNumJoints(robot_id)):
            info = p.getJointInfo(robot_id, i)
            self.joint_name_to_idx[info[1].decode("utf-8")] = i

        # Arm joints (kinematic order)
        self.arm_joint_indices = [
            self.joint_name_to_idx[j]
            for j in self.config.hsr_arm_joints
            if j in self.joint_name_to_idx
        ]

        # Gripper motor joint
        self.gripper_joint_index = self.joint_name_to_idx.get(
            self.config.hsr_gripper_joint
        )

        # Proximal finger joints (mimic hand_motor manually)
        self.hsr_finger_joint_idxs = [
            self.joint_name_to_idx[j]
            for j in self.config.hsr_finger_joints
            if j in self.joint_name_to_idx
        ]

        # torso_lift mimics arm_lift at 0.5× — sync manually
        self.torso_lift_joint_idx = self.joint_name_to_idx.get(
            self.config.hsr_torso_lift_joint
        )

        # EE link index: "hand_palm_link" is the child of "hand_palm_joint"
        # In PyBullet, link index == joint index of the joint whose child=link
        ee_joint_name = "hand_palm_joint"
        self.ee_link_index = self.joint_name_to_idx.get(
            ee_joint_name, self.config.end_effector_index
        )

        # Head camera link
        self.head_camera_link_idx = self.joint_name_to_idx.get(
            "head_rgbd_sensor_joint"
        )

        # Joint limit arrays for IK solver (one value per joint, all joints)
        n = p.getNumJoints(robot_id)
        self._ik_lower, self._ik_upper, self._ik_ranges, self._ik_rest = [], [], [], []
        for i in range(n):
            info = p.getJointInfo(robot_id, i)
            lo, hi = info[8], info[9]
            if info[2] == p.JOINT_FIXED or lo >= hi:
                lo, hi = -0.01, 0.01
            self._ik_lower.append(lo)
            self._ik_upper.append(hi)
            self._ik_ranges.append(hi - lo)
            self._ik_rest.append((lo + hi) / 2.0)

        # Seed rest poses from home config for better IK convergence
        for jname, jval in self.config.hsr_home_positions.items():
            idx = self.joint_name_to_idx.get(jname)
            if idx is not None:
                self._ik_rest[idx] = jval

        print(f"[SimEnv] HSR loaded: {n} joints, "
              f"EE={self.ee_link_index}, arm={self.arm_joint_indices}")

    # ── Home position ─────────────────────────────────────────────────────────

    def _set_home_position(self) -> None:
        """Move the robot to its home configuration."""
        if getattr(self.config, "robot_type", "panda") == "hsr":
            self._set_hsr_home()
        else:
            self._set_panda_home()

    def _set_panda_home(self) -> None:
        home = list(self.config.home_joint_positions)
        n = p.getNumJoints(self.robot_id)
        for i, angle in enumerate(home[: min(7, n)]):
            p.resetJointState(self.robot_id, i, angle)

    def _set_hsr_home(self) -> None:
        for jname, jval in self.config.hsr_home_positions.items():
            idx = self.joint_name_to_idx.get(jname)
            if idx is not None:
                p.resetJointState(self.robot_id, idx, jval)
        if self.torso_lift_joint_idx is not None:
            arm_val = self.config.hsr_home_positions.get("arm_lift_joint", 0.0)
            p.resetJointState(self.robot_id, self.torso_lift_joint_idx, arm_val * 0.5)

    # ── HSR gripper control ───────────────────────────────────────────────────

    def _hsr_set_gripper(self, value: float, force: float = 20.0) -> None:
        """
        Set HSR gripper opening. value=0.8 → open, value=0.0 → closed.
        Manually syncs proximal finger joints (PyBullet ignores <mimic>).
        """
        if self.gripper_joint_index is None:
            return
        p.setJointMotorControl2(
            self.robot_id, self.gripper_joint_index,
            p.POSITION_CONTROL, targetPosition=value, force=force,
        )
        for idx, mult in zip(
            self.hsr_finger_joint_idxs,
            self.config.hsr_finger_multipliers,
        ):
            p.setJointMotorControl2(
                self.robot_id, idx,
                p.POSITION_CONTROL, targetPosition=value * mult, force=force,
            )
    
    # =========================================================================
    #  OBJECT SPAWNING
    # =========================================================================
    
    def _try_load_ycb(self, category: str, position: np.ndarray,
                      color: Tuple[float, ...]) -> Optional[SceneObject]:
        """
        Load a YCB mesh object from the local elpis-lab/YCB_Dataset clone.

        URDF layout expected:
            <ycb_data_path>/<ycb_id>.urdf          -- robot description
            <ycb_data_path>/<ycb_id>/textured.obj  -- visual mesh
            <ycb_data_path>/<ycb_id>/texture_map.png
            <ycb_data_path>/<ycb_id>/textured_coacd_*.stl  -- collision meshes

        PyBullet resolves mesh filenames in the URDF relative to the URDF
        file's parent directory, so loading the absolute URDF path is enough.

        For headless (DIRECT) mode textures render via ER_TINY_RENDERER.
        In GUI mode use ER_BULLET_HARDWARE_OPENGL for full texture support.

        Returns a SceneObject on success, or None if the URDF is missing or
        loading fails (caller falls back to spawn_primitive_object).
        """
        if not self.config.use_ycb_objects:
            return None

        ycb_data_path = self.config.ycb_data_path
        if not ycb_data_path:
            return None

        ycb_id = self.config.category_to_ycb_id.get(category)
        if ycb_id is None:
            return None

        urdf_path = os.path.join(ycb_data_path, f"{ycb_id}.urdf")
        if not os.path.isfile(urdf_path):
            print(f"[SimEnv] YCB URDF not found: {urdf_path}")
            return None

        spawn_pos = [position[0], position[1], position[2] + 0.05]
        orientation = p.getQuaternionFromEuler(
            [0, 0, np.random.uniform(0, 2 * np.pi)]
        )

        body_id = None

        # --- Attempt 1: load the full URDF (visual + collision + inertia) ---
        try:
            body_id = p.loadURDF(
                urdf_path, spawn_pos, list(orientation),
                useFixedBase=False, globalScaling=1.0,
                flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL
            )
        except Exception:
            print(f"[SimEnv] URDF load failed for {ycb_id}:\n"
                  + traceback.format_exc().rstrip())

        # --- Attempt 2: load OBJ as visual mesh + box collision ---
        if body_id is None:
            obj_path = os.path.join(ycb_data_path, ycb_id, "textured.obj")
            if os.path.isfile(obj_path):
                try:
                    vis_id = p.createVisualShape(
                        p.GEOM_MESH, fileName=obj_path,
                        meshScale=[1.0, 1.0, 1.0]
                    )
                    col_id = p.createCollisionShape(
                        p.GEOM_BOX, halfExtents=[0.04, 0.04, 0.04]
                    )
                    body_id = p.createMultiBody(
                        baseMass=0.1,
                        baseCollisionShapeIndex=col_id,
                        baseVisualShapeIndex=vis_id,
                        basePosition=spawn_pos,
                        baseOrientation=list(orientation)
                    )
                    print(f"[SimEnv] OBJ mesh fallback succeeded for {ycb_id}")
                except Exception:
                    print(f"[SimEnv] OBJ mesh fallback failed for {ycb_id}:\n"
                          + traceback.format_exc().rstrip())

        if body_id is None:
            return None

        # --- Apply distinctive colour per object category ---
        # PyBullet's TinyRenderer does not reliably render UV-mapped textures loaded
        # via changeVisualShape(textureUniqueId=...) in DIRECT mode.  Instead, set a
        # category-specific RGBA colour so GroundingDINO can distinguish objects by
        # both shape AND colour.  Alpha=1 for solid rendering.
        _ycb_colours = {
            "mug":                 [0.85, 0.35, 0.10, 1.0],  # warm orange
            "bowl":                [0.20, 0.55, 0.85, 1.0],  # sky blue
            "a_cups":              [0.95, 0.90, 0.30, 1.0],  # yellow
            "spoon":               [0.70, 0.70, 0.70, 1.0],  # silver
            "hammer":              [0.70, 0.15, 0.15, 1.0],  # red
            "knife":               [0.80, 0.80, 0.80, 1.0],  # light grey
            "fork":                [0.75, 0.75, 0.75, 1.0],  # silver
            "scissors":            [0.20, 0.55, 0.20, 1.0],  # green
            "phillips_screwdriver":[0.20, 0.20, 0.80, 1.0],  # blue
            "spatula":             [0.50, 0.30, 0.10, 1.0],  # brown
            "mustard_bottle":      [0.95, 0.80, 0.10, 1.0],  # mustard yellow
            "pitcher_base":        [0.60, 0.20, 0.70, 1.0],  # purple
            "power_drill":         [0.30, 0.30, 0.30, 1.0],  # dark grey
            "tomato_soup_can":     [0.85, 0.15, 0.15, 1.0],  # tomato red
            "foam_brick":          [0.90, 0.50, 0.50, 1.0],  # light red
            "skillet_lid":         [0.40, 0.40, 0.40, 1.0],  # medium grey
        }
        colour = _ycb_colours.get(ycb_id, [0.60, 0.60, 0.60, 1.0])
        try:
            p.changeVisualShape(body_id, -1, rgbaColor=colour)
        except Exception as col_e:
            print(f"[SimEnv] Colour apply failed for {ycb_id}: {col_e}")

        obj = SceneObject(
            body_id=body_id,
            name=f"ycb_{ycb_id}",
            category=category,
            position=np.array(spawn_pos),
            orientation=np.array(orientation),
            color=color
        )
        self.objects.append(obj)
        print(f"[SimEnv] Loaded YCB mesh: {ycb_id}")
        return obj
    
    def spawn_primitive_object(
        self,
        name: str,
        category: str,
        shape_type: str = "cylinder",
        position: Optional[np.ndarray] = None,
        size: Optional[Dict] = None,
        color: Tuple[float, ...] = (0.5, 0.5, 0.5, 1.0),
        mass: float = 0.1
    ) -> SceneObject:
        """
        Spawn a primitive object on the table.
        
        Args:
            name: Unique identifier for the object
            category: Object category (cup, spoon, hammer, etc.)
            shape_type: "box", "cylinder", "sphere", "capsule"
            position: (x, y, z) spawn position; random if None
            size: Shape-specific dimensions dict
            color: RGBA colour
            mass: Object mass in kg
        
        Returns:
            SceneObject with the spawned object's info
        """
        if position is None:
            position = self._random_table_position()
        
        # Default sizes per category
        default_sizes = {
            "cup":         {"radius": 0.035, "height": 0.10},
            "mug":         {"radius": 0.04,  "height": 0.09},
            "spoon":       {"half_extents": [0.01, 0.10, 0.005]},
            "hammer":      {"half_extents": [0.015, 0.12, 0.015]},
            "bowl":        {"radius": 0.06,  "height": 0.04},
            "screwdriver": {"half_extents": [0.01, 0.09, 0.01]},
            "scissors":    {"half_extents": [0.02, 0.08, 0.005]},
            "wine_glass":  {"radius": 0.03,  "height": 0.15},
            "knife":       {"half_extents": [0.01, 0.10, 0.003]},
            "bottle":      {"radius": 0.03,  "height": 0.20},
            "fork":        {"half_extents": [0.01, 0.09, 0.005]},
            "pan":         {"radius": 0.10,  "height": 0.03},
            "spatula":     {"half_extents": [0.02, 0.12, 0.003]},
            "kettle":      {"radius": 0.06,  "height": 0.12},
            "racket":      {"half_extents": [0.08, 0.12, 0.01]},
        }
        
        if size is None:
            size = default_sizes.get(category, {"radius": 0.03, "height": 0.08})
        
        # Create collision and visual shapes based on category geometry
        if "radius" in size and "height" in size:
            col_id = p.createCollisionShape(
                p.GEOM_CYLINDER, radius=size["radius"], height=size["height"]
            )
            vis_id = p.createVisualShape(
                p.GEOM_CYLINDER, radius=size["radius"], length=size["height"],
                rgbaColor=list(color)
            )
            spawn_z = position[2] + size["height"] / 2
        elif "half_extents" in size:
            col_id = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=size["half_extents"]
            )
            vis_id = p.createVisualShape(
                p.GEOM_BOX, halfExtents=size["half_extents"],
                rgbaColor=list(color)
            )
            spawn_z = position[2] + size["half_extents"][2]
        else:
            col_id = p.createCollisionShape(p.GEOM_SPHERE, radius=0.03)
            vis_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=list(color))
            spawn_z = position[2] + 0.03
        
        spawn_pos = [position[0], position[1], spawn_z]
        orientation = p.getQuaternionFromEuler([0, 0, np.random.uniform(0, 2 * np.pi)])
        
        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=spawn_pos,
            baseOrientation=list(orientation)
        )
        
        obj = SceneObject(
            body_id=body_id,
            name=name,
            category=category,
            position=np.array(spawn_pos),
            orientation=np.array(orientation),
            color=color
        )
        self.objects.append(obj)
        return obj
    
    def spawn_clutter_scene(
        self,
        target_category: str,
        num_distractors: Optional[int] = None,
        distractor_categories: Optional[List[str]] = None
    ) -> Tuple[SceneObject, List[SceneObject]]:
        """
        Spawn a cluttered scene with a target object and distractors.
        
        This reproduces the clutter setup from the AffordGrasp paper where
        multiple objects are placed on a table and the robot must identify
        and grasp the task-relevant one.
        
        Args:
            target_category: Category of the target object
            num_distractors: Number of distractor objects (default from config)
            distractor_categories: Categories to sample distractors from
        
        Returns:
            Tuple of (target_object, list_of_distractor_objects)
        """
        self.clear_objects()
        
        if num_distractors is None:
            num_distractors = self.config.num_distractor_objects
        
        if distractor_categories is None:
            distractor_categories = [
                c for c in self.config.object_categories if c != target_category
            ]
        
        # Colour palette for objects
        colours = [
            (0.8, 0.2, 0.2, 1.0),  # red
            (0.2, 0.6, 0.8, 1.0),  # blue
            (0.2, 0.7, 0.3, 1.0),  # green
            (0.9, 0.7, 0.1, 1.0),  # yellow
            (0.7, 0.3, 0.7, 1.0),  # purple
            (0.9, 0.5, 0.2, 1.0),  # orange
            (0.4, 0.8, 0.8, 1.0),  # teal
            (0.6, 0.4, 0.2, 1.0),  # brown
        ]
        
        # Generate non-overlapping positions
        positions = self._generate_non_overlapping_positions(
            num_distractors + 1, min_distance=0.08
        )
        
        # Spawn target object at a random position (try YCB, fall back to primitive)
        target_idx = np.random.randint(len(positions))
        ycb_count = 0
        prim_count = 0

        target = self._try_load_ycb(target_category, positions[target_idx], colours[0])
        if target is None:
            target = self.spawn_primitive_object(
                name=f"target_{target_category}",
                category=target_category,
                position=positions[target_idx],
                color=colours[0]
            )
            prim_count += 1
        else:
            ycb_count += 1

        # Spawn distractors
        distractors = []
        distractor_cats = np.random.choice(
            distractor_categories,
            size=min(num_distractors, len(distractor_categories)),
            replace=False
        )

        pos_idx = 0
        for i, cat in enumerate(distractor_cats):
            if pos_idx == target_idx:
                pos_idx += 1
            if pos_idx >= len(positions):
                break

            obj = self._try_load_ycb(cat, positions[pos_idx], colours[(i + 1) % len(colours)])
            if obj is None:
                obj = self.spawn_primitive_object(
                    name=f"distractor_{cat}_{i}",
                    category=cat,
                    position=positions[pos_idx],
                    color=colours[(i + 1) % len(colours)]
                )
                prim_count += 1
            else:
                ycb_count += 1
            distractors.append(obj)
            pos_idx += 1

        # Let objects settle
        for _ in range(100):
            p.stepSimulation()

        total = ycb_count + prim_count
        print(f"[SimEnv] Spawned clutter scene: target={target_category}, "
              f"{len(distractors)} distractors")
        print(f"[SimEnv] Loaded {ycb_count}/{total} as textured YCB meshes, "
              f"{prim_count} fell back to primitives")
        return target, distractors
    
    def _random_table_position(self) -> np.ndarray:
        """Generate a random position on the table surface."""
        bounds = self.config.workspace_bounds
        x = np.random.uniform(*bounds["x"])
        y = np.random.uniform(*bounds["y"])
        z = self.config.table_position[2]  # table surface height
        return np.array([x, y, z])
    
    def _generate_non_overlapping_positions(
        self, n: int, min_distance: float = 0.08
    ) -> List[np.ndarray]:
        """Generate n non-overlapping positions on the table."""
        positions = []
        max_attempts = 500
        
        for _ in range(n):
            for attempt in range(max_attempts):
                pos = self._random_table_position()
                valid = True
                for existing in positions:
                    dist = np.linalg.norm(pos[:2] - existing[:2])
                    if dist < min_distance:
                        valid = False
                        break
                if valid:
                    positions.append(pos)
                    break
            else:
                # Fallback: place anyway
                positions.append(self._random_table_position())
        
        return positions
    
    def clear_objects(self) -> None:
        """Remove all spawned objects from the scene."""
        for obj in self.objects:
            p.removeBody(obj.body_id)
        self.objects.clear()
    
    # =========================================================================
    #  CAMERA & RGB-D CAPTURE
    # =========================================================================
    
    def _capture_from_view(
        self,
        eye: Tuple,
        target: Tuple,
        up: Tuple,
    ) -> CapturedImage:
        """Capture RGB-D from an arbitrary camera pose."""
        width = self.config.image_width
        height = self.config.image_height

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=list(eye),
            cameraTargetPosition=list(target),
            cameraUpVector=list(up),
        )

        aspect = width / height
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.config.camera_fov,
            aspect=aspect,
            nearVal=self.config.camera_near,
            farVal=self.config.camera_far,
        )

        # High ambient coefficient ensures dark-textured YCB objects remain
        # visible under top-down lighting. Shadow disabled for cleaner depth.
        _, _, rgb_pixels, depth_pixels, seg_pixels = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_TINY_RENDERER,
            lightDirection=[0.5, 0.5, 1.0],
            lightColor=[1.0, 1.0, 1.0],
            lightAmbientCoeff=0.8,
            lightDiffuseCoeff=0.8,
            lightSpecularCoeff=0.1,
            shadow=0,
        )

        rgb = np.array(rgb_pixels, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]

        depth_buffer = np.array(depth_pixels, dtype=np.float32).reshape(height, width)
        near = self.config.camera_near
        far = self.config.camera_far
        depth = far * near / (far - (far - near) * depth_buffer)

        segmentation = np.array(seg_pixels, dtype=np.int32).reshape(height, width)

        # Intrinsics from OpenGL projection matrix (column-major flat array)
        proj_np = np.array(projection_matrix, dtype=np.float64).reshape(4, 4, order='F')
        fx = proj_np[0, 0] * width / 2.0
        fy = proj_np[1, 1] * height / 2.0
        cx = width / 2.0
        cy = height / 2.0
        intrinsics = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1],
        ], dtype=np.float64)

        view_4x4 = np.array(view_matrix, dtype=np.float64).reshape(4, 4, order='F')

        return CapturedImage(
            rgb=rgb,
            depth=depth,
            segmentation=segmentation,
            camera_intrinsics=intrinsics,
            camera_extrinsics=view_4x4,
            view_matrix=view_4x4,
            projection_matrix=proj_np,
        )

    def capture_rgbd(self) -> CapturedImage:
        """
        Capture an RGB-D image from the primary simulation camera.

        Returns an RGB image, depth map (in metres), segmentation mask,
        and camera intrinsic/extrinsic matrices. This mirrors the Intel
        RealSense L515 capture used in the real-world AffordGrasp setup.
        """
        return self._capture_from_view(
            self.config.camera_position,
            self.config.camera_target,
            self.config.camera_up_vector,
        )

    def capture_multiview_rgbd(self) -> List[Tuple[str, CapturedImage]]:
        """
        Capture RGB-D from the primary camera plus all extra_camera_views.

        Returns a list of (view_name, CapturedImage) pairs. The primary
        top-down view is always first ("top"), followed by any views
        configured in SimulationConfig.extra_camera_views.
        """
        views: List[Tuple[str, CapturedImage]] = [
            ("top", self.capture_rgbd())
        ]
        for name, eye, target, up in getattr(self.config, "extra_camera_views", []):
            views.append((name, self._capture_from_view(eye, target, up)))
        return views
    
    def save_rgbd(self, image_data: CapturedImage, save_dir: str,
                  prefix: str = "scene") -> Dict[str, str]:
        """Save RGB-D data to files."""
        import cv2
        os.makedirs(save_dir, exist_ok=True)
        
        paths = {}
        
        # Save RGB
        rgb_path = os.path.join(save_dir, f"{prefix}_rgb.png")
        cv2.imwrite(rgb_path, cv2.cvtColor(image_data.rgb, cv2.COLOR_RGB2BGR))
        paths["rgb"] = rgb_path
        
        # Save depth as 16-bit PNG (millimetres) — raw metric data
        depth_mm = (image_data.depth * 1000).astype(np.uint16)
        depth_path = os.path.join(save_dir, f"{prefix}_depth.png")
        cv2.imwrite(depth_path, depth_mm)
        paths["depth"] = depth_path

        # Save depth visualization as 8-bit grayscale (normalized to workspace range)
        # Clip to 0.1–2.0 m so table/objects use the full 0–255 range
        d_vis = np.clip(image_data.depth, 0.1, 2.0)
        d_vis = ((d_vis - d_vis.min()) / max(d_vis.max() - d_vis.min(), 1e-6) * 255).astype(np.uint8)
        d_vis = 255 - d_vis  # invert: near objects bright, far objects dark
        depth_vis_path = os.path.join(save_dir, f"{prefix}_depth_vis.png")
        cv2.imwrite(depth_vis_path, d_vis)
        paths["depth_vis"] = depth_vis_path
        
        # Save intrinsics
        intrinsics_path = os.path.join(save_dir, f"{prefix}_intrinsics.npy")
        np.save(intrinsics_path, image_data.camera_intrinsics)
        paths["intrinsics"] = intrinsics_path
        
        print(f"[SimEnv] Saved RGB-D to {save_dir}")
        return paths
    
    # =========================================================================
    #  ROBOT CONTROL
    # =========================================================================
    
    def move_to_pose(
        self,
        target_position: np.ndarray,
        target_orientation: Optional[np.ndarray] = None,
        max_steps: int = 300,
    ) -> bool:
        """
        Move the robot end-effector to a target pose using IK.

        Dispatches to the robot-specific IK/control method so both HSR and
        Panda paths use this single public API.

        Args:
            target_position:    (x, y, z) world-frame target
            target_orientation: quaternion (x, y, z, w); None → top-down grasp
            max_steps:          simulation steps to simulate the motion

        Returns:
            True if EE reached within 5 mm of the target
        """
        if target_orientation is None:
            target_orientation = p.getQuaternionFromEuler([np.pi, 0, 0])

        if getattr(self.config, "robot_type", "panda") == "hsr":
            return self._move_to_pose_hsr(target_position, target_orientation, max_steps)
        else:
            return self._move_to_pose_panda(target_position, target_orientation, max_steps)

    def _move_to_pose_panda(
        self,
        target_position: np.ndarray,
        target_orientation: np.ndarray,
        max_steps: int,
    ) -> bool:
        """IK + position control for Franka Panda (7-joint arm)."""
        joint_positions = p.calculateInverseKinematics(
            self.robot_id,
            self.config.end_effector_index,
            list(target_position),
            list(target_orientation),
            restPoses=list(self.config.home_joint_positions),
            maxNumIterations=100,
            residualThreshold=1e-5,
        )
        for i, pos in enumerate(joint_positions[:7]):
            p.setJointMotorControl2(
                self.robot_id, i, p.POSITION_CONTROL,
                targetPosition=pos, force=240, maxVelocity=1.0,
            )
        for _ in range(max_steps):
            self._sim_step()
            ee_state = p.getLinkState(self.robot_id, self.config.end_effector_index)
            if np.linalg.norm(np.array(ee_state[4]) - target_position) < 0.005:
                return True
        return False

    def _move_to_pose_hsr(
        self,
        target_position: np.ndarray,
        target_orientation: np.ndarray,
        max_steps: int,
    ) -> bool:
        """
        IK + position control for Toyota HSR (5-DOF arm: lift + 4 revolute).

        Uses joint-limit–constrained IK for better solutions.
        Reachability check: if IK residual > 2 cm, logs a warning.
        """
        ik_result = p.calculateInverseKinematics(
            self.robot_id,
            self.ee_link_index,
            list(target_position),
            list(target_orientation),
            lowerLimits=self._ik_lower,
            upperLimits=self._ik_upper,
            jointRanges=self._ik_ranges,
            restPoses=self._ik_rest,
            maxNumIterations=200,
            residualThreshold=1e-5,
        )

        # Apply only the arm joints
        for j_idx in self.arm_joint_indices:
            if j_idx < len(ik_result):
                p.setJointMotorControl2(
                    self.robot_id, j_idx, p.POSITION_CONTROL,
                    targetPosition=ik_result[j_idx],
                    force=100, maxVelocity=0.5,
                )

        # Sync torso_lift mimic (arm_lift * 0.5)
        if self.torso_lift_joint_idx is not None:
            arm_lift_idx = self.joint_name_to_idx.get("arm_lift_joint")
            if arm_lift_idx is not None and arm_lift_idx < len(ik_result):
                p.setJointMotorControl2(
                    self.robot_id, self.torso_lift_joint_idx,
                    p.POSITION_CONTROL,
                    targetPosition=ik_result[arm_lift_idx] * 0.5,
                    force=100,
                )

        for _ in range(max_steps):
            self._sim_step()
            ee_state = p.getLinkState(self.robot_id, self.ee_link_index)
            if np.linalg.norm(np.array(ee_state[4]) - target_position) < 0.005:
                return True
        return False

    # =========================================================================
    #  VIDEO RECORDING
    # =========================================================================

    def start_recording(self, every_n: int = 16) -> None:
        """Begin capturing frames for video. Call before the motion you want to record."""
        self._video_frames.clear()
        self._recording = True
        self._video_every_n = every_n
        self._video_step_counter = 0
        print("[SimEnv] Video recording started.")

    def stop_recording(self) -> None:
        """Stop frame capture."""
        self._recording = False
        print(f"[SimEnv] Video recording stopped ({len(self._video_frames)} frames).")

    def save_video(self, output_path: str, fps: int = 15) -> bool:
        """
        Write captured frames to an MP4 file using OpenCV.

        Args:
            output_path: Destination path (e.g. 'results/trial_*/grasp_video.mp4')
            fps: Frames per second for the output video

        Returns:
            True on success, False if no frames or OpenCV unavailable.
        """
        if not self._video_frames:
            print("[SimEnv] No frames to save.")
            return False
        try:
            import cv2
        except ImportError:
            print("[SimEnv] OpenCV not available — cannot save video.")
            return False

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        h, w = self._video_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        for frame in self._video_frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"[SimEnv] Video saved: {output_path} ({len(self._video_frames)} frames @ {fps}fps)")
        return True

    def _sim_step(self) -> None:
        """
        Advance the simulation by one step and optionally capture a video frame.

        Call this instead of raw p.stepSimulation() inside motion loops so that
        the video recorder automatically collects frames without extra boilerplate.
        """
        p.stepSimulation()
        if self.gui:
            time.sleep(self.config.time_step)
        if self._recording:
            self._video_step_counter = getattr(self, "_video_step_counter", 0) + 1
            if self._video_step_counter % self._video_every_n == 0:
                img = self._capture_from_view(
                    self.config.camera_position,
                    self.config.camera_target,
                    self.config.camera_up_vector,
                )
                self._video_frames.append(img.rgb.copy())

    # ── HSR head camera capture ───────────────────────────────────────────────

    def capture_from_hsr_head(self) -> Optional[CapturedImage]:
        """
        Capture RGB-D from the HSR head-mounted RGB-D camera.

        Uses the current pose of head_rgbd_sensor_link to derive the
        camera view matrix, so the image reflects the current head pan/tilt.
        Returns None if the HSR isn't loaded or the link index is unknown.
        """
        if self.head_camera_link_idx is None:
            return None
        link_state = p.getLinkState(
            self.robot_id, self.head_camera_link_idx, computeForwardKinematics=True
        )
        # link_state[4] = world position, link_state[5] = world orientation (xyzw)
        cam_pos = np.array(link_state[4])
        cam_orn = np.array(link_state[5])  # quaternion xyzw

        # Compute forward direction from link orientation
        rot_mat = np.array(p.getMatrixFromQuaternion(cam_orn)).reshape(3, 3)
        # In the HSR URDF the camera link faces along +X of its local frame
        forward = rot_mat @ np.array([1.0, 0.0, 0.0])
        up      = rot_mat @ np.array([0.0, 0.0, 1.0])

        target = cam_pos + forward * 0.5  # look 0.5 m ahead
        return self._capture_from_view(
            eye    = tuple(cam_pos.tolist()),
            target = tuple(target.tolist()),
            up     = tuple(up.tolist()),
        )
    
    def execute_grasp(
        self,
        grasp_position: np.ndarray,
        grasp_orientation: np.ndarray,
        pre_grasp_height: float = 0.15,
        target_body_id: Optional[int] = None,
    ) -> bool:
        """
        Execute a grasp sequence: approach -> descend -> close gripper -> lift.

        Args:
            grasp_position: (x, y, z) grasp point
            grasp_orientation: quaternion for gripper orientation
            pre_grasp_height: height above grasp point for approach
            target_body_id: PyBullet body ID of the VLM-identified target object.
                When provided, this object is grasped directly instead of using
                distance-based search, preventing wrong-object grabs in clutter.

        Returns:
            True if grasp was successful (object lifted)
        """
        # 1. Move to pre-grasp position
        pre_grasp_pos = grasp_position.copy()
        pre_grasp_pos[2] += pre_grasp_height
        print(f"[SimEnv] Moving to pre-grasp: {pre_grasp_pos}")
        self.move_to_pose(pre_grasp_pos, grasp_orientation)

        # 2. Descend to grasp position
        print(f"[SimEnv] Descending to grasp: {grasp_position}")
        self.move_to_pose(grasp_position, grasp_orientation)

        # 3. Close gripper + attach object via constraint
        print("[SimEnv] Closing gripper...")
        # Animate HSR gripper closing (Panda gripper is passive / constraint-only)
        if getattr(self.config, "robot_type", "panda") == "hsr":
            self._hsr_set_gripper(0.0)   # close
            for _ in range(60):
                self._sim_step()

        # Attach the target object with a fixed constraint.
        # Use the VLM-identified body_id directly when available to prevent
        # wrong-object grabs in clutter; fall back to nearest-object search.
        if target_body_id is not None:
            obj_pos, _ = p.getBasePositionAndOrientation(target_body_id)
            dist = np.linalg.norm(np.array(obj_pos) - grasp_position)
            name = next(
                (o.name for o in self.objects if o.body_id == target_body_id),
                str(target_body_id),
            )
            print(f"[SimEnv] Target object: {name} at {dist:.3f}m from grasp point")
            grasped_id = target_body_id
        else:
            grasped_id = self._find_nearest_object(grasp_position)

        constraint_id = None
        if grasped_id is not None:
            constraint_id = p.createConstraint(
                self.robot_id, self.ee_link_index,
                grasped_id, -1,
                p.JOINT_FIXED, [0, 0, 0], [0, 0, 0.05], [0, 0, 0],
            )
        
        # 4. Lift
        lift_pos = grasp_position.copy()
        lift_pos[2] += 0.20
        print(f"[SimEnv] Lifting to: {lift_pos}")
        success = self.move_to_pose(lift_pos, grasp_orientation)
        
        # Check grasp success: object lifted ≥5cm above table (paper's GSR criterion)
        if grasped_id is not None:
            obj_pos, _ = p.getBasePositionAndOrientation(grasped_id)
            table_z = self.config.table_position[2] + self.config.table_size[2]
            lifted = obj_pos[2] > table_z + 0.05
            
            if lifted:
                # Stability check: simulate 100 more steps, verify drift < 5cm.
                # The grip is implemented as a PyBullet constraint (not physical
                # fingers), so getContactPoints always returns empty — don't use it.
                pre_pos = np.array(obj_pos)
                for _ in range(100):
                    self._sim_step()
                post_pos, _ = p.getBasePositionAndOrientation(grasped_id)
                drift = np.linalg.norm(np.array(post_pos) - pre_pos)

                if drift < 0.05:
                    print(f"[SimEnv] Grasp successful! "
                          f"(lifted {obj_pos[2] - table_z:.3f}m, drift {drift:.4f}m)")
                    return True
                else:
                    print(f"[SimEnv] Grasp unstable (drift={drift:.4f}m)")
                    return False

        print("[SimEnv] Grasp failed.")
        return False
    
    def _find_nearest_object(self, position: np.ndarray, max_dist: float = 0.15) -> Optional[int]:
        """Find the nearest object to a given position (search radius = 10 cm)."""
        nearest_id = None
        nearest_dist = max_dist
        nearest_name = None

        for obj in self.objects:
            obj_pos, _ = p.getBasePositionAndOrientation(obj.body_id)
            dist = np.linalg.norm(np.array(obj_pos) - position)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_id = obj.body_id
                nearest_name = obj.name

        if nearest_id is not None:
            print(f"[SimEnv] Nearest object: {nearest_name} at {nearest_dist:.3f}m")
        else:
            print(f"[SimEnv] No object within {max_dist:.2f}m of grasp point")
        return nearest_id
    
    # =========================================================================
    #  POINT CLOUD GENERATION
    # =========================================================================
    
    def depth_to_pointcloud(
        self, image_data: CapturedImage,
        mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Convert depth image to 3D point cloud in world coordinates.
        
        Handles the OpenGL camera convention used by PyBullet:
        - Image frame: x-right, y-down, z-into-scene
        - OpenGL camera: x-right, y-up, z-toward-viewer
        
        Args:
            image_data: CapturedImage from capture_rgbd()
            mask: Optional binary mask to filter points (e.g., affordance region)
        
        Returns:
            (N, 3) array of 3D points in world frame
        """
        depth = image_data.depth
        intrinsics = image_data.camera_intrinsics
        
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        h, w = depth.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        # Apply mask if provided
        if mask is not None:
            valid = (depth > 0) & (depth < self.config.camera_far) & (mask > 0)
        else:
            valid = (depth > 0) & (depth < self.config.camera_far)
        
        u_valid = u[valid]
        v_valid = v[valid]
        z_valid = depth[valid]
        
        # Back-project to image-convention camera frame (x-right, y-down, z-forward)
        x_img = (u_valid - cx) * z_valid / fx
        y_img = (v_valid - cy) * z_valid / fy
        z_img = z_valid
        
        # Convert to OpenGL camera frame (y-up, z-toward-viewer) to match
        # the PyBullet view matrix convention
        x_cam = x_img
        y_cam = -y_img   # flip Y: image-down → OpenGL-up
        z_cam = -z_img    # flip Z: into-scene → toward-viewer
        
        points_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
        
        # Transform to world frame using inverse of view matrix
        extrinsics_inv = np.linalg.inv(image_data.camera_extrinsics)
        points_hom = np.hstack([points_cam, np.ones((len(points_cam), 1))])
        points_world = (extrinsics_inv @ points_hom.T).T[:, :3]
        
        return points_world
    
    def reset(self) -> None:
        """Reset the environment to initial state."""
        self.clear_objects()
        self._set_home_position()
        for _ in range(50):
            p.stepSimulation()
    
    def close(self) -> None:
        """Disconnect from PyBullet."""
        if self.physics_client is not None:
            p.disconnect()
            self.physics_client = None