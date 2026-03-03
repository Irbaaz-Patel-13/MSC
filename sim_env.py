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
import numpy as np
import pybullet as p
import pybullet_data
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .config import SimulationConfig


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
        
        # Table legs
        leg_radius = 0.025
        leg_height = table_pos[2] - self.config.table_size[2]
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
    
    def _load_robot(self) -> int:
        """Load the UR5 robot URDF."""
        robot_id = p.loadURDF(
            self.config.robot_urdf,
            basePosition=list(self.config.robot_base_position),
            baseOrientation=list(self.config.robot_base_orientation),
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION
        )
        return robot_id
    
    def _set_home_position(self) -> None:
        """Set the robot to a neutral home position above the table."""
        home_joints = list(self.config.home_joint_positions)
        num_joints = p.getNumJoints(self.robot_id)
        
        for i, angle in enumerate(home_joints[:min(6, num_joints)]):
            p.resetJointState(self.robot_id, i, angle)
    
    # =========================================================================
    #  OBJECT SPAWNING
    # =========================================================================
    
    def _try_load_ycb(self, category: str, position: np.ndarray,
                      color: Tuple[float, ...]) -> Optional[SceneObject]:
        """
        Attempt to load a YCB mesh object. Returns None if unavailable.
        Uses pybullet_object_models if installed, else returns None.
        """
        if not self.config.use_ycb_objects:
            return None
        
        # Map category names to YCB model names
        category_to_ycb = {
            "mug": "YcbMug", "cup": "YcbMug",
            "bowl": "YcbBowl",
            "bottle": "YcbMustardBottle",
            "hammer": "YcbHammer",
            "knife": "YcbKnife",
            "fork": "YcbFork",
            "spoon": "YcbSpoon",
            "scissors": "YcbScissors",
            "pan": "YcbSkillet",
        }
        
        ycb_name = category_to_ycb.get(category)
        if ycb_name is None:
            return None
        
        try:
            from pybullet_object_models import ycb_objects
            urdf_path = os.path.join(
                ycb_objects.getDataPath(), ycb_name, "model.urdf"
            )
            if not os.path.exists(urdf_path):
                return None
            
            spawn_pos = [position[0], position[1], position[2] + 0.05]
            orientation = p.getQuaternionFromEuler(
                [0, 0, np.random.uniform(0, 2 * np.pi)]
            )
            
            body_id = p.loadURDF(urdf_path, spawn_pos, list(orientation))
            
            obj = SceneObject(
                body_id=body_id,
                name=f"ycb_{ycb_name}",
                category=category,
                position=np.array(spawn_pos),
                orientation=np.array(orientation),
                color=color
            )
            self.objects.append(obj)
            return obj
            
        except ImportError:
            return None
        except Exception:
            return None
    
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
        target = self._try_load_ycb(target_category, positions[target_idx], colours[0])
        if target is None:
            target = self.spawn_primitive_object(
                name=f"target_{target_category}",
                category=target_category,
                position=positions[target_idx],
                color=colours[0]
            )
        
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
            distractors.append(obj)
            pos_idx += 1
        
        # Let objects settle
        for _ in range(100):
            p.stepSimulation()
        
        print(f"[SimEnv] Spawned clutter scene: target={target_category}, "
              f"{len(distractors)} distractors")
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
    
    def capture_rgbd(self) -> CapturedImage:
        """
        Capture an RGB-D image from the simulation camera.
        
        Returns an RGB image, depth map (in metres), segmentation mask,
        and camera intrinsic/extrinsic matrices. This mirrors the Intel
        RealSense L515 capture used in the real-world AffordGrasp setup.
        """
        width = self.config.image_width
        height = self.config.image_height
        
        # Compute view and projection matrices
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=list(self.config.camera_position),
            cameraTargetPosition=list(self.config.camera_target),
            cameraUpVector=list(self.config.camera_up_vector)
        )
        
        aspect = width / height
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.config.camera_fov,
            aspect=aspect,
            nearVal=self.config.camera_near,
            farVal=self.config.camera_far
        )
        
        # Capture image
        _, _, rgb_pixels, depth_pixels, seg_pixels = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL if self.gui else p.ER_TINY_RENDERER
        )
        
        # Process RGB (remove alpha channel)
        rgb = np.array(rgb_pixels, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
        
        # Convert depth buffer to metric depth
        depth_buffer = np.array(depth_pixels, dtype=np.float32).reshape(height, width)
        near = self.config.camera_near
        far = self.config.camera_far
        depth = far * near / (far - (far - near) * depth_buffer)
        
        # Segmentation mask
        segmentation = np.array(seg_pixels, dtype=np.int32).reshape(height, width)
        
        # Compute camera intrinsic matrix from the OpenGL projection matrix.
        # PyBullet returns column-major (OpenGL convention) flat arrays.
        proj_np = np.array(projection_matrix, dtype=np.float64).reshape(4, 4, order='F')
        fx = proj_np[0, 0] * width / 2.0
        fy = proj_np[1, 1] * height / 2.0
        cx = width / 2.0
        cy = height / 2.0
        intrinsics = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ], dtype=np.float64)
        
        # Camera extrinsics (view matrix as 4x4, column-major → standard row-major)
        view_4x4 = np.array(view_matrix, dtype=np.float64).reshape(4, 4, order='F')
        proj_4x4 = proj_np
        
        return CapturedImage(
            rgb=rgb,
            depth=depth,
            segmentation=segmentation,
            camera_intrinsics=intrinsics,
            camera_extrinsics=view_4x4,
            view_matrix=view_4x4,
            projection_matrix=proj_4x4
        )
    
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
        
        # Save depth as 16-bit PNG (millimetres)
        depth_mm = (image_data.depth * 1000).astype(np.uint16)
        depth_path = os.path.join(save_dir, f"{prefix}_depth.png")
        cv2.imwrite(depth_path, depth_mm)
        paths["depth"] = depth_path
        
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
        max_steps: int = 300
    ) -> bool:
        """
        Move the robot end-effector to a target pose using IK.
        
        Args:
            target_position: (x, y, z) target position
            target_orientation: quaternion (x, y, z, w); if None, uses top-down
            max_steps: maximum simulation steps for the motion
        
        Returns:
            True if target was reached within tolerance
        """
        if target_orientation is None:
            # Default: top-down grasp approach
            target_orientation = p.getQuaternionFromEuler([np.pi, 0, 0])
        
        # Compute IK solution — seeded with home rest poses for better convergence
        joint_positions = p.calculateInverseKinematics(
            self.robot_id,
            self.config.end_effector_index,
            list(target_position),
            list(target_orientation),
            restPoses=list(self.config.home_joint_positions),
            maxNumIterations=100,
            residualThreshold=1e-5
        )
        
        # Apply joint positions with position control
        for i, pos in enumerate(joint_positions[:6]):
            p.setJointMotorControl2(
                self.robot_id, i, p.POSITION_CONTROL,
                targetPosition=pos,
                force=240,
                maxVelocity=1.0
            )
        
        # Step simulation
        for step in range(max_steps):
            p.stepSimulation()
            if self.gui:
                time.sleep(self.config.time_step)
            
            # Check convergence
            ee_state = p.getLinkState(self.robot_id, self.config.end_effector_index)
            ee_pos = np.array(ee_state[4])
            error = np.linalg.norm(ee_pos - target_position)
            if error < 0.005:
                return True
        
        return False
    
    def execute_grasp(
        self,
        grasp_position: np.ndarray,
        grasp_orientation: np.ndarray,
        pre_grasp_height: float = 0.15
    ) -> bool:
        """
        Execute a grasp sequence: approach -> descend -> close gripper -> lift.
        
        Args:
            grasp_position: (x, y, z) grasp point
            grasp_orientation: quaternion for gripper orientation
            pre_grasp_height: height above grasp point for approach
        
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
        
        # 3. Close gripper (simplified - constraints-based)
        print("[SimEnv] Closing gripper...")
        # In a full implementation, this would actuate gripper joints
        # For simplicity, we create a fixed constraint to "grasp" the nearest object
        grasped_id = self._find_nearest_object(grasp_position)
        constraint_id = None
        if grasped_id is not None:
            constraint_id = p.createConstraint(
                self.robot_id, self.config.end_effector_index,
                grasped_id, -1,
                p.JOINT_FIXED, [0, 0, 0], [0, 0, 0.05], [0, 0, 0]
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
                # Stability check: simulate 50 more steps, verify drift < 2cm
                pre_pos = np.array(obj_pos)
                for _ in range(50):
                    p.stepSimulation()
                post_pos, _ = p.getBasePositionAndOrientation(grasped_id)
                drift = np.linalg.norm(np.array(post_pos) - pre_pos)
                stable = drift < 0.02
                
                # Also check persistent contact with gripper
                contacts = p.getContactPoints(self.robot_id, grasped_id)
                gripping = len(contacts) > 0
                
                if stable and gripping:
                    print(f"[SimEnv] ✓ Grasp successful! "
                          f"(lifted {obj_pos[2] - table_z:.3f}m, drift {drift:.4f}m)")
                    return True
                else:
                    print(f"[SimEnv] ✗ Grasp unstable "
                          f"(drift={drift:.4f}m, contact={gripping})")
                    return False
        
        print("[SimEnv] ✗ Grasp failed.")
        return False
    
    def _find_nearest_object(self, position: np.ndarray, max_dist: float = 0.05) -> Optional[int]:
        """Find the nearest object to a given position."""
        nearest_id = None
        nearest_dist = max_dist
        
        for obj in self.objects:
            obj_pos, _ = p.getBasePositionAndOrientation(obj.body_id)
            dist = np.linalg.norm(np.array(obj_pos) - position)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_id = obj.body_id
        
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