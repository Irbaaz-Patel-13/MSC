"""
Task-Oriented Grasp Generation Module
======================================
Generates 6-DoF grasp poses within the affordance region identified by the
visual grounding module.

Primary method: Contact-GraspNet (PyTorch port by elchun) — open-source 6-DoF
grasp detection from depth/point cloud input, no license required.
Fallback: Antipodal grasp sampling on the point cloud.

The key AffordGrasp insight is the ranking formula:
    ranking(g) = score(g) / ||t(g) - c||₂
where t(g) is the grasp translation, c is the 3D affordance centre, and
score(g) is the grasp confidence. This naturally biases selection toward
high-quality grasps near the task-relevant part.

Reference: AffordGrasp, Section III-C: Task-Oriented Grasp Generation
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from scipy.spatial.transform import Rotation

from config import GraspConfig


@dataclass
class GraspPose:
    """A 6-DoF grasp pose with quality metrics."""
    position: np.ndarray        # (3,) grasp centre position in world frame
    orientation: np.ndarray     # (4,) quaternion (x, y, z, w) 
    rotation_matrix: np.ndarray # (3, 3) rotation matrix
    width: float                # gripper opening width in metres
    quality_score: float        # grasp quality (antipodal score, etc.)
    affordance_score: float     # proximity to affordance centre
    combined_score: float       # weighted combination
    approach_direction: np.ndarray  # (3,) approach vector
    
    def as_4x4(self) -> np.ndarray:
        """Return the grasp pose as a 4x4 homogeneous transformation matrix."""
        T = np.eye(4)
        T[:3, :3] = self.rotation_matrix
        T[:3, 3] = self.position
        return T


class AffordanceGraspGenerator:
    """
    Generates task-oriented grasp poses guided by affordance information.
    
    Primary: Contact-GraspNet (PyTorch port) for 6-DoF grasp detection.
    Fallback: Antipodal grasp sampling on the point cloud.
    
    Both methods are followed by the AffordGrasp ranking formula:
        ranking(g) = score(g) / max(||t(g) - c||₂, ε)
    
    Usage:
        config = GraspConfig()
        generator = AffordanceGraspGenerator(config)
        grasps = generator.generate(
            point_cloud=points,
            affordance_center_3d=center
        )
    """
    
    def __init__(self, config: GraspConfig):
        self.config = config
        self.rng = np.random.RandomState(42)
        self.cgn_model = None
        
        # Try to load Contact-GraspNet if configured
        if config.method == "contact_graspnet":
            self._try_load_contact_graspnet()
    
    def _try_load_contact_graspnet(self) -> None:
        """Attempt to load Contact-GraspNet PyTorch model."""
        # --- Step 1: check package is on sys.path without importing it ---
        import importlib.util
        if importlib.util.find_spec("contact_graspnet_pytorch") is None:
            print("[GraspGen] contact_graspnet_pytorch not found on sys.path.")
            print("  Install:")
            print("    git clone https://github.com/elchun/contact_graspnet_pytorch.git")
            print("    cd contact_graspnet_pytorch && pip install -e .")
            print("")
            print("WARNING: Using antipodal sampling fallback. "
                  "Grasp quality will be significantly lower than Contact-GraspNet.")
            self.cgn_model = None
            return

        # --- Step 2: import the actual classes (GraspEstimator, not ContactGraspNet) ---
        try:
            from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
            from contact_graspnet_pytorch import config_utils
            from contact_graspnet_pytorch.checkpoints import CheckpointIO
        except ImportError as e:
            print(f"[GraspGen] contact_graspnet_pytorch sub-module missing: {e}")
            print("  Try reinstalling: cd contact_graspnet_pytorch && pip install -e .")
            print("")
            print("WARNING: Using antipodal sampling fallback. "
                  "Grasp quality will be significantly lower than Contact-GraspNet.")
            self.cgn_model = None
            return

        # --- Step 3: load config + model + checkpoint ---
        # Expected layout for cgn_checkpoint_dir:
        #   <ckpt_dir>/config.yaml         ← shipped with the repo clone
        #   <ckpt_dir>/checkpoints/model.pt ← download from repo README
        try:
            import os

            ckpt_dir = self.config.cgn_checkpoint_dir
            global_config = config_utils.load_config(ckpt_dir)
            grasp_estimator = GraspEstimator(global_config)

            model_ckpt_dir = os.path.join(ckpt_dir, "checkpoints")
            model_pt = os.path.join(model_ckpt_dir, "model.pt")
            if not os.path.isfile(model_pt):
                print(f"[GraspGen] No checkpoint at {model_pt}")
                print("  Download the pretrained weights from the elchun/"
                      "contact_graspnet_pytorch README (Google Drive link)")
                print("  and place model.pt in: " + model_ckpt_dir)
                print("")
                print("WARNING: Using antipodal sampling fallback. "
                      "Grasp quality will be significantly lower than Contact-GraspNet.")
                self.cgn_model = None
                return

            checkpoint_io = CheckpointIO(
                checkpoint_dir=model_ckpt_dir, model=grasp_estimator.model
            )
            checkpoint_io.load("model.pt")
            grasp_estimator.model.eval()

            self.cgn_model = grasp_estimator
            print(f"[GraspGen] Contact-GraspNet loaded from {model_pt}")

        except Exception as e:
            print(f"[GraspGen] Contact-GraspNet load failed: {e}")
            print("")
            print("WARNING: Using antipodal sampling fallback. "
                  "Grasp quality will be significantly lower than Contact-GraspNet.")
            self.cgn_model = None
    
    def generate(
        self,
        point_cloud: np.ndarray,
        affordance_center_3d: np.ndarray,
        affordance_points: Optional[np.ndarray] = None,
        surface_normals: Optional[np.ndarray] = None,
        table_height: float = 0.0,
        depth_image: Optional[np.ndarray] = None,
        camera_intrinsics: Optional[np.ndarray] = None,
        segmentation_mask: Optional[np.ndarray] = None,
    ) -> List[GraspPose]:
        """
        Generate task-oriented grasp poses.
        
        Args:
            point_cloud: (N, 3) full scene point cloud
            affordance_center_3d: (3,) 3D affordance centre from visual grounding
            affordance_points: (M, 3) point cloud of the affordance region only
            surface_normals: (N, 3) surface normals (estimated if None)
            table_height: Height of the table surface for collision filtering
            depth_image: (H, W) depth in metres (for Contact-GraspNet)
            camera_intrinsics: (3, 3) camera matrix K (for Contact-GraspNet)
            segmentation_mask: (H, W) affordance mask (for Contact-GraspNet)
        
        Returns:
            List of GraspPose objects sorted by affordance-aware ranking (best first)
        """
        # Use affordance points if available, else use full cloud
        grasp_cloud = affordance_points if affordance_points is not None else point_cloud
        
        if len(grasp_cloud) < 10:
            print("[GraspGen] Warning: Too few points for grasp generation.")
            return []
        
        # ---- Generate candidate grasps ----
        if self.cgn_model is not None and depth_image is not None:
            candidates = self._generate_contact_graspnet(
                depth_image, camera_intrinsics, segmentation_mask
            )
        else:
            # Fallback: antipodal sampling
            if surface_normals is None:
                surface_normals = self._estimate_normals(grasp_cloud)
            elif affordance_points is not None:
                surface_normals = self._estimate_normals(affordance_points)
            
            candidates = self._sample_antipodal_grasps(grasp_cloud, surface_normals)
            
            # Evaluate quality for antipodal grasps
            for grasp in candidates:
                grasp.quality_score = self._evaluate_quality(grasp, grasp_cloud)
        
        if len(candidates) == 0:
            print("[GraspGen] Warning: No valid grasp candidates generated.")
            return []
        
        # ---- AffordGrasp ranking: score / distance ----
        # This is the paper's core contribution to grasp selection.
        eps = self.config.min_distance_epsilon
        max_dist = self.config.max_distance_from_affordance_center
        
        for grasp in candidates:
            distance = np.linalg.norm(grasp.position - affordance_center_3d)
            grasp.affordance_score = distance  # store raw distance for logging
            
            if distance > max_dist:
                grasp.combined_score = 0.0  # hard cutoff
            else:
                # Paper formula: ranking = score / distance
                grasp.combined_score = grasp.quality_score / max(distance, eps)
        
        # ---- Filter and sort ----
        # Remove grasps below table
        candidates = [
            g for g in candidates if g.position[2] > table_height + 0.01
        ]
        # Remove zero-scored grasps (beyond max distance)
        candidates = [g for g in candidates if g.combined_score > 0.0]
        
        # Sort by AffordGrasp ranking (highest first)
        candidates.sort(key=lambda g: g.combined_score, reverse=True)
        
        # Return top-K
        top_grasps = candidates[:self.config.top_k_grasps]
        
        if top_grasps:
            print(f"[GraspGen] Generated {len(candidates)} candidates, "
                  f"returning top {len(top_grasps)}")
            best = top_grasps[0]
            print(f"  Best grasp: pos={best.position}, "
                  f"score={best.quality_score:.3f}, "
                  f"dist_to_aff={best.affordance_score:.4f}, "
                  f"ranking={best.combined_score:.3f}")
        
        return top_grasps
    
    # =========================================================================
    #  CONTACT-GRASPNET INFERENCE
    # =========================================================================
    
    def _generate_contact_graspnet(
        self,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        segmask: Optional[np.ndarray] = None
    ) -> List[GraspPose]:
        """
        Generate grasps using Contact-GraspNet (elchun/contact_graspnet_pytorch).

        The GraspEstimator API:
          1. extract_point_clouds(depth, K, segmap, z_range)
               → pc_full (N,3), pc_segments {seg_id: (M,3)}, pc_colors
          2. predict_scene_grasps(pc_full, pc_segments, local_regions, ...)
               → pred_grasps_cam {seg_id: (N,4,4)}, scores {seg_id: (N,)},
                  contact_pts, gripper_openings {seg_id: (N,)}

        segmask is passed as an integer segmap so the affordance region
        (pixel value = 1) becomes its own pc_segment for local-region inference.
        """
        try:
            z_range = list(self.config.cgn_z_range)

            # Integer segmap: 0 = background, 1 = affordance region
            segmap = segmask.astype(np.int32) if segmask is not None else None

            # Step 1: depth → point clouds
            pc_full, pc_segments, _ = self.cgn_model.extract_point_clouds(
                depth, intrinsics, segmap=segmap, z_range=z_range
            )

            # Step 2: grasp prediction
            pred_grasps_cam, scores, _, gripper_openings = \
                self.cgn_model.predict_scene_grasps(
                    pc_full,
                    pc_segments=pc_segments,
                    local_regions=self.config.cgn_local_regions,
                    filter_grasps=self.config.cgn_filter_grasps,
                    forward_passes=self.config.cgn_forward_passes
                )

            # Step 3: convert to GraspPose objects
            candidates = []
            for seg_id, poses_raw in pred_grasps_cam.items():
                if poses_raw is None or len(poses_raw) == 0:
                    continue

                poses_4x4 = np.array(poses_raw)           # (N, 4, 4)
                seg_scores = np.atleast_1d(
                    np.array(scores.get(seg_id, []))
                )
                seg_widths = np.atleast_1d(
                    np.array(gripper_openings.get(seg_id, []))
                )

                for i in range(len(poses_4x4)):
                    T = poses_4x4[i]
                    rot = T[:3, :3]
                    pos = T[:3, 3].copy()
                    try:
                        quat = Rotation.from_matrix(rot).as_quat()
                    except Exception:
                        continue  # degenerate rotation matrix

                    width = float(seg_widths[i]) if i < len(seg_widths) else 0.08
                    score = float(seg_scores[i]) if i < len(seg_scores) else 0.0

                    candidates.append(GraspPose(
                        position=pos,
                        orientation=quat,
                        rotation_matrix=rot,
                        width=width,
                        quality_score=score,
                        affordance_score=0.0,
                        combined_score=0.0,
                        approach_direction=rot[:, 2]
                    ))

            print(f"[GraspGen] Contact-GraspNet produced {len(candidates)} grasps")
            return candidates

        except Exception as e:
            import traceback
            print(f"[GraspGen] Contact-GraspNet inference failed: {e}")
            traceback.print_exc()
            print("  Falling back to antipodal sampling on available point cloud.")
            return []
    
    # =========================================================================
    #  ANTIPODAL GRASP SAMPLING
    # =========================================================================
    
    def _sample_antipodal_grasps(
        self,
        points: np.ndarray,
        normals: np.ndarray
    ) -> List[GraspPose]:
        """
        Sample antipodal grasp candidates on the point cloud.
        
        For each sampled contact point, we:
        1. Sample a contact point on the surface
        2. Use the surface normal as the grasp approach direction
        3. Sample a second contact point on the opposite side
        4. Compute the grasp frame (position, orientation, width)
        """
        n_samples = self.config.num_grasp_samples
        n_points = len(points)
        candidates = []
        
        for _ in range(n_samples):
            # Sample a contact point
            idx = self.rng.randint(n_points)
            contact1 = points[idx]
            normal1 = normals[idx]
            
            # Ensure normal points "upward-ish" (away from table)
            if normal1[2] < 0:
                normal1 = -normal1
            
            # Generate multiple approach angles
            for angle_deg in np.linspace(
                self.config.approach_angle_range[0],
                self.config.approach_angle_range[1],
                self.config.num_approach_angles
            ):
                angle_rad = np.deg2rad(angle_deg)
                
                # Compute approach direction (rotate normal around random axis)
                approach = self._rotate_approach(normal1, angle_rad)
                
                # Grasp centre is slightly above contact along approach
                grasp_center = contact1 + approach * self.config.grasp_depth
                
                # Compute grasp frame
                # z-axis: approach direction (into the surface)
                # x-axis: closing direction (perpendicular to approach)
                z_axis = -approach / (np.linalg.norm(approach) + 1e-8)
                
                # Choose x-axis perpendicular to z
                if abs(z_axis[0]) < 0.9:
                    x_axis = np.cross(z_axis, np.array([1, 0, 0]))
                else:
                    x_axis = np.cross(z_axis, np.array([0, 1, 0]))
                x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
                y_axis = np.cross(z_axis, x_axis)
                
                rotation = np.column_stack([x_axis, y_axis, z_axis])
                
                # Estimate grasp width from local point cloud
                width = self._estimate_grasp_width(
                    grasp_center, x_axis, points
                )
                
                if (self.config.grasp_width_range[0] <= width 
                    <= self.config.grasp_width_range[1]):
                    
                    quat = Rotation.from_matrix(rotation).as_quat()  # (x,y,z,w)
                    
                    candidates.append(GraspPose(
                        position=grasp_center,
                        orientation=quat,
                        rotation_matrix=rotation,
                        width=width,
                        quality_score=0.0,
                        affordance_score=0.0,
                        combined_score=0.0,
                        approach_direction=approach
                    ))
        
        return candidates
    
    def _rotate_approach(self, normal: np.ndarray, angle: float) -> np.ndarray:
        """Rotate approach direction around a random perpendicular axis."""
        # Create a random axis perpendicular to normal
        random_vec = self.rng.randn(3)
        perp_axis = np.cross(normal, random_vec)
        perp_norm = np.linalg.norm(perp_axis)
        if perp_norm < 1e-6:
            return normal
        perp_axis = perp_axis / perp_norm
        
        # Rodrigues rotation
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rotated = (normal * cos_a + 
                   np.cross(perp_axis, normal) * sin_a +
                   perp_axis * np.dot(perp_axis, normal) * (1 - cos_a))
        
        norm = np.linalg.norm(rotated)
        return rotated / norm if norm > 1e-6 else normal
    
    def _estimate_grasp_width(
        self,
        center: np.ndarray,
        closing_dir: np.ndarray,
        points: np.ndarray,
        search_radius: float = 0.05
    ) -> float:
        """Estimate grasp width based on object extent along closing direction."""
        # Find nearby points
        dists = np.linalg.norm(points - center, axis=1)
        nearby_mask = dists < search_radius
        nearby_points = points[nearby_mask]
        
        if len(nearby_points) < 2:
            return 0.04  # default width
        
        # Project onto closing direction
        projections = np.dot(nearby_points - center, closing_dir)
        width = projections.max() - projections.min() + 0.005  # small margin
        
        return float(np.clip(width, 0.01, 0.10))
    
    # =========================================================================
    #  GRASP QUALITY EVALUATION
    # =========================================================================
    
    def _evaluate_quality(
        self, grasp: GraspPose, points: np.ndarray
    ) -> float:
        """
        Evaluate grasp quality using a simplified force-closure metric.
        
        Considers:
        - Antipodal alignment of contact normals
        - Point density near grasp contacts
        - Grasp width (prefer tighter grasps for stability)
        """
        center = grasp.position
        closing_dir = grasp.rotation_matrix[:, 0]  # x-axis = closing direction
        half_width = grasp.width / 2.0
        
        # Find points near each finger
        contact1 = center + closing_dir * half_width
        contact2 = center - closing_dir * half_width
        
        contact_radius = 0.01  # 1cm contact patch
        
        points_near_c1 = np.linalg.norm(points - contact1, axis=1) < contact_radius
        points_near_c2 = np.linalg.norm(points - contact2, axis=1) < contact_radius
        
        density1 = points_near_c1.sum()
        density2 = points_near_c2.sum()
        
        # Quality components
        contact_score = min(density1, density2) / max(density1 + density2, 1)
        width_score = 1.0 - (grasp.width / self.config.grasp_width_range[1])
        
        # Combined quality
        quality = 0.6 * contact_score + 0.4 * width_score
        return float(np.clip(quality, 0.0, 1.0))
    
    # =========================================================================
    #  AFFORDANCE-GUIDED SCORING
    # =========================================================================
    
    def _compute_affordance_score(
        self,
        grasp: GraspPose,
        affordance_center: np.ndarray
    ) -> float:
        """
        Score a grasp by its proximity to the affordance centre.
        
        This is the key insight from AffordGrasp: grasps closer to the
        affordance centre (e.g., the handle of a mug) are preferred
        because they enable task-oriented manipulation.
        """
        distance = np.linalg.norm(grasp.position - affordance_center)
        max_dist = self.config.max_distance_from_affordance_center
        
        # Gaussian-like scoring: closer = higher score
        score = np.exp(-0.5 * (distance / max_dist) ** 2)
        
        return float(score)
    
    # =========================================================================
    #  SURFACE NORMAL ESTIMATION
    # =========================================================================
    
    def _estimate_normals(
        self, points: np.ndarray, k_neighbours: int = 20
    ) -> np.ndarray:
        """
        Estimate surface normals using PCA on local neighbourhoods.
        Falls back to vertical normals if too few points.
        """
        n = len(points)
        normals = np.zeros_like(points)
        
        if n < k_neighbours:
            # Default: all normals point up
            normals[:, 2] = 1.0
            return normals
        
        # Simple KNN-based normal estimation
        from scipy.spatial import KDTree
        tree = KDTree(points)
        
        for i in range(n):
            _, indices = tree.query(points[i], k=min(k_neighbours, n))
            neighbors = points[indices]
            
            # PCA: smallest eigenvector = surface normal
            centroid = neighbors.mean(axis=0)
            centered = neighbors - centroid
            cov = centered.T @ centered / len(centered)
            
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                normal = eigenvectors[:, 0]  # smallest eigenvalue
                
                # Orient normals to point upward
                if normal[2] < 0:
                    normal = -normal
                
                normals[i] = normal
            except np.linalg.LinAlgError:
                normals[i] = np.array([0, 0, 1])
        
        return normals
    
    # =========================================================================
    #  VISUALISATION
    # =========================================================================
    
    def visualize_grasps(
        self,
        point_cloud: np.ndarray,
        grasps: List[GraspPose],
        affordance_center: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualise grasp poses on the point cloud using matplotlib.
        Shows the point cloud, affordance centre, and grasp frames.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot point cloud (subsample for speed)
        if len(point_cloud) > 5000:
            indices = self.rng.choice(len(point_cloud), 5000, replace=False)
            pts = point_cloud[indices]
        else:
            pts = point_cloud
        
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c='lightblue', s=1, alpha=0.3, label='Scene')
        
        # Plot affordance centre
        ax.scatter(*affordance_center, c='red', s=200, marker='*',
                   label='Affordance Centre', zorder=5)
        
        # Plot grasp poses
        colors = plt.cm.RdYlGn(np.linspace(0, 1, len(grasps)))
        for i, grasp in enumerate(grasps):
            pos = grasp.position
            approach = grasp.approach_direction
            closing = grasp.rotation_matrix[:, 0]
            
            # Draw approach vector
            ax.quiver(pos[0], pos[1], pos[2],
                      approach[0], approach[1], approach[2],
                      length=0.03, color=colors[i], arrow_length_ratio=0.3)
            
            # Draw gripper fingers
            hw = grasp.width / 2
            f1 = pos + closing * hw
            f2 = pos - closing * hw
            ax.plot([f1[0], f2[0]], [f1[1], f2[1]], [f1[2], f2[2]],
                    c=colors[i], linewidth=2)
            
            ax.scatter(*pos, c=[colors[i]], s=50, marker='o')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'AffordGrasp: Top {len(grasps)} Task-Oriented Grasps')
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[GraspGen] Visualisation saved to {save_path}")
        
        plt.close()