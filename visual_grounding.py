"""
Visual Affordance Grounding Module
===================================
Grounds the VLM's affordance reasoning results into pixel-level masks using 
visual grounding (GroundingDINO + SAM). This module takes the object and part 
names from affordance reasoning and produces segmentation masks that identify 
the exact affordance region in the image.

The paper uses LangSAM (Language Segment Anything Model) which combines:
- GroundingDINO for open-vocabulary object detection (text -> bounding box)
- SAM (Segment Anything Model) for high-quality segmentation (box -> mask)

For environments without GPU or full LangSAM installation, a simulation-based
fallback using PyBullet's segmentation masks is provided.

Reference: AffordGrasp, Section III-B: Visual Affordance Grounding
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .config import VisualGroundingConfig


@dataclass
class GroundingResult:
    """Output of the visual grounding module."""
    object_mask: np.ndarray          # (H, W) binary mask of the full target object
    affordance_mask: np.ndarray      # (H, W) binary mask of the graspable part
    object_bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) bounding box
    affordance_bbox: Optional[Tuple[int, int, int, int]]
    affordance_center: Tuple[int, int]  # (cx, cy) pixel center of affordance region
    affordance_center_3d: Optional[np.ndarray]  # (x, y, z) in world coordinates
    confidence: float
    object_label: str
    part_label: str


class VisualAffordanceGrounder:
    """
    Grounds affordance reasoning into pixel-level masks.
    
    Pipeline:
    1. Use GroundingDINO to detect the target object given its text description
    2. Use SAM to segment the detected object into a precise mask
    3. Use GroundingDINO again to locate the specific part (e.g., "handle")
    4. Intersect part detection with object mask to get affordance region
    5. Compute affordance center for grasp pose generation
    
    Usage:
        config = VisualGroundingConfig()
        grounder = VisualAffordanceGrounder(config)
        result = grounder.ground(
            rgb_image=scene_rgb,
            depth_image=scene_depth,
            target_object="wooden spoon",
            target_part="handle",
            camera_intrinsics=K
        )
        affordance_mask = result.affordance_mask
    """
    
    def __init__(self, config: VisualGroundingConfig):
        self.config = config
        self.lang_sam = None
        self.use_langsam = False
        
        # Try to load LangSAM
        try:
            from lang_sam import LangSAM
            self.lang_sam = LangSAM()
            self.use_langsam = True
            print("[VisualGrounder] LangSAM loaded successfully.")
        except ImportError:
            print("[VisualGrounder] LangSAM not available.")
            print("  Install with: pip install lang-sam")
            print("  Using simulation segmentation fallback.")
    
    def ground(
        self,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        target_object: str,
        target_part: str,
        camera_intrinsics: Optional[np.ndarray] = None,
        camera_extrinsics: Optional[np.ndarray] = None,
        simulation_segmask: Optional[np.ndarray] = None,
        target_body_id: Optional[int] = None
    ) -> GroundingResult:
        """
        Ground object and part affordances into pixel masks.
        
        Args:
            rgb_image: (H, W, 3) RGB image
            depth_image: (H, W) depth in metres
            target_object: Name of the target object (from VLM reasoning)
            target_part: Name of the graspable part (from VLM reasoning)
            camera_intrinsics: (3, 3) camera intrinsic matrix
            camera_extrinsics: (4, 4) camera extrinsic matrix
            simulation_segmask: PyBullet segmentation mask (fallback)
            target_body_id: PyBullet body ID of target object (fallback)
        
        Returns:
            GroundingResult with object and affordance masks
        """
        h, w = rgb_image.shape[:2]
        
        if self.use_langsam:
            result = self._ground_with_langsam(
                rgb_image, depth_image, target_object, target_part,
                camera_intrinsics, camera_extrinsics
            )
        elif simulation_segmask is not None and target_body_id is not None:
            result = self._ground_with_simulation(
                rgb_image, depth_image, simulation_segmask,
                target_body_id, target_object, target_part,
                camera_intrinsics, camera_extrinsics
            )
        else:
            # Fallback: use simple colour-based or region-based grounding
            result = self._ground_heuristic(
                rgb_image, depth_image, target_object, target_part,
                camera_intrinsics, camera_extrinsics
            )
        
        return result
    
    # =========================================================================
    #  Method 1: LangSAM-based Grounding (Paper's approach)
    # =========================================================================
    
    def _ground_with_langsam(
        self,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        target_object: str,
        target_part: str,
        camera_intrinsics: Optional[np.ndarray],
        camera_extrinsics: Optional[np.ndarray]
    ) -> GroundingResult:
        """
        Full LangSAM-based grounding matching the paper's two-pass approach.
        
        Pass 1: Detect and segment the full target object.
        Pass 2: Crop to the object bounding box (with padding), then detect
                 the specific part within the crop. This isolates the part
                 query from clutter distractors.
        
        Supports both LangSAM v0.1.x (single-image API) and v0.2.x (batched API).
        """
        from PIL import Image
        
        pil_image = Image.fromarray(rgb_image)
        h, w = rgb_image.shape[:2]
        pad = self.config.crop_padding
        
        # ---- Pass 1: Segment the full target object ----
        # Use a short noun phrase with trailing period for best GroundingDINO results
        object_query = f"{target_object}."
        obj_result = self._langsam_predict(pil_image, object_query)
        
        if obj_result is None or len(obj_result["masks"]) == 0:
            print(f"[VisualGrounder] Warning: No object found for '{target_object}'")
            return self._empty_result(h, w, target_object, target_part)
        
        # Take the highest-confidence detection
        best_idx = obj_result["scores"].argmax()
        object_mask = obj_result["masks"][best_idx].astype(np.uint8)
        obj_box_raw = obj_result["boxes"][best_idx]
        
        # Normalised boxes need denormalisation
        if obj_box_raw.max() <= 1.0:
            obj_box = [
                int(obj_box_raw[0] * w), int(obj_box_raw[1] * h),
                int(obj_box_raw[2] * w), int(obj_box_raw[3] * h)
            ]
        else:
            obj_box = [int(v) for v in obj_box_raw]
        
        # ---- Pass 2: Crop and segment the part ----
        # Crop to object bounding box with padding (research: 20px padding)
        x1 = max(0, obj_box[0] - pad)
        y1 = max(0, obj_box[1] - pad)
        x2 = min(w, obj_box[2] + pad)
        y2 = min(h, obj_box[3] + pad)
        cropped = pil_image.crop((x1, y1, x2, y2))
        
        # Use short noun phrase for part — "handle.", "blade.", "rim."
        part_query = f"{target_part}."
        part_result = self._langsam_predict(cropped, part_query)
        
        if part_result is not None and len(part_result["masks"]) > 0:
            best_part_idx = part_result["scores"].argmax()
            part_mask_crop = part_result["masks"][best_part_idx].astype(np.uint8)
            
            # Map the cropped part mask back to full image coordinates
            part_mask_full = np.zeros((h, w), dtype=np.uint8)
            part_mask_full[y1:y2, x1:x2] = part_mask_crop
            
            # Intersect with object mask to ensure part is within the object
            affordance_mask = (part_mask_full & object_mask).astype(np.uint8)
            
            if affordance_mask.sum() < self.config.min_mask_area:
                # Part mask too small after intersection — fall back to heuristic
                print(f"[VisualGrounder] Part mask too small after intersection, "
                      f"using heuristic for '{target_part}'")
                affordance_mask = self._estimate_part_region(
                    object_mask, target_part
                )
        else:
            # No part found — estimate from object geometry
            print(f"[VisualGrounder] No part detected for '{target_part}', "
                  f"using heuristic estimate")
            affordance_mask = self._estimate_part_region(
                object_mask, target_part
            )
        
        # Compute affordance center and bbox
        aff_center = self._compute_mask_center(affordance_mask)
        aff_bbox = self._mask_to_bbox(affordance_mask)
        
        # Compute 3D affordance center
        center_3d = None
        if camera_intrinsics is not None and aff_center is not None:
            center_3d = self._pixel_to_3d(
                aff_center, depth_image, camera_intrinsics, camera_extrinsics
            )
        
        return GroundingResult(
            object_mask=object_mask,
            affordance_mask=affordance_mask,
            object_bbox=tuple(obj_box),
            affordance_bbox=aff_bbox,
            affordance_center=aff_center if aff_center else (w // 2, h // 2),
            affordance_center_3d=center_3d,
            confidence=float(obj_result["scores"][best_idx]),
            object_label=target_object,
            part_label=target_part
        )
    
    def _langsam_predict(self, pil_image, text_prompt: str) -> Optional[dict]:
        """
        Wrapper for LangSAM predict that handles both v0.1.x and v0.2.x APIs.
        
        Returns dict with keys: masks (N,H,W np.array), boxes (N,4), scores (N,).
        """
        import torch
        
        try:
            # Try v0.2.x batched API first
            results = self.lang_sam.predict([pil_image], [text_prompt])
            if isinstance(results, list) and len(results) > 0:
                r = results[0]
                masks_t = r.get("masks", r.get("segmentation", torch.tensor([])))
                boxes_t = r.get("boxes", torch.tensor([]))
                scores_t = r.get("scores", r.get("logits", torch.tensor([])))
                
                if isinstance(masks_t, torch.Tensor) and masks_t.numel() > 0:
                    return {
                        "masks": masks_t.cpu().numpy(),
                        "boxes": boxes_t.cpu().numpy(),
                        "scores": scores_t.cpu().numpy()
                    }
        except TypeError:
            pass  # v0.2.x API not supported, fall through
        except Exception:
            pass
        
        try:
            # Fall back to v0.1.x single-image API
            masks, boxes, phrases, logits = self.lang_sam.predict(
                pil_image, text_prompt
            )
            if isinstance(masks, torch.Tensor) and masks.numel() > 0:
                return {
                    "masks": masks.cpu().numpy(),
                    "boxes": boxes.cpu().numpy(),
                    "scores": logits.cpu().numpy()
                }
        except Exception as e:
            print(f"[VisualGrounder] LangSAM predict failed: {e}")
        
        return None
    
    # =========================================================================
    #  Method 2: Simulation Segmentation Fallback
    # =========================================================================
    
    def _ground_with_simulation(
        self,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        segmentation_mask: np.ndarray,
        target_body_id: int,
        target_object: str,
        target_part: str,
        camera_intrinsics: Optional[np.ndarray],
        camera_extrinsics: Optional[np.ndarray]
    ) -> GroundingResult:
        """
        Use PyBullet's built-in segmentation mask as ground truth.
        This is the recommended approach for simulation experiments as it
        provides perfect object segmentation without requiring LangSAM.
        """
        h, w = rgb_image.shape[:2]
        
        # Extract object mask from segmentation
        object_mask = (segmentation_mask == target_body_id).astype(np.uint8)
        
        if object_mask.sum() < self.config.min_mask_area:
            print(f"[VisualGrounder] Warning: Object {target_body_id} not visible.")
            return self._empty_result(h, w, target_object, target_part)
        
        # Estimate part region heuristically based on object shape
        affordance_mask = self._estimate_part_region(object_mask, target_part)
        
        # Compute centres and bounding boxes
        aff_center = self._compute_mask_center(affordance_mask)
        obj_bbox = self._mask_to_bbox(object_mask)
        aff_bbox = self._mask_to_bbox(affordance_mask)
        
        # 3D centre
        center_3d = None
        if camera_intrinsics is not None and aff_center is not None:
            center_3d = self._pixel_to_3d(
                aff_center, depth_image, camera_intrinsics, camera_extrinsics
            )
        
        return GroundingResult(
            object_mask=object_mask,
            affordance_mask=affordance_mask,
            object_bbox=obj_bbox if obj_bbox else (0, 0, w, h),
            affordance_bbox=aff_bbox,
            affordance_center=aff_center if aff_center else (w // 2, h // 2),
            affordance_center_3d=center_3d,
            confidence=1.0,  # Perfect segmentation in simulation
            object_label=target_object,
            part_label=target_part
        )
    
    # =========================================================================
    #  Method 3: Heuristic Fallback
    # =========================================================================
    
    def _ground_heuristic(
        self,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        target_object: str,
        target_part: str,
        camera_intrinsics: Optional[np.ndarray],
        camera_extrinsics: Optional[np.ndarray]
    ) -> GroundingResult:
        """
        Simple heuristic grounding based on image centre.
        Used when neither LangSAM nor simulation segmentation is available.
        """
        h, w = rgb_image.shape[:2]
        
        # Create a simple central mask
        cy, cx = h // 2, w // 2
        y_range = slice(cy - h // 6, cy + h // 6)
        x_range = slice(cx - w // 6, cx + w // 6)
        
        object_mask = np.zeros((h, w), dtype=np.uint8)
        object_mask[y_range, x_range] = 1
        
        affordance_mask = self._estimate_part_region(object_mask, target_part)
        aff_center = self._compute_mask_center(affordance_mask)
        
        center_3d = None
        if camera_intrinsics is not None and aff_center is not None:
            center_3d = self._pixel_to_3d(
                aff_center, depth_image, camera_intrinsics, camera_extrinsics
            )
        
        return GroundingResult(
            object_mask=object_mask,
            affordance_mask=affordance_mask,
            object_bbox=self._mask_to_bbox(object_mask) or (0, 0, w, h),
            affordance_bbox=self._mask_to_bbox(affordance_mask),
            affordance_center=aff_center if aff_center else (cx, cy),
            affordance_center_3d=center_3d,
            confidence=0.5,
            object_label=target_object,
            part_label=target_part
        )
    
    # =========================================================================
    #  UTILITY METHODS
    # =========================================================================
    
    def _estimate_part_region(
        self, object_mask: np.ndarray, part_name: str
    ) -> np.ndarray:
        """
        Heuristically estimate a part's region within an object mask.
        
        For tools (spoon, hammer, knife, etc.), the "handle" is typically
        the elongated portion. For cups/mugs, the "handle" is on the side.
        
        This is a simplified approximation - the real AffordGrasp uses
        LangSAM to directly segment the part.
        """
        part_lower = part_name.lower()
        mask = object_mask.copy()
        h, w = mask.shape
        
        # Find object bounding box
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            return mask
        
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        obj_h = y_max - y_min
        obj_w = x_max - x_min
        
        affordance_mask = np.zeros_like(mask)
        
        if "handle" in part_lower:
            # For elongated objects: handle is typically one end
            # For vertical objects: handle is the bottom half
            # For horizontal objects: handle is one side
            if obj_h > obj_w:
                # Vertical orientation - handle is bottom portion
                handle_start = y_min + int(obj_h * 0.5)
                affordance_mask[handle_start:y_max, x_min:x_max] = 1
            else:
                # Horizontal orientation - handle is right portion
                handle_start = x_min + int(obj_w * 0.5)
                affordance_mask[y_min:y_max, handle_start:x_max] = 1
        elif "body" in part_lower or "surface" in part_lower:
            # Central region of the object
            cy = (y_min + y_max) // 2
            cx = (x_min + x_max) // 2
            r_y = int(obj_h * 0.3)
            r_x = int(obj_w * 0.3)
            affordance_mask[cy - r_y:cy + r_y, cx - r_x:cx + r_x] = 1
        elif "rim" in part_lower or "top" in part_lower:
            # Top portion
            top_end = y_min + int(obj_h * 0.3)
            affordance_mask[y_min:top_end, x_min:x_max] = 1
        elif "blade" in part_lower or "head" in part_lower or "tip" in part_lower:
            # Top/front portion (opposite of handle)
            if obj_h > obj_w:
                tip_end = y_min + int(obj_h * 0.4)
                affordance_mask[y_min:tip_end, x_min:x_max] = 1
            else:
                tip_end = x_min + int(obj_w * 0.4)
                affordance_mask[y_min:y_max, x_min:tip_end] = 1
        else:
            # Default: use entire object mask
            affordance_mask = mask.copy()
        
        # Intersect with object mask
        affordance_mask = (affordance_mask & mask).astype(np.uint8)
        
        # Ensure minimum area
        if affordance_mask.sum() < self.config.min_mask_area:
            affordance_mask = mask.copy()
        
        return affordance_mask
    
    def _compute_mask_center(self, mask: np.ndarray) -> Optional[Tuple[int, int]]:
        """Compute the centroid of a binary mask."""
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            return None
        cy = int(np.mean(coords[0]))
        cx = int(np.mean(coords[1]))
        return (cx, cy)
    
    def _mask_to_bbox(
        self, mask: np.ndarray
    ) -> Optional[Tuple[int, int, int, int]]:
        """Convert binary mask to bounding box (x1, y1, x2, y2)."""
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            return None
        y1, y2 = int(coords[0].min()), int(coords[0].max())
        x1, x2 = int(coords[1].min()), int(coords[1].max())
        return (x1, y1, x2, y2)
    
    def _pixel_to_3d(
        self,
        pixel: Tuple[int, int],
        depth: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        Convert a pixel coordinate + depth to a 3D world coordinate.
        
        Applies the OpenGL camera convention flip (y-down → y-up, z-forward → z-back)
        to match PyBullet's view matrix convention before transforming to world frame.
        """
        cx_px, cy_px = pixel
        
        # Clamp to image bounds
        h, w = depth.shape
        cy_px = max(0, min(cy_px, h - 1))
        cx_px = max(0, min(cx_px, w - 1))
        
        z = float(depth[cy_px, cx_px])
        if z <= 0 or z > 5.0:
            return None
        
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        px, py = intrinsics[0, 2], intrinsics[1, 2]
        
        # Back-project to image-convention camera frame
        x_img = (cx_px - px) * z / fx
        y_img = (cy_px - py) * z / fy
        
        # Convert to OpenGL camera frame (PyBullet view matrix convention)
        x_cam = x_img
        y_cam = -y_img   # flip Y
        z_cam = -z        # flip Z
        
        point_cam = np.array([x_cam, y_cam, z_cam, 1.0])
        
        # Transform to world frame
        if extrinsics is not None:
            extrinsics_inv = np.linalg.inv(extrinsics)
            point_world = extrinsics_inv @ point_cam
            return point_world[:3]
        
        return point_cam[:3]
    
    def compute_affordance_center_3d_from_mask(
        self,
        affordance_mask: np.ndarray,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        Compute the 3D affordance centre as the mean of all masked 3D points.
        
        This is more robust than single-pixel back-projection because it
        averages over the entire affordance region, reducing noise from
        any single depth measurement.
        """
        h, w = depth.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        valid = (affordance_mask > 0) & (depth > 0.01) & (depth < 5.0)
        if valid.sum() == 0:
            return None
        
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        px, py = intrinsics[0, 2], intrinsics[1, 2]
        
        z_vals = depth[valid]
        x_img = (u[valid] - px) * z_vals / fx
        y_img = (v[valid] - py) * z_vals / fy
        
        # OpenGL convention flip
        points_cam = np.stack([x_img, -y_img, -z_vals, np.ones_like(z_vals)], axis=-1)
        
        if extrinsics is not None:
            extrinsics_inv = np.linalg.inv(extrinsics)
            points_world = (extrinsics_inv @ points_cam.T).T[:, :3]
        else:
            points_world = points_cam[:, :3]
        
        return points_world.mean(axis=0)
    
    def _empty_result(
        self, h: int, w: int, object_label: str, part_label: str
    ) -> GroundingResult:
        """Return an empty grounding result when detection fails."""
        return GroundingResult(
            object_mask=np.zeros((h, w), dtype=np.uint8),
            affordance_mask=np.zeros((h, w), dtype=np.uint8),
            object_bbox=(0, 0, 0, 0),
            affordance_bbox=None,
            affordance_center=(w // 2, h // 2),
            affordance_center_3d=None,
            confidence=0.0,
            object_label=object_label,
            part_label=part_label
        )
    
    # =========================================================================
    #  VISUALISATION
    # =========================================================================
    
    def visualize_grounding(
        self,
        rgb_image: np.ndarray,
        result: GroundingResult,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Create a visualisation of the grounding results overlaid on the RGB image.
        Shows object mask (blue), affordance mask (red), and affordance center (star).
        """
        import cv2
        
        vis = rgb_image.copy()
        h, w = vis.shape[:2]
        
        # Overlay object mask in blue (semi-transparent)
        obj_overlay = np.zeros_like(vis)
        obj_overlay[:, :, 0] = result.object_mask * 180  # blue channel
        vis = cv2.addWeighted(vis, 0.7, obj_overlay, 0.3, 0)
        
        # Overlay affordance mask in red
        aff_overlay = np.zeros_like(vis)
        aff_overlay[:, :, 2] = result.affordance_mask * 200  # red channel
        vis = cv2.addWeighted(vis, 0.7, aff_overlay, 0.3, 0)
        
        # Draw affordance center as a star marker
        cx, cy = result.affordance_center
        cv2.drawMarker(
            vis, (cx, cy), color=(0, 255, 0),
            markerType=cv2.MARKER_STAR, markerSize=15, thickness=2
        )
        
        # Draw bounding boxes
        if result.object_bbox and result.object_bbox != (0, 0, 0, 0):
            x1, y1, x2, y2 = result.object_bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 200, 0), 2)
        
        if result.affordance_bbox:
            x1, y1, x2, y2 = result.affordance_bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Labels
        cv2.putText(
            vis, f"Object: {result.object_label}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
        cv2.putText(
            vis, f"Affordance: {result.part_label} (conf: {result.confidence:.2f})",
            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
        
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            print(f"[VisualGrounder] Visualisation saved to {save_path}")
        
        return vis