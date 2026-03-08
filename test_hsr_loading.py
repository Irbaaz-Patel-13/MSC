"""
HSR PyBullet Loading Test
==========================
Standalone test script for the Toyota HSR integration.  Run this AFTER
running fix_hsr_urdf.py to generate hsrb4s_pybullet.urdf.

Usage:
    # Step 1 — generate the fixed URDF (once):
    python fix_hsr_urdf.py --validate

    # Step 2 — test the HSR in simulation:
    python test_hsr_loading.py

The script:
  1. Connects to PyBullet GUI
  2. Loads the fixed URDF and prints all joints with limits
  3. Moves the arm to its home pose
  4. Runs IK to reach a target position above the table
  5. Opens and closes the gripper
  6. Captures an RGB image from the head camera and saves it
  7. Keeps the GUI open for visual inspection (press Ctrl-C to exit)
"""

import os
import sys
import time
import numpy as np

import pybullet as p
import pybullet_data

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
URDF_PATH   = os.path.join(SCRIPT_DIR, "hsr_description", "robots", "hsrb4s_pybullet.urdf")
OUTPUT_IMG  = os.path.join(SCRIPT_DIR, "results", "hsr_head_camera_test.png")

# HSR arm joints (kinematic order)
ARM_JOINTS = [
    "arm_lift_joint",
    "arm_flex_joint",
    "arm_roll_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
]
GRIPPER_JOINT  = "hand_motor_joint"
FINGER_JOINTS  = ["hand_l_proximal_joint", "hand_r_proximal_joint"]
TORSO_JOINT    = "torso_lift_joint"
HEAD_JOINTS    = ["head_pan_joint", "head_tilt_joint"]
EE_JOINT       = "hand_palm_joint"        # child link = hand_palm_link
HEAD_CAM_JOINT = "head_rgbd_sensor_joint" # child link = head_rgbd_sensor_link

HOME = {
    "arm_lift_joint":    0.10,
    "arm_flex_joint":   -2.00,
    "arm_roll_joint":    0.00,
    "wrist_flex_joint": -1.50,
    "wrist_roll_joint":  0.00,
    "hand_motor_joint":  0.80,
    "head_pan_joint":    0.00,
    "head_tilt_joint":  -0.40,
}

JOINT_TYPE_NAMES = {
    p.JOINT_REVOLUTE:  "revolute",
    p.JOINT_PRISMATIC: "prismatic",
    p.JOINT_SPHERICAL: "spherical",
    p.JOINT_PLANAR:    "planar",
    p.JOINT_FIXED:     "fixed",
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def build_joint_map(robot_id):
    jmap = {}
    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        jmap[info[1].decode("utf-8")] = i
    return jmap


def print_joints(robot_id):
    n = p.getNumJoints(robot_id)
    print(f"\n  {n} joints total:")
    print(f"  {'idx':>4}  {'name':<38}  {'type':<10}  {'limits'}")
    print(f"  {'-'*70}")
    for i in range(n):
        info = p.getJointInfo(robot_id, i)
        name  = info[1].decode("utf-8")
        jtype = JOINT_TYPE_NAMES.get(info[2], str(info[2]))
        lo, hi = info[8], info[9]
        limits = f"[{lo:.3f}, {hi:.3f}]" if info[2] != p.JOINT_FIXED else "—"
        print(f"  {i:>4}  {name:<38}  {jtype:<10}  {limits}")


def set_joint(robot_id, jmap, name, value):
    idx = jmap.get(name)
    if idx is not None:
        p.resetJointState(robot_id, idx, value)


def drive_joint(robot_id, jmap, name, target, force=100, vel=0.5):
    idx = jmap.get(name)
    if idx is not None:
        p.setJointMotorControl2(
            robot_id, idx, p.POSITION_CONTROL,
            targetPosition=target, force=force, maxVelocity=vel,
        )


def set_gripper(robot_id, jmap, value, force=20):
    drive_joint(robot_id, jmap, GRIPPER_JOINT, value, force=force)
    for fname in FINGER_JOINTS:
        drive_joint(robot_id, jmap, fname, value * 0.5, force=force)


def step_sim(n, gui, dt):
    for _ in range(n):
        p.stepSimulation()
        if gui:
            time.sleep(dt)


def move_to_home(robot_id, jmap):
    print("\n  [1] Moving to home position …")
    for jname, jval in HOME.items():
        set_joint(robot_id, jmap, jname, jval)
    torso_idx = jmap.get(TORSO_JOINT)
    if torso_idx is not None:
        p.resetJointState(robot_id, torso_idx, HOME.get("arm_lift_joint", 0.0) * 0.5)
    print("     Home position set.")


def run_ik(robot_id, jmap, ee_idx, target_pos, target_orn):
    """Compute IK and apply to arm joints. Returns (reached, ik_result)."""
    n = p.getNumJoints(robot_id)
    lower, upper, ranges, rest = [], [], [], []
    for i in range(n):
        info = p.getJointInfo(robot_id, i)
        lo, hi = info[8], info[9]
        if info[2] == p.JOINT_FIXED or lo >= hi:
            lo, hi = -0.01, 0.01
        lower.append(lo); upper.append(hi)
        ranges.append(hi - lo)
        rest.append((lo + hi) / 2.0)

    # Seed rest poses from HOME config for better convergence
    for jname, jval in HOME.items():
        idx = jmap.get(jname)
        if idx is not None:
            rest[idx] = jval

    ik = p.calculateInverseKinematics(
        robot_id, ee_idx,
        list(target_pos), list(target_orn),
        lowerLimits=lower, upperLimits=upper,
        jointRanges=ranges, restPoses=rest,
        maxNumIterations=300,
        residualThreshold=1e-5,
    )

    # Apply arm joints
    arm_idxs = [jmap[j] for j in ARM_JOINTS if j in jmap]
    for j_idx in arm_idxs:
        if j_idx < len(ik):
            p.setJointMotorControl2(
                robot_id, j_idx, p.POSITION_CONTROL,
                targetPosition=ik[j_idx], force=100, maxVelocity=0.5,
            )

    # Sync torso mimic
    arm_lift_idx = jmap.get("arm_lift_joint")
    torso_idx    = jmap.get(TORSO_JOINT)
    if arm_lift_idx is not None and torso_idx is not None and arm_lift_idx < len(ik):
        p.setJointMotorControl2(
            robot_id, torso_idx, p.POSITION_CONTROL,
            targetPosition=ik[arm_lift_idx] * 0.5, force=100,
        )

    return ik


def capture_head_camera(robot_id, jmap, width=640, height=480):
    """Capture RGB from the HSR head camera using its current pose."""
    cam_idx = jmap.get(HEAD_CAM_JOINT)
    if cam_idx is None:
        print("  [!] head_rgbd_sensor_joint not found in URDF — skipping head camera.")
        return None

    link_state = p.getLinkState(robot_id, cam_idx, computeForwardKinematics=True)
    cam_pos = np.array(link_state[4])
    cam_orn = np.array(link_state[5])

    rot_mat = np.array(p.getMatrixFromQuaternion(cam_orn)).reshape(3, 3)
    forward = rot_mat @ np.array([1.0, 0.0, 0.0])
    up      = rot_mat @ np.array([0.0, 0.0, 1.0])
    target  = cam_pos + forward * 0.5

    view_mat = p.computeViewMatrix(
        cameraEyePosition=cam_pos.tolist(),
        cameraTargetPosition=target.tolist(),
        cameraUpVector=up.tolist(),
    )
    proj_mat = p.computeProjectionMatrixFOV(
        fov=60.0, aspect=width / height, nearVal=0.01, farVal=5.0,
    )
    _, _, rgb_pixels, _, _ = p.getCameraImage(
        width=width, height=height,
        viewMatrix=view_mat, projectionMatrix=proj_mat,
        renderer=p.ER_TINY_RENDERER,
        lightAmbientCoeff=0.8, lightDiffuseCoeff=0.8, shadow=0,
    )
    rgb = np.array(rgb_pixels, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
    return rgb


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not os.path.isfile(URDF_PATH):
        print(f"ERROR: URDF not found at {URDF_PATH}")
        print("Run `python fix_hsr_urdf.py` first to generate the PyBullet URDF.")
        sys.exit(1)

    gui = True
    client = p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 240.0)
    if gui:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

    # Ground plane + table
    p.loadURDF("plane.urdf")
    table_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3, 0.4, 0.01])
    table_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3, 0.4, 0.01],
                                    rgbaColor=[0.6, 0.5, 0.4, 1.0])
    p.createMultiBody(0, table_col, table_vis, basePosition=[0.5, 0.0, -0.01])

    # Add a coloured box on the table as a target object
    box_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.03, 0.03, 0.05])
    box_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.03, 0.03, 0.05],
                                  rgbaColor=[0.8, 0.2, 0.1, 1.0])
    p.createMultiBody(0.1, box_col, box_vis, basePosition=[0.45, 0.05, 0.05])

    # Load HSR
    print(f"\n  Loading HSR from {URDF_PATH} …")
    robot_id = p.loadURDF(URDF_PATH, basePosition=[0, 0, 0], useFixedBase=True)
    jmap = build_joint_map(robot_id)
    ee_idx = jmap.get(EE_JOINT, 0)

    print(f"  Loaded OK — EE link idx = {ee_idx}")
    print_joints(robot_id)

    dt = 1.0 / 240.0

    # ── Phase 1: Home position ────────────────────────────────────────────────
    move_to_home(robot_id, jmap)
    step_sim(200, gui, dt)

    # ── Phase 2: IK to reach above the table (pre-grasp) ─────────────────────
    TARGET_POS = np.array([0.45, 0.05, 0.20])  # 20 cm above the red box
    TARGET_ORN = p.getQuaternionFromEuler([np.pi, 0, 0])  # top-down

    print(f"\n  [2] IK → reach {TARGET_POS} (pre-grasp) …")
    set_gripper(robot_id, jmap, 0.8)  # open gripper
    ik = run_ik(robot_id, jmap, ee_idx, TARGET_POS, TARGET_ORN)
    step_sim(400, gui, dt)

    ee_state = p.getLinkState(robot_id, ee_idx, computeForwardKinematics=True)
    ee_actual = np.array(ee_state[4])
    pos_err = np.linalg.norm(ee_actual - TARGET_POS)
    print(f"     Target: {TARGET_POS}  |  Actual EE: {np.round(ee_actual, 4)}")
    print(f"     Position error: {pos_err*100:.1f} cm  "
          f"({'REACHABLE' if pos_err < 0.05 else 'UNREACHABLE — too far for HSR 4-DOF arm'})")

    # ── Phase 3: Descend and close gripper (mock grasp) ───────────────────────
    GRASP_POS = TARGET_POS.copy(); GRASP_POS[2] = 0.10
    print(f"\n  [3] Descend to {GRASP_POS} …")
    run_ik(robot_id, jmap, ee_idx, GRASP_POS, TARGET_ORN)
    step_sim(300, gui, dt)

    print("  [3] Close gripper …")
    set_gripper(robot_id, jmap, 0.0)
    step_sim(120, gui, dt)

    # ── Phase 4: Head camera capture ──────────────────────────────────────────
    # Tilt head to look at the table
    drive_joint(robot_id, jmap, "head_tilt_joint", -0.5)
    step_sim(120, gui, dt)

    print("\n  [4] Capturing head camera image …")
    rgb = capture_head_camera(robot_id, jmap)
    if rgb is not None:
        os.makedirs(os.path.dirname(OUTPUT_IMG), exist_ok=True)
        try:
            import cv2
            cv2.imwrite(OUTPUT_IMG, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            print(f"     Saved: {OUTPUT_IMG}")
        except ImportError:
            # Fall back to PIL
            from PIL import Image
            Image.fromarray(rgb).save(OUTPUT_IMG)
            print(f"     Saved (PIL): {OUTPUT_IMG}")

    # ── Phase 5: Return to home ────────────────────────────────────────────────
    print("\n  [5] Returning to home …")
    for jname, jval in HOME.items():
        drive_joint(robot_id, jmap, jname, jval)
    torso_idx = jmap.get(TORSO_JOINT)
    if torso_idx is not None:
        drive_joint(robot_id, jmap, TORSO_JOINT, HOME.get("arm_lift_joint", 0.0) * 0.5)
    step_sim(400, gui, dt)

    print("\n  ─────────────────────────────────────────────")
    print("  HSR test complete.")
    print("  GUI is open — press Ctrl-C or close the window to exit.")

    if gui:
        try:
            while p.isConnected():
                p.stepSimulation()
                time.sleep(dt)
        except KeyboardInterrupt:
            pass

    p.disconnect()
    print("  Disconnected.")


if __name__ == "__main__":
    main()
