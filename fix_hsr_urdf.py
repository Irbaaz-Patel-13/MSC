"""
HSR URDF Fixer for PyBullet
============================
Converts the Toyota HSR hsrb4s.urdf (which uses ROS package:// URIs and Gazebo
extensions) into a PyBullet-compatible hsrb4s_pybullet.urdf by:

  1. Replacing package://hsr_meshes/  → ../../hsr_meshes/
  2. Replacing package://hsr_description/ → ../
  3. Replacing .dae visual meshes with .obj equivalents (PyBullet handles .obj
     natively; .dae requires pyassimp which is not always available)
  4. Stripping <gazebo>, <transmission>, and <plugin> blocks (PyBullet ignores
     these and they can cause URDF parsing warnings)
  5. Fixing known typos in the source URDF (e.g. torso.std → torso.stl)

Run once after cloning the URDF repos:
    python fix_hsr_urdf.py

Then validate the result with:
    python fix_hsr_urdf.py --validate
"""

import os
import re
import sys
import argparse
from pathlib import Path


# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
SRC_URDF   = SCRIPT_DIR / "hsr_description" / "robots" / "hsrb4s.urdf"
DST_URDF   = SCRIPT_DIR / "hsr_description" / "robots" / "hsrb4s_pybullet.urdf"
# Relative from DST_URDF's directory to the mesh repos
HSR_MESHES_REL   = "../../hsr_meshes/"
HSR_DESC_REL     = "../"


def replace_package_uris(content: str) -> str:
    """Replace all package:// URIs with filesystem-relative paths."""
    content = content.replace("package://hsr_meshes/",     HSR_MESHES_REL)
    content = content.replace("package://hsr_description/", HSR_DESC_REL)
    return content


def replace_dae_with_obj(content: str) -> str:
    """
    Replace .dae mesh paths with .obj where the .obj file exists on disk.
    If no .obj is found, fall back to .stl; if that also missing, keep .dae.

    Matches patterns like:
        <mesh filename="../../hsr_meshes/meshes/arm_v0/shoulder.dae"/>
    """
    def _subst(match):
        prefix = match.group(1)   # e.g. '../../hsr_meshes/meshes/arm_v0/shoulder'
        closing = match.group(2)  # e.g. '"/>'

        # Resolve absolute path of the candidate mesh
        base_abs = (DST_URDF.parent / prefix).resolve()

        for ext in (".obj", ".stl"):
            candidate = base_abs.with_suffix(ext)
            if candidate.exists():
                return f'"{prefix}{ext}{closing}'

        # Neither .obj nor .stl found — keep .dae (PyBullet may skip visual)
        return match.group(0)

    # Match quoted .dae paths that have already had package:// replaced
    pattern = r'"([^"]*?)\.dae("(?:\s*/>|>))'
    return re.sub(pattern, _subst, content)


def fix_typos(content: str) -> str:
    """Fix known typos in the source URDF."""
    # torso.std → torso.stl
    content = content.replace("torso.std", "torso.stl")
    return content


def remove_xml_blocks(content: str, tag: str) -> str:
    """
    Remove all occurrences of <tag ...>...</tag> from the content.
    Handles multi-line blocks and nested same-tag elements with a depth counter.
    """
    result = []
    idx = 0
    open_pat  = re.compile(rf'<{tag}(?:\s[^>]*)?>',  re.DOTALL)
    close_pat = re.compile(rf'</{tag}>')
    self_close_pat = re.compile(rf'<{tag}(?:\s[^>]*)?/>', re.DOTALL)

    while idx < len(content):
        # Look for self-closing tags first
        sc_m = self_close_pat.search(content, idx)
        # Look for opening tags
        op_m = open_pat.search(content, idx)

        if op_m is None and sc_m is None:
            # No more tags of this type
            result.append(content[idx:])
            break

        # Pick whichever comes first
        if sc_m is not None and (op_m is None or sc_m.start() < op_m.start()):
            # Self-closing tag: keep content before, skip tag
            result.append(content[idx:sc_m.start()])
            idx = sc_m.end()
            continue

        # Opening tag found
        result.append(content[idx:op_m.start()])
        idx = op_m.end()
        depth = 1

        while depth > 0 and idx < len(content):
            next_open  = open_pat.search(content, idx)
            next_close = close_pat.search(content, idx)

            if next_close is None:
                # Malformed URDF — skip to end
                idx = len(content)
                break

            if next_open is not None and next_open.start() < next_close.start():
                depth += 1
                idx = next_open.end()
            else:
                depth -= 1
                idx = next_close.end()

    return "".join(result)


def clean_blank_lines(content: str) -> str:
    """Collapse 3+ consecutive blank lines down to 1."""
    return re.sub(r'\n{3,}', '\n\n', content)


def fix_urdf(src: Path = SRC_URDF, dst: Path = DST_URDF) -> Path:
    print(f"[fix_hsr_urdf] Reading: {src}")
    with open(src, "r", encoding="utf-8") as f:
        content = f.read()

    print("[fix_hsr_urdf] Replacing package:// URIs …")
    content = replace_package_uris(content)

    print("[fix_hsr_urdf] Replacing .dae → .obj where available …")
    content = replace_dae_with_obj(content)

    print("[fix_hsr_urdf] Fixing typos …")
    content = fix_typos(content)

    print("[fix_hsr_urdf] Removing <gazebo> blocks …")
    content = remove_xml_blocks(content, "gazebo")

    print("[fix_hsr_urdf] Removing <transmission> blocks …")
    content = remove_xml_blocks(content, "transmission")

    print("[fix_hsr_urdf] Removing <plugin> blocks …")
    content = remove_xml_blocks(content, "plugin")

    content = clean_blank_lines(content)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[fix_hsr_urdf] Saved: {dst}  ({dst.stat().st_size // 1024} KB)")
    return dst


def validate_urdf(urdf_path: Path) -> bool:
    """Load the URDF in PyBullet DIRECT mode and print joint information."""
    import pybullet as p
    import pybullet_data

    print(f"\n[fix_hsr_urdf] Validating in PyBullet: {urdf_path}")
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    try:
        robot_id = p.loadURDF(
            str(urdf_path),
            basePosition=[0, 0, 0],
            useFixedBase=True,
        )
    except Exception as exc:
        print(f"[fix_hsr_urdf] ERROR loading URDF: {exc}")
        p.disconnect()
        return False

    n_joints = p.getNumJoints(robot_id)
    print(f"\n  Loaded OK — {n_joints} joints:")
    print(f"  {'idx':>4}  {'name':<35}  {'type':<12}  {'limits'}")
    print(f"  {'-'*75}")

    joint_type_names = {
        p.JOINT_REVOLUTE:   "revolute",
        p.JOINT_PRISMATIC:  "prismatic",
        p.JOINT_SPHERICAL:  "spherical",
        p.JOINT_PLANAR:     "planar",
        p.JOINT_FIXED:      "fixed",
    }

    for i in range(n_joints):
        info = p.getJointInfo(robot_id, i)
        name = info[1].decode("utf-8")
        jtype = joint_type_names.get(info[2], str(info[2]))
        lo, hi = info[8], info[9]
        if info[2] != p.JOINT_FIXED:
            limits = f"[{lo:.3f}, {hi:.3f}]"
        else:
            limits = "—"
        print(f"  {i:>4}  {name:<35}  {jtype:<12}  {limits}")

    p.disconnect()
    print("\n[fix_hsr_urdf] Validation passed.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix HSR URDF for PyBullet.")
    parser.add_argument("--src",      type=Path, default=SRC_URDF)
    parser.add_argument("--dst",      type=Path, default=DST_URDF)
    parser.add_argument("--validate", action="store_true",
                        help="Load in PyBullet after fixing and print joint info")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate an existing fixed URDF (skip fixing)")
    args = parser.parse_args()

    if not args.validate_only:
        out = fix_urdf(args.src, args.dst)
    else:
        out = args.dst

    if args.validate or args.validate_only:
        ok = validate_urdf(out)
        sys.exit(0 if ok else 1)
