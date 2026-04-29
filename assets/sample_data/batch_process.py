#!/usr/bin/env python3
"""
Batch process RoboTool annotated data into a format suitable for training.

Usage:
    # Process all tasks
    python data/robotool_batch/batch_process.py

    # Process specific task(s)
    python data/robotool_batch/batch_process.py --tasks grasp_cup grasp_brush

    # Process specific sequence(s)
    python data/robotool_batch/batch_process.py --sequences grasp_cup/grasp_cup_1

    # Custom paths
    python data/robotool_batch/batch_process.py \
        --annotated_dir /path/to/data_annotated \
        --raw_dir /path/to/data \
        --models_dir /path/to/models \
        --output_dir data/robotool_batch
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pickle
import yaml
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.robotool.load_data import (
    load_pkl_and_get_hand_data,
    init_mano_layers,
    reconstruct_left_hand,
    reconstruct_right_hand,
    EXPORT_KEYS,
)


DEFAULT_ANNOTATED_DIR = "/home/ruoqu/crq_ws/robotool/DataCollection/data_annotated"
DEFAULT_RAW_DIR = "/home/ruoqu/crq_ws/robotool/DataCollection/data"
DEFAULT_MODELS_DIR = "/home/ruoqu/crq_ws/robotool/HO-Cap-Annotation/data/models"
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "data" / "robotool_batch")


def load_meta_yaml(raw_dir: str, task: str, sequence: str) -> dict:
    """Load meta.yaml from raw data directory to get object_ids and hand sides."""
    meta_path = os.path.join(raw_dir, task, sequence, "meta.yaml")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.yaml not found at {meta_path}")
    with open(meta_path, "r") as f:
        return yaml.safe_load(f)


def load_object_pose(annotated_dir: str, task: str, sequence: str) -> np.ndarray:
    """Load object pose from joint_pose_solver (preferred) or object_pose_solver."""
    seq_dir = os.path.join(annotated_dir, task, sequence, "processed")

    # Prefer joint_pose_solver output
    joint_pose_path = os.path.join(seq_dir, "joint_pose_solver", "poses_o.npy")
    if os.path.exists(joint_pose_path):
        poses = np.load(joint_pose_path)
        return poses

    # Fallback to object_pose_solver
    obj_pose_path = os.path.join(seq_dir, "object_pose_solver", "poses_o.npy")
    if os.path.exists(obj_pose_path):
        poses = np.load(obj_pose_path)
        return poses

    # Fallback to fd_poses_merged_fixed
    fd_path = os.path.join(seq_dir, "fd_pose_solver", "fd_poses_merged_fixed.npy")
    if os.path.exists(fd_path):
        poses = np.load(fd_path)
        return poses

    return None


def quat_pos_to_4x4(poses: np.ndarray) -> np.ndarray:
    """Convert (N, 7) [qx, qy, qz, qw, tx, ty, tz] to (N, 4, 4) SE(3) matrices."""
    from scipy.spatial.transform import Rotation as R

    if poses.ndim == 3 and poses.shape[1] == 4 and poses.shape[2] == 4:
        return poses  # Already 4x4

    if poses.ndim == 3:
        # (num_objects, N, 7) -> take first object
        poses = poses[0]

    N = poses.shape[0]
    result = np.eye(4, dtype=np.float32)[None].repeat(N, axis=0)

    if poses.shape[1] == 7:
        # [qx, qy, qz, qw, tx, ty, tz]
        quats = poses[:, :4]  # xyzw
        trans = poses[:, 4:7]
        rots = R.from_quat(quats).as_matrix()
        result[:, :3, :3] = rots
        result[:, :3, 3] = trans
    else:
        raise ValueError(f"Unexpected pose shape: {poses.shape}")

    return result


def setup_object_model(src_models_dir: str, object_id: str, output_dir: str) -> str:
    """Copy object mesh to centralized models dir and generate URDF.

    Models are stored at: {output_dir}/models/{object_id}/cleaned_mesh_10000.obj
    (not per-experiment, shared across all experiments using the same object)

    Returns path to obj file.
    """
    src_obj = os.path.join(src_models_dir, object_id, "cleaned_mesh_10000.obj")
    if not os.path.exists(src_obj):
        print(f"  [WARN] Object mesh not found: {src_obj}")
        return None

    dst_model_dir = os.path.join(output_dir, "models", object_id)
    os.makedirs(dst_model_dir, exist_ok=True)

    # Copy .obj (and replace symlink if exists)
    import shutil
    dst_obj = os.path.join(dst_model_dir, "cleaned_mesh_10000.obj")
    if os.path.islink(dst_obj):
        os.remove(dst_obj)  # remove stale symlink
    if not os.path.exists(dst_obj):
        shutil.copy2(os.path.abspath(src_obj), dst_obj)
        print(f"  Copied mesh: {object_id}/cleaned_mesh_10000.obj")

    # Copy all related model assets (textures, materials)
    src_model_dir = os.path.join(src_models_dir, object_id)
    for fname in os.listdir(src_model_dir):
        if fname.endswith(('.png', '.jpeg', '.jpg', '.mtl')):
            src_f = os.path.join(src_model_dir, fname)
            dst_f = os.path.join(dst_model_dir, fname)
            if os.path.islink(dst_f):
                os.remove(dst_f)
            if not os.path.exists(dst_f):
                shutil.copy2(src_f, dst_f)
                print(f"  Copied asset: {object_id}/{fname}")

    # Generate URDF
    dst_urdf = os.path.join(dst_model_dir, "cleaned_mesh_10000.urdf")
    if not os.path.exists(dst_urdf):
        urdf_content = f"""<?xml version="1.0"?>
<robot name="robotool_object">
  <material name="obj_color">
      <color rgba="1.0 0.423529411765 0.0392156862745 1.0"/>
  </material>
  <link name="base">
    <visual>
      <origin xyz="0.0 0.0 0.0"/>
      <geometry>
        <mesh filename="cleaned_mesh_10000.obj" scale="1 1 1"/>
      </geometry>
      <material name="obj_color"/>
    </visual>
    <collision>
      <origin xyz="0.0 0.0 0.0"/>
      <geometry>
        <mesh filename="cleaned_mesh_10000.obj" scale="1 1 1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>
</robot>
"""
        with open(dst_urdf, "w") as f:
            f.write(urdf_content)

    return dst_obj


def process_sequence(
    annotated_dir: str,
    raw_dir: str,
    models_dir: str,
    output_dir: str,
    task: str,
    sequence: str,
    mano_layer_left,
    mano_layer_right,
    force: bool = False,
):
    """Process a single sequence: MANO reconstruction + object pose + mesh setup."""
    output_seq_dir = os.path.join(output_dir, task, sequence)
    mano_joints_path = os.path.join(output_seq_dir, "mano_joints.pkl")
    meta_json_path = os.path.join(output_seq_dir, "meta.json")

    # Skip if already processed (unless force)
    if not force and os.path.exists(mano_joints_path) and os.path.exists(meta_json_path):
        print(f"  [SKIP] {task}/{sequence} already processed")
        return True

    os.makedirs(output_seq_dir, exist_ok=True)

    # 1. Load meta.yaml for object_ids
    try:
        meta = load_meta_yaml(raw_dir, task, sequence)
    except FileNotFoundError as e:
        print(f"  [ERROR] {e}")
        return False

    object_ids = meta.get("object_ids", [])
    mano_sides = meta.get("mano_sides", ["right"])
    num_frames_meta = meta.get("num_frames", 0)

    # 2. Load hand data from result_hand_optimized.pkl
    pkl_path = os.path.join(annotated_dir, task, sequence, "result_hand_optimized.pkl")
    if not os.path.exists(pkl_path):
        pkl_path = os.path.join(annotated_dir, task, sequence, "result.pkl")
    if not os.path.exists(pkl_path):
        print(f"  [ERROR] No result pkl found for {task}/{sequence}")
        return False

    try:
        hand_data = load_pkl_and_get_hand_data(pkl_path)
    except Exception as e:
        print(f"  [ERROR] Failed to load hand data: {e}")
        return False

    num_frames = len(hand_data["left_hand_pose"]) if "left" in mano_sides else len(hand_data["right_hand_pose"])
    print(f"  Frames: {num_frames}, Objects: {object_ids}, Hands: {mano_sides}")

    # 3. MANO reconstruction for each hand
    left_hand_joints = {name: [] for name in EXPORT_KEYS}
    left_hand_joints["wrist_translation"] = []
    left_hand_joints["wrist_orientation"] = []

    right_hand_joints = {name: [] for name in EXPORT_KEYS}
    right_hand_joints["wrist_translation"] = []
    right_hand_joints["wrist_orientation"] = []

    for frame_idx in tqdm(range(num_frames), desc=f"  MANO {sequence}", leave=False):
        if "left" in mano_sides:
            left_joint_dict = reconstruct_left_hand(hand_data, frame_idx, mano_layer_right)
            if left_joint_dict is not None:
                for key in EXPORT_KEYS:
                    left_hand_joints[key].append(left_joint_dict[key])
                left_hand_joints["wrist_translation"].append(left_joint_dict["wrist_translation"])
                left_hand_joints["wrist_orientation"].append(left_joint_dict["wrist_orientation"])
            else:
                for key in EXPORT_KEYS:
                    left_hand_joints[key].append(np.zeros(3))
                left_hand_joints["wrist_translation"].append(np.zeros(3))
                left_hand_joints["wrist_orientation"].append(np.array([0, 0, 0, 1]))

        if "right" in mano_sides:
            right_joint_dict = reconstruct_right_hand(hand_data, frame_idx, mano_layer_right)
            if right_joint_dict is not None:
                for key in EXPORT_KEYS:
                    right_hand_joints[key].append(right_joint_dict[key])
                right_hand_joints["wrist_translation"].append(right_joint_dict["wrist_translation"])
                right_hand_joints["wrist_orientation"].append(right_joint_dict["wrist_orientation"])
            else:
                for key in EXPORT_KEYS:
                    right_hand_joints[key].append(np.zeros(3))
                right_hand_joints["wrist_translation"].append(np.zeros(3))
                right_hand_joints["wrist_orientation"].append(np.array([0, 0, 0, 1]))

    # 4. Build output dict
    output_dict = {"left": {}, "right": {}}

    if "left" in mano_sides:
        for key in EXPORT_KEYS:
            output_dict["left"][key] = np.array(left_hand_joints[key])
        output_dict["left"]["wrist_translation"] = np.array(left_hand_joints["wrist_translation"])
        output_dict["left"]["wrist_orientation"] = np.array(left_hand_joints["wrist_orientation"])

    if "right" in mano_sides:
        for key in EXPORT_KEYS:
            output_dict["right"][key] = np.array(right_hand_joints[key])
        output_dict["right"]["wrist_translation"] = np.array(right_hand_joints["wrist_translation"])
        output_dict["right"]["wrist_orientation"] = np.array(right_hand_joints["wrist_orientation"])

    # 5. Load original pkl data for tool_object_pose etc.
    with open(pkl_path, "rb") as f:
        original_data = pickle.load(f)
    output_dict["original_data"] = original_data

    # 6. Load and attach object pose from annotation pipeline
    obj_poses = load_object_pose(annotated_dir, task, sequence)
    if obj_poses is not None:
        obj_poses_4x4 = quat_pos_to_4x4(obj_poses)
        output_dict["original_data"]["tool_object_pose"] = obj_poses_4x4
        print(f"  Object pose loaded: shape={obj_poses.shape}")
    else:
        print(f"  [WARN] No object pose found for {task}/{sequence}")

    # 7. Save mano_joints.pkl
    with open(mano_joints_path, "wb") as f:
        pickle.dump(output_dict, f)

    # 8. Setup object model (symlink + URDF)
    obj_mesh_paths = {}
    for obj_id in object_ids:
        obj_path = setup_object_model(models_dir, obj_id, output_dir)
        if obj_path:
            obj_mesh_paths[obj_id] = obj_path

    # 9. Save meta.json
    meta_info = {
        "task": task,
        "sequence": sequence,
        "object_ids": object_ids,
        "mano_sides": mano_sides,
        "num_frames": num_frames,
        "source_fps": 25.0,
        "obj_mesh_paths": obj_mesh_paths,
        "annotated_source": os.path.join(annotated_dir, task, sequence),
    }
    with open(meta_json_path, "w") as f:
        json.dump(meta_info, f, indent=2)

    print(f"  [OK] Saved to {output_seq_dir}")
    return True


def discover_sequences(annotated_dir: str, tasks: list = None) -> list:
    """Discover all task/sequence pairs in annotated directory."""
    results = []
    for task in sorted(os.listdir(annotated_dir)):
        task_dir = os.path.join(annotated_dir, task)
        if not os.path.isdir(task_dir):
            continue
        if tasks and task not in tasks:
            continue
        for seq in sorted(os.listdir(task_dir)):
            seq_dir = os.path.join(task_dir, seq)
            if os.path.isdir(seq_dir):
                results.append((task, seq))
    return results


def main():
    parser = argparse.ArgumentParser(description="Batch process RoboTool annotated data")
    parser.add_argument("--annotated_dir", default=DEFAULT_ANNOTATED_DIR,
                        help="Path to data_annotated directory")
    parser.add_argument("--raw_dir", default=DEFAULT_RAW_DIR,
                        help="Path to raw data directory (for meta.yaml)")
    parser.add_argument("--models_dir", default=DEFAULT_MODELS_DIR,
                        help="Path to object models directory")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR,
                        help="Output directory")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Process only these tasks (e.g., grasp_cup grasp_brush)")
    parser.add_argument("--sequences", nargs="+", default=None,
                        help="Process only these sequences (e.g., grasp_cup/grasp_cup_1)")
    parser.add_argument("--force", action="store_true",
                        help="Reprocess even if output already exists")
    args = parser.parse_args()

    # Discover sequences
    if args.sequences:
        sequences = [(s.split("/")[0], s.split("/")[1]) for s in args.sequences]
    else:
        sequences = discover_sequences(args.annotated_dir, args.tasks)

    print(f"Found {len(sequences)} sequences to process:")
    for task, seq in sequences:
        print(f"  {task}/{seq}")

    if not sequences:
        print("No sequences found. Check --annotated_dir path.")
        return

    # Initialize MANO layers once
    print("\nInitializing MANO layers...")
    import torch
    from manopth.manolayer import ManoLayer

    mano_layer_right = ManoLayer(
        side="right",
        mano_root="/home/ruoqu/crq_ws/data/manopth/mano/models",
        use_pca=False,
        ncomps=45,
    ).to("cuda")
    mano_layer_left = ManoLayer(
        side="left",
        mano_root="/home/ruoqu/crq_ws/data/manopth/mano/models",
        use_pca=False,
        ncomps=45,
    ).to("cuda")

    # Process each sequence
    success_count = 0
    fail_count = 0
    for task, seq in sequences:
        print(f"\nProcessing {task}/{seq}...")
        ok = process_sequence(
            annotated_dir=args.annotated_dir,
            raw_dir=args.raw_dir,
            models_dir=args.models_dir,
            output_dir=args.output_dir,
            task=task,
            sequence=seq,
            mano_layer_left=mano_layer_left,
            mano_layer_right=mano_layer_right,
            force=args.force,
        )
        if ok:
            success_count += 1
        else:
            fail_count += 1

    print(f"\n{'='*50}")
    print(f"Batch processing complete: {success_count} succeeded, {fail_count} failed")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
