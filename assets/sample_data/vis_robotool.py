#!/usr/bin/env python3
"""
Visualize robotool mano_joints.pkl data using viser.
Works with both old-format (data/robotool/mano_joints.pkl) and
batch-format (data/robotool_batch/{task}/{seq}/mano_joints.pkl).

Usage:
    conda run -n dexmanip python visualizer/vis_robotool.py \
        --pkl data/robotool/mano_joints.pkl \
        --obj_mesh data/robotool/models/squeegee/cleaned_mesh_10000.obj

    # Object mesh is optional (auto-detected from batch format meta.json)
    conda run -n dexmanip python visualizer/vis_robotool.py \
        --pkl data/robotool_batch/blue_cup/blue_cup_1/mano_joints.pkl
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import trimesh
import viser
from scipy.spatial.transform import Rotation as R

# ---- Hand skeleton definition ----

FINGER_CHAINS = {
    "thumb":  ["wrist", "thumb_proximal", "thumb_intermediate", "thumb_distal", "thumb_tip"],
    "index":  ["wrist", "index_proximal", "index_intermediate", "index_distal", "index_tip"],
    "middle": ["wrist", "middle_proximal", "middle_intermediate", "middle_distal", "middle_tip"],
    "ring":   ["wrist", "ring_proximal", "ring_intermediate", "ring_distal", "ring_tip"],
    "pinky":  ["wrist", "pinky_proximal", "pinky_intermediate", "pinky_distal", "pinky_tip"],
}

JOINT_NAMES = [
    "wrist",
    "thumb_proximal", "thumb_intermediate", "thumb_distal", "thumb_tip",
    "index_proximal", "index_intermediate", "index_distal", "index_tip",
    "middle_proximal", "middle_intermediate", "middle_distal", "middle_tip",
    "ring_proximal", "ring_intermediate", "ring_distal", "ring_tip",
    "pinky_proximal", "pinky_intermediate", "pinky_distal", "pinky_tip",
]

FINGER_COLORS = {
    "thumb":  (255, 100, 100),
    "index":  (100, 255, 100),
    "middle": (100, 100, 255),
    "ring":   (255, 255, 100),
    "pinky":  (255, 100, 255),
}
LEFT_WRIST_COLOR = (100, 149, 237)
RIGHT_WRIST_COLOR = (255, 160, 122)
OBJ_COLOR = (180, 180, 180)




def pose7_to_4x4(pose: np.ndarray) -> np.ndarray:
    """Convert (7,) [qx,qy,qz,qw, tx,ty,tz] to (4,4) SE(3)."""
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.from_quat(pose[:4]).as_matrix()
    T[:3, 3] = pose[4:7]
    return T


def transform_verts(vertices: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply 4x4 transform to vertices."""
    return (T[:3, :3] @ vertices.T).T + T[:3, 3]


def main():
    parser = argparse.ArgumentParser(description="Visualize robotool mano_joints.pkl with viser")
    parser.add_argument("--pkl", type=str, required=True,
                        help="Path to mano_joints.pkl")
    parser.add_argument("--obj_mesh", type=str, default=None,
                        help="Path to object mesh (.obj/.ply). Auto-detected for batch format.")
    parser.add_argument("--side", type=str, default="both", choices=["left", "right", "both"],
                        help="Which hand(s) to show")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    args = parser.parse_args()

    # Load data
    print(f"[INFO] Loading {args.pkl}...")
    with open(args.pkl, "rb") as f:
        data = pickle.load(f)

    has_left = "left" in data and len(data["left"].get("wrist", [])) > 0
    has_right = "right" in data and len(data["right"].get("wrist", [])) > 0

    # Determine num_frames
    if has_left:
        num_frames = len(np.array(data["left"]["wrist"]))
    elif has_right:
        num_frames = len(np.array(data["right"]["wrist"]))
    else:
        print("[ERROR] No hand data found")
        return

    print(f"[INFO] {num_frames} frames, left={has_left}, right={has_right}")

    # Load object mesh
    obj_vertices = None
    obj_faces = None
    obj_mesh_path = args.obj_mesh

    # Auto-detect from batch format meta.json
    if obj_mesh_path is None:
        meta_path = os.path.join(os.path.dirname(args.pkl), "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("object_ids"):
                obj_id = meta["object_ids"][0]
                candidate = os.path.join(os.path.dirname(args.pkl), "models", obj_id, "cleaned_mesh_10000.obj")
                if os.path.exists(candidate):
                    obj_mesh_path = candidate

    if obj_mesh_path and os.path.exists(obj_mesh_path):
        mesh = trimesh.load(obj_mesh_path, process=False, force="mesh")
        obj_vertices = mesh.vertices.astype(np.float32)
        obj_faces = mesh.faces.copy()
        print(f"[INFO] Object mesh: {obj_mesh_path} ({obj_vertices.shape[0]} verts)")

    # Load object poses
    obj_poses = None
    if "original_data" in data:
        tp = data["original_data"].get("tool_object_pose")
        if tp is not None:
            tp = np.array(tp)
            # Normalize to (N, 4, 4)
            if tp.ndim == 2 and tp.shape[1] == 7:
                obj_poses = np.stack([pose7_to_4x4(p) for p in tp])
            elif tp.ndim == 3 and tp.shape[1] == 4 and tp.shape[2] == 4:
                obj_poses = tp
            print(f"[INFO] Object poses: {obj_poses.shape if obj_poses is not None else 'unsupported format'}")

    has_object = obj_vertices is not None and obj_poses is not None

    # ---- Viser server ----
    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    print(f"[INFO] Viser server at http://localhost:{args.port}")

    # GUI
    frame_slider = server.gui.add_slider("Frame", min=0, max=num_frames - 1, step=1, initial_value=0)
    playing = server.gui.add_checkbox("Play", initial_value=False)
    fps_slider = server.gui.add_slider("FPS", min=1, max=60, step=1, initial_value=15)

    show_left = server.gui.add_checkbox("Show Left Hand", initial_value=has_left and args.side in ("left", "both"))
    show_right = server.gui.add_checkbox("Show Right Hand", initial_value=has_right and args.side in ("right", "both"))
    show_object = server.gui.add_checkbox("Show Object", initial_value=has_object)
    show_bones = server.gui.add_checkbox("Show Bones", initial_value=True)
    joint_radius = server.gui.add_slider("Joint Radius", min=0.001, max=0.015, step=0.001, initial_value=0.005)

    server.scene.add_frame("/world", axes_length=0.1, axes_radius=0.002)

    handles = {}

    def clear_handles(prefix: str):
        keys = [k for k in handles if k.startswith(prefix)]
        for k in keys:
            handles[k].remove()
            del handles[k]

    def draw_hand(hand_data: dict, frame_idx: int, side: str):
        prefix = f"/{side}_hand"
        clear_handles(prefix)

        show = show_left.value if side == "left" else show_right.value
        if not show:
            return

        wrist_color = LEFT_WRIST_COLOR if side == "left" else RIGHT_WRIST_COLOR
        r = joint_radius.value

        positions = {}
        for name in JOINT_NAMES:
            if name in hand_data:
                arr = np.array(hand_data[name])
                if frame_idx < len(arr):
                    positions[name] = arr[frame_idx]

        if not positions:
            return

        for name, pos in positions.items():
            color = wrist_color if name == "wrist" else None
            if color is None:
                for finger, chain in FINGER_CHAINS.items():
                    if name in chain:
                        color = FINGER_COLORS[finger]
                        break
                if color is None:
                    color = (200, 200, 200)

            key = f"{prefix}/joint_{name}"
            handles[key] = server.scene.add_icosphere(
                key,
                radius=r * 1.5 if name == "wrist" else r,
                position=pos.astype(np.float64),
                color=color,
            )

        if show_bones.value:
            for finger, chain in FINGER_CHAINS.items():
                color = FINGER_COLORS[finger]
                for i in range(len(chain) - 1):
                    if chain[i] in positions and chain[i + 1] in positions:
                        p0 = positions[chain[i]].astype(np.float64)
                        p1 = positions[chain[i + 1]].astype(np.float64)
                        key = f"{prefix}/bone_{finger}_{i}"
                        handles[key] = server.scene.add_spline_catmull_rom(
                            key,
                            positions=np.stack([p0, p1]),
                            color=color,
                            line_width=2.0,
                        )

    def draw_object(frame_idx: int):
        clear_handles("/object")
        if not show_object.value or obj_poses is None or obj_vertices is None:
            return
        if frame_idx >= len(obj_poses):
            return

        verts = transform_verts(obj_vertices, obj_poses[frame_idx])

        key = "/object/mesh"
        handles[key] = server.scene.add_mesh_simple(
            key,
            vertices=verts.astype(np.float32),
            faces=obj_faces,
            color=OBJ_COLOR,
        )

    def update_frame(frame_idx: int):
        if has_left:
            draw_hand(data["left"], frame_idx, "left")
        if has_right:
            draw_hand(data["right"], frame_idx, "right")
        draw_object(frame_idx)

    # Initial render
    update_frame(0)

    @frame_slider.on_update
    def _on_frame(event: viser.GuiEvent) -> None:
        update_frame(int(frame_slider.value))

    @show_left.on_update
    def _on_left(event: viser.GuiEvent) -> None:
        update_frame(int(frame_slider.value))

    @show_right.on_update
    def _on_right(event: viser.GuiEvent) -> None:
        update_frame(int(frame_slider.value))

    @show_object.on_update
    def _on_obj(event: viser.GuiEvent) -> None:
        update_frame(int(frame_slider.value))

    @show_bones.on_update
    def _on_bones(event: viser.GuiEvent) -> None:
        update_frame(int(frame_slider.value))

    @joint_radius.on_update
    def _on_radius(event: viser.GuiEvent) -> None:
        update_frame(int(frame_slider.value))

    try:
        while True:
            if playing.value:
                frame = (int(frame_slider.value) + 1) % num_frames
                frame_slider.value = frame
                update_frame(frame)
                time.sleep(1.0 / fps_slider.value)
            else:
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[INFO] Server stopped.")


if __name__ == "__main__":
    main()
