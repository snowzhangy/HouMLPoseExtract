"""
bone_utils.py
Utilities for discovering OBJ bone nodes and sampling world transforms.
Works with Houdini 20 OBJ-level bone nodes + bonedeform SOP workflow.
"""
from __future__ import annotations

import fnmatch
from typing import TYPE_CHECKING

import numpy as np

try:
    import hou
    from scipy.spatial.transform import Rotation
except ImportError:
    hou = None  # allow import in test context with mock_hou
    Rotation = None


def find_bones(root_path: str, pattern: str = "*") -> list:
    """Recursively find all ObjNode of type 'bone' under root_path."""
    if not root_path:
        return []
    root = hou.node(root_path)
    if root is None:
        return []

    all_bones: list = []
    _collect_bones(root, pattern, all_bones)
    return all_bones


def find_bones_from_geo(geo) -> list[str]:
    """Find bone names from a KineFX skeleton geometry (points with 'name' attr).

    Args:
        geo: hou.Geometry containing skeleton points.

    Returns:
        List of bone names in point order.
    """
    if geo is None:
        return []
    name_attr = geo.findPointAttrib("name")
    if name_attr is None:
        return []
    
    return [pt.stringAttribValue("name") for pt in geo.points()]


def build_hierarchy_from_geo(geo) -> tuple[list[str], list[int], dict[str, int]]:
    """Build parent-index table from KineFX skeleton points (uses 'parentidx' or hierarchy).

    Args:
        geo: hou.Geometry containing skeleton points.

    Returns:
        bone_names, parent_indices, name_to_idx.
    """
    bone_names = find_bones_from_geo(geo)
    if not bone_names:
        raise ValueError(
            "Input 1 geometry has no 'name' point attribute — it does not look like a "
            "KineFX skeleton. Connect a KineFX skeleton SOP (Skeleton, Agent, Rig Match "
            "Pose, etc.) to Input 1, or leave Input 1 disconnected and use the OBJ "
            "'Character Root' path instead."
        )

    name_to_idx = {n: i for i, n in enumerate(bone_names)}
    num_pts = len(bone_names)

    parent_indices = [-1] * num_pts
    parent_attr = geo.findPointAttrib("parentidx")

    if parent_attr is not None:
        # Explicit parentidx attribute — read directly
        for i, pt in enumerate(geo.points()):
            if i >= num_pts:
                break
            pidx = int(pt.attribValue("parentidx"))
            if pidx >= num_pts or pidx < -1:
                pidx = -1
            parent_indices[i] = pidx
    else:
        # No parentidx attribute (e.g. fbxcharacterimport) — derive hierarchy
        # from polyline primitives.  Each 2-vertex polyline: vertex0 = parent,
        # vertex1 = child.
        for prim in geo.prims():
            verts = prim.vertices()
            if len(verts) == 2:
                parent_pt_num = verts[0].point().number()
                child_pt_num = verts[1].point().number()
                if child_pt_num < num_pts and parent_pt_num < num_pts:
                    parent_indices[child_pt_num] = parent_pt_num

    return bone_names, parent_indices, name_to_idx


def get_input_geometry_at_frame(node, input_index: int, frame: float) -> hou.Geometry | None:
    """Evaluate the geometry of a specific input at a specific frame.
    
    This avoids hou.setFrame() which can trigger recursive cooks if called
    inside a cook function. It correctly handles multi-output upstream nodes.
    """
    if node is None:
        return None
    
    connections = node.inputConnections()
    for conn in connections:
        if conn.inputIndex() == input_index:
            upstream_node = conn.inputNode()
            output_index = conn.outputIndex()
            return upstream_node.geometryAtFrame(frame, output_index)
            
    return None


def sample_world_transforms_from_geo(accessor_node, input_index: int,
                                     frames: list[int],
                                     bone_names: list[str] | None = None,
                                     status_node=None,
                                     progress_cb=None) -> np.ndarray:
    """Sample world-space matrices from a KineFX skeleton across time.

    Args:
        accessor_node: A SOP node whose *input_index* receives KineFX skeleton
                       geometry (e.g. bonedeform, or the HDA itself).
        input_index:   Which input of *accessor_node* carries the skeleton.
        frames:        List of frames to sample.
        bone_names:    Canonical list of bone names. If None, derived from the
                       skeleton geometry at the current frame.
        status_node:   Optional node for progress reporting.

    Returns:
        float32 array [num_frames, num_bones, 4, 4].
    """
    num_frames = len(frames)
    if bone_names is None:
        first_geo = accessor_node.inputGeometry(input_index)
        if first_geo is None:
            raise ValueError("No geometry found on skeleton input")
        bone_names = find_bones_from_geo(first_geo)
    
    num_bones = len(bone_names)
    name_to_idx = {name: i for i, name in enumerate(bone_names)}
        
    result = np.zeros((num_frames, num_bones, 4, 4), dtype=np.float32)
    # Default to identity
    for fi in range(num_frames):
        for bi in range(num_bones):
            result[fi, bi] = np.eye(4, dtype=np.float32)

    report_interval = max(1, num_frames // 20)

    for fi, frame in enumerate(frames):
        geo = get_input_geometry_at_frame(accessor_node, input_index, float(frame))
            
        if geo is None:
            # Fallback to current geometry if geometryAtFrame failed or input is disconnected
            geo = accessor_node.inputGeometry(input_index)
            
        if geo is None:
            continue
            
        attr_world = geo.findPointAttrib("worldtransform")
        attr_local = geo.findPointAttrib("transform")
        attr_name = geo.findPointAttrib("name")
        
        for pt in geo.points():
            if attr_name is not None:
                name = pt.stringAttribValue(attr_name)
                bi = name_to_idx.get(name, -1)
                if bi == -1:
                    continue
            else:
                bi = pt.number()
                if bi >= num_bones:
                    continue
                
            m = None
            try:
                if attr_world is not None:
                    m = pt.attribValue(attr_world) # Usually hou.Matrix4
                elif attr_local is not None:
                    # Build 4x4 from 3x3 transform + position
                    rot3 = pt.attribValue(attr_local)
                    r = rot3.asTuple() if hasattr(rot3, 'asTuple') else tuple(rot3)
                    pos = pt.position()
                    p = (pos[0], pos[1], pos[2])
                    m = hou.Matrix4((
                        r[0], r[1], r[2], 0,
                        r[3], r[4], r[5], 0,
                        r[6], r[7], r[8], 0,
                        p[0], p[1], p[2], 1,
                    ))
            except Exception:
                pass
            
            if m is not None:
                m_val = m.asTuple() if hasattr(m, "asTuple") else m
                result[fi, bi] = np.array(m_val, dtype=np.float32).reshape(4, 4)

        if fi % report_interval == 0 or fi == num_frames - 1:
            if progress_cb is not None:
                progress_cb(fi, num_frames)
            if status_node is not None:
                pct = 100 * (fi + 1) // num_frames
                try:
                    status_node.parm("status_message").set(
                        f"Sampling KineFX: {pct}% ({fi+1}/{num_frames} frames)"
                    )
                except Exception:
                    pass

    return result


def sample_world_transforms_from_node(sop_node, frames: list[int],
                                      bone_names: list[str] | None = None,
                                      status_node=None,
                                      progress_cb=None) -> np.ndarray:
    """Sample world-space matrices from a referenced SOP node across time.

    Unlike ``sample_world_transforms_from_geo`` (which follows an input
    connection), this reads geometry directly from *sop_node* using
    ``geometryAtFrame()``.  Used for extra animation sources specified
    by path rather than wired inputs.

    Returns:
        float32 array [num_frames, num_bones, 4, 4].
    """
    num_frames = len(frames)
    if bone_names is None:
        first_geo = sop_node.geometry()
        if first_geo is None:
            raise ValueError("No geometry found on referenced SOP node")
        bone_names = find_bones_from_geo(first_geo)
    
    num_bones = len(bone_names)
    name_to_idx = {name: i for i, name in enumerate(bone_names)}
        
    result = np.zeros((num_frames, num_bones, 4, 4), dtype=np.float32)
    # Default to identity
    for fi in range(num_frames):
        for bi in range(num_bones):
            result[fi, bi] = np.eye(4, dtype=np.float32)

    report_interval = max(1, num_frames // 20)

    for fi, frame in enumerate(frames):
        geo = sop_node.geometryAtFrame(float(frame))
        if geo is None:
            continue
            
        attr_world = geo.findPointAttrib("worldtransform")
        attr_local = geo.findPointAttrib("transform")
        attr_name = geo.findPointAttrib("name")
        
        for pt in geo.points():
            if attr_name is not None:
                name = pt.stringAttribValue(attr_name)
                bi = name_to_idx.get(name, -1)
                if bi == -1:
                    continue
            else:
                bi = pt.number()
                if bi >= num_bones:
                    continue
                
            m = None
            try:
                if attr_world is not None:
                    m = pt.attribValue(attr_world)
                elif attr_local is not None:
                    rot3 = pt.attribValue(attr_local)
                    r = rot3.asTuple() if hasattr(rot3, 'asTuple') else tuple(rot3)
                    pos = pt.position()
                    p = (pos[0], pos[1], pos[2])
                    m = hou.Matrix4((
                        r[0], r[1], r[2], 0,
                        r[3], r[4], r[5], 0,
                        r[6], r[7], r[8], 0,
                        p[0], p[1], p[2], 1,
                    ))
            except Exception:
                pass
            
            if m is not None:
                m_val = m.asTuple() if hasattr(m, "asTuple") else m
                result[fi, bi] = np.array(m_val, dtype=np.float32).reshape(4, 4)

        if fi % report_interval == 0 or fi == num_frames - 1:
            if progress_cb is not None:
                progress_cb(fi, num_frames)
            if status_node is not None:
                pct = 100 * (fi + 1) // num_frames
                try:
                    status_node.parm("status_message").set(
                        f"Sampling extra SOP: {pct}% ({fi+1}/{num_frames} frames)"
                    )
                except Exception:
                    pass

    return result


def _collect_bones(node, pattern: str, result: list) -> None:
    """DFS traversal collecting bone nodes that match *pattern*."""
    if node.type().name() == "bone" and fnmatch.fnmatch(node.name(), pattern):
        result.append(node)
    for child in node.children():
        _collect_bones(child, pattern, result)


def build_hierarchy(bones: list) -> tuple[list[str], list[int], dict[str, int]]:
    """Build parent-index table for a list of bone nodes.

    Args:
        bones: Ordered list of hou.ObjNode (must be in hierarchy order).

    Returns:
        bone_names:    List of bone name strings.
        parent_indices: parent_indices[i] = index of parent in *bones*, -1 for roots.
        name_to_idx:   Dict mapping bone name → index in *bones*.
    """
    bone_names = [b.name() for b in bones]
    name_to_idx: dict[str, int] = {n: i for i, n in enumerate(bone_names)}

    parent_indices: list[int] = []
    for bone in bones:
        parent = bone.parent()
        if parent is not None and parent.type().name() == "bone":
            pidx = name_to_idx.get(parent.name(), -1)
        else:
            pidx = -1
        parent_indices.append(pidx)

    return bone_names, parent_indices, name_to_idx


def sample_world_transforms(bones: list, frames: list[int],
                             status_node=None,
                             progress_cb=None) -> np.ndarray:
    """Sample world-space 4×4 transform matrices for every bone at every frame.

    Args:
        bones:       List of hou.ObjNode.
        frames:      List of integer frame numbers to sample.
        status_node: Optional hou.Node whose 'status_message' parm receives
                     progress updates every 5 %.

    Returns:
        float32 array of shape [num_frames, num_bones, 4, 4].
    """
    num_frames = len(frames)
    num_bones = len(bones)
    result = np.zeros((num_frames, num_bones, 4, 4), dtype=np.float32)

    report_interval = max(1, num_frames // 20)  # ~5 % steps

    for fi, frame in enumerate(frames):
        t = hou.frameToTime(frame)
        for bi, bone in enumerate(bones):
            m = bone.worldTransformAtTime(t)
            # hou.Matrix4 → numpy 4×4 row-major
            result[fi, bi] = np.array(m.asTuple(), dtype=np.float32).reshape(4, 4)

        if fi % report_interval == 0 or fi == num_frames - 1:
            if progress_cb is not None:
                progress_cb(fi, num_frames)
            if status_node is not None:
                pct = 100 * (fi + 1) // num_frames
                try:
                    status_node.parm("status_message").set(
                        f"Sampling transforms: {pct}% ({fi+1}/{num_frames} frames)"
                    )
                except Exception:
                    pass

    return result


def matrix_to_quat(mats: np.ndarray) -> np.ndarray:
    """Convert a batch of 3x3 rotation matrices to quaternions (xyzw) using NumPy.
    
    mats: [N, 3, 3] float32.
    Returns: [N, 4] float32 (xyzw).
    """
    # mats is [N, 3, 3]
    # trace = m00 + m11 + m22
    t0 = mats[:, 0, 0]
    t1 = mats[:, 1, 1]
    t2 = mats[:, 2, 2]
    trace = t0 + t1 + t2
    
    N = mats.shape[0]
    q = np.zeros((N, 4), dtype=np.float32)
    
    # Shoemake's algorithm — safe division to avoid NaN / RuntimeWarning
    # from degenerate (zero-scale) transforms.

    def _safe_div(num, denom):
        """Divide avoiding RuntimeWarning; return 0 where denom is tiny."""
        out = np.zeros_like(num, dtype=np.float64)
        mask = np.abs(denom) > 1e-7
        np.divide(num, denom, out=out, where=mask)
        return out.astype(np.float32)

    # Case 1: Trace > 0
    c1 = trace > 0
    if np.any(c1):
        s = np.sqrt(np.maximum(0.0, trace[c1] + 1.0)) * 2.0  # s = 4*qw
        q[c1, 3] = 0.25 * s
        q[c1, 0] = _safe_div(mats[c1, 1, 2] - mats[c1, 2, 1], s)
        q[c1, 1] = _safe_div(mats[c1, 2, 0] - mats[c1, 0, 2], s)
        q[c1, 2] = _safe_div(mats[c1, 0, 1] - mats[c1, 1, 0], s)

    # Case 2: m00 is the largest diagonal element
    c2 = (~c1) & (t0 > t1) & (t0 > t2)
    if np.any(c2):
        s = np.sqrt(np.maximum(0.0, 1.0 + t0[c2] - t1[c2] - t2[c2])) * 2.0
        q[c2, 3] = _safe_div(mats[c2, 1, 2] - mats[c2, 2, 1], s)
        q[c2, 0] = 0.25 * s
        q[c2, 1] = _safe_div(mats[c2, 0, 1] + mats[c2, 1, 0], s)
        q[c2, 2] = _safe_div(mats[c2, 0, 2] + mats[c2, 2, 0], s)

    # Case 3: m11 is the largest diagonal element
    c3 = (~c1) & (~c2) & (t1 > t2)
    if np.any(c3):
        s = np.sqrt(np.maximum(0.0, 1.0 + t1[c3] - t0[c3] - t2[c3])) * 2.0
        q[c3, 3] = _safe_div(mats[c3, 2, 0] - mats[c3, 0, 2], s)
        q[c3, 0] = _safe_div(mats[c3, 0, 1] + mats[c3, 1, 0], s)
        q[c3, 1] = 0.25 * s
        q[c3, 2] = _safe_div(mats[c3, 1, 2] + mats[c3, 2, 1], s)

    # Case 4: m22 is the largest
    c4 = (~c1) & (~c2) & (~c3)
    if np.any(c4):
        s = np.sqrt(np.maximum(0.0, 1.0 + t2[c4] - t0[c4] - t1[c4])) * 2.0
        q[c4, 3] = _safe_div(mats[c4, 0, 1] - mats[c4, 1, 0], s)
        q[c4, 0] = _safe_div(mats[c4, 0, 2] + mats[c4, 2, 0], s)
        q[c4, 1] = _safe_div(mats[c4, 1, 2] + mats[c4, 2, 1], s)
        q[c4, 2] = 0.25 * s

    # Default to identity quaternion for any degenerate matrices
    invalid = np.isnan(q).any(axis=1) | np.isinf(q).any(axis=1) | (np.abs(q).sum(axis=1) < 1e-8)
    if np.any(invalid):
        q[invalid] = [0.0, 0.0, 0.0, 1.0]

    # Normalize to unit quaternions
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    norms = np.where(norms > 1e-8, norms, 1.0)
    q = q / norms

    return q


def world_to_local_quats(world_xforms: np.ndarray,
                          parent_indices: list[int]) -> np.ndarray:
    """Convert world-space 4×4 transforms to local quaternions (xyzw).

    local_xform[f, b] = inv(world_xform[f, parent(b)]) @ world_xform[f, b]
    For root bones (parent_idx == -1) the local xform equals the world xform.

    Args:
        world_xforms:   [num_frames, num_bones, 4, 4] float32.
        parent_indices: List of length num_bones; -1 for roots.

    Returns:
        float32 array of shape [num_frames, num_bones, 4] (quaternion xyzw).
    """
    num_frames, num_bones = world_xforms.shape[:2]
    if len(parent_indices) != num_bones:
        raise ValueError(
            f"world_to_local_quats: parent_indices length ({len(parent_indices)}) "
            f"does not match world_xforms bone axis ({num_bones}). "
            "Pass num_bones to sample_world_transforms_from_geo to synchronise counts."
        )
    quats = np.zeros((num_frames, num_bones, 4), dtype=np.float32)

    for b, pidx in enumerate(parent_indices):
        # Safety: ensure pidx is within valid range [0, num_bones-1]
        if pidx == -1 or pidx >= num_bones or pidx < -1:
            local_rot_mats = world_xforms[:, b, :3, :3]  # [F, 3, 3]
        else:
            # Rb = Rl @ Rp → Rl = Rb @ inv(Rp) (for row vectors)
            parent_mats = world_xforms[:, pidx, :3, :3]  # [F, 3, 3]
            parent_inv = np.transpose(parent_mats, (0, 2, 1))
            local_rot_mats = np.einsum("fij,fjk->fik", world_xforms[:, b, :3, :3], parent_inv)

        # local_rot_mats are row-major Houdini rotations.
        # matrix_to_quat expects column-major math matrices M where v_out = M @ v_in.
        # Since R_houdini = M.T, we transpose.
        col_mats = np.transpose(local_rot_mats, (0, 2, 1))
        quats[:, b, :] = matrix_to_quat(col_mats)

    return quats
