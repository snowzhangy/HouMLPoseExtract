"""
lbs_intersection.py
Linear Blend Skinning deformation and body-part self-intersection detection.
"""
from __future__ import annotations

import fnmatch

import numpy as np

try:
    from scipy.spatial import KDTree
except ImportError:
    KDTree = None  # type: ignore[assignment,misc]

try:
    import hou
except ImportError:
    hou = None  # test context


# ---------------------------------------------------------------------------
# Bone capture reading
# ---------------------------------------------------------------------------

def read_bone_capture(geo, bones: list) -> tuple[np.ndarray, np.ndarray]:
    """Read boneCapture point attribute from a Houdini geometry object.

    Args:
        geo:   hou.Geometry at rest frame.
        bones: Ordered list of hou.ObjNode (the target bone set).

    Returns:
        weights:      float32 [num_vertices, num_bones]
        bone_indices: int32 [num_vertices, max_influences]
    """
    num_vertices = len(geo.points())
    num_bones = len(bones)

    capture_attr = geo.findPointAttrib("boneCapture")
    if capture_attr is None:
        raise ValueError("Input geometry has no 'boneCapture' point attribute.")

    # Map bone paths from capture regions to indices in our *bones* list
    # geo.attributeCaptureRegions() returns tuple of bone paths for each index
    try:
        capture_paths = geo.attributeCaptureRegions()
    except Exception:
        # Fallback for older Houdini or missing metadata
        capture_paths = []

    # Build mapping from capture index → our bone index
    path_to_idx = {b.path(): i for i, b in enumerate(bones)}
    capture_to_bone_idx = {}
    for ci, path in enumerate(capture_paths):
        # Resolve path (could be absolute or relative to some root)
        # If it's relative, we might need more logic, but try direct lookup first.
        target_idx = path_to_idx.get(path, -1)
        if target_idx == -1:
            # Try finding the node to get its absolute path
            n = hou.node(path)
            if n:
                target_idx = path_to_idx.get(n.path(), -1)
        capture_to_bone_idx[ci] = target_idx

    weights = np.zeros((num_vertices, num_bones), dtype=np.float32)
    max_influences = 0
    if num_vertices > 0:
        sample_data = geo.points()[0].attribValue("boneCapture")
        max_influences = len(sample_data) // 2

    bone_indices_raw = np.full((num_vertices, max_influences), -1, dtype=np.int32)

    for vi, pt in enumerate(geo.points()):
        data = pt.attribValue("boneCapture")
        for pair_idx in range(len(data) // 2):
            raw_idx = int(data[pair_idx * 2])
            weight = float(data[pair_idx * 2 + 1])
            
            if raw_idx in capture_to_bone_idx:
                target_idx = capture_to_bone_idx[raw_idx]
                if target_idx >= 0:
                    weights[vi, target_idx] += weight
                    if pair_idx < max_influences:
                        bone_indices_raw[vi, pair_idx] = target_idx

    return weights, bone_indices_raw


# ---------------------------------------------------------------------------
# LBS deformation
# ---------------------------------------------------------------------------

def compute_lbs(rest_positions: np.ndarray,
                weights: np.ndarray,
                rest_world_xforms_inv: np.ndarray,
                world_xforms: np.ndarray) -> np.ndarray:
    """Linear Blend Skinning deformation (vectorized).

    Each vertex is transformed by the weighted sum of per-bone skinning matrices:
        p_world[v] = sum_b( weights[v,b] * world_xforms[b] @ rest_inv[b] @ p_rest_hom[v] )

    Args:
        rest_positions:       float32 [V, 3] — rest-pose vertex positions.
        weights:              float32 [V, B] — skinning weights (rows sum to ≤1).
        rest_world_xforms_inv: float32 [B, 4, 4] — inverse rest-pose world transforms.
        world_xforms:         float32 [B, 4, 4] — current-frame world transforms.

    Returns:
        deformed_positions: float32 [V, 3].
    """
    V = rest_positions.shape[0]
    B = world_xforms.shape[0]

    # Homogeneous rest positions [V, 4]
    ones = np.ones((V, 1), dtype=np.float32)
    p_rest_hom = np.concatenate([rest_positions, ones], axis=1)  # [V, 4]

    # Per-bone skinning matrix: M_bind[b] = rest_inv[b] @ world[b]  [B, 4, 4]
    # In row-major, composing transforms A then B is A @ B.
    M_bind = np.einsum("bij,bjk->bik", rest_world_xforms_inv.astype(np.float32),
                       world_xforms.astype(np.float32))  # [B, 4, 4]

    # Weighted contribution of each bone to each vertex
    # Each vertex (row vector) is multiplied by the skinning matrix (row-major).
    # p_world[v] = sum_b( weights[v,b] * p_rest_hom[v] @ M_bind[b] )
    deformed_hom = np.einsum("vb,vj,bjk->vk",
                              weights.astype(np.float32),
                              p_rest_hom,
                              M_bind)  # [V, 4]

    return deformed_hom[:, :3]


# ---------------------------------------------------------------------------
# Body-part vertex map
# ---------------------------------------------------------------------------

def build_body_part_vertex_map(weights: np.ndarray,
                                bone_name_to_idx: dict[str, int],
                                body_part_defs: dict[str, list[str]],
                                weight_threshold: float = 0.01) -> dict[str, np.ndarray]:
    """Map body-part names to the vertex indices most influenced by their bones.

    Args:
        weights:          float32 [V, B] skinning weights.
        bone_name_to_idx: Dict mapping bone name → column index in *weights*.
        body_part_defs:   Dict { part_name: [bone_names_matching_pattern] }.
        weight_threshold: Minimum bone weight for a vertex to belong to a part.

    Returns:
        Dict { part_name: int32 array of vertex indices }.
    """
    part_vertex_map: dict[str, np.ndarray] = {}

    for part_name, bone_names in body_part_defs.items():
        # Collect column indices for all bones in this body part
        bone_cols = [bone_name_to_idx[bn] for bn in bone_names
                     if bn in bone_name_to_idx]
        if not bone_cols:
            part_vertex_map[part_name] = np.array([], dtype=np.int32)
            continue

        # A vertex belongs to this part if any of its part-bones has weight > threshold
        part_weights = weights[:, bone_cols]  # [V, len(bone_cols)]
        mask = np.any(part_weights > weight_threshold, axis=1)
        part_vertex_map[part_name] = np.where(mask)[0].astype(np.int32)

    return part_vertex_map


def resolve_body_part_defs(bone_names: list[str],
                            raw_defs: list[dict[str, str]]) -> dict[str, list[str]]:
    """Resolve fnmatch patterns in body-part definitions to concrete bone name lists.

    Args:
        bone_names: All bone names in the skeleton.
        raw_defs:   List of {'name': str, 'bone_pattern': str} from HDA multiparm.

    Returns:
        Dict { part_name: [matched_bone_names] }.
    """
    result: dict[str, list[str]] = {}
    for d in raw_defs:
        name = d.get("name", "")
        pattern = d.get("bone_pattern", "")
        if not name or not pattern:
            continue
        matched = [bn for bn in bone_names if fnmatch.fnmatch(bn, pattern)]
        result[name] = matched
    return result


# ---------------------------------------------------------------------------
# Intersection detection
# ---------------------------------------------------------------------------

def check_intersection_pair(verts_a: np.ndarray,
                              verts_b: np.ndarray,
                              threshold: float) -> bool:
    """Detect geometric intersection between two sets of vertices.

    Builds a KDTree on *verts_b* and queries nearest neighbours for each
    vertex in *verts_a*. Returns True if any distance is below *threshold*.

    Args:
        verts_a:   float32 [Na, 3] — vertices for body part A.
        verts_b:   float32 [Nb, 3] — vertices for body part B.
        threshold: Penetration depth in scene units.

    Returns:
        True if an intersection is detected.
    """
    if KDTree is None:
        raise ImportError("scipy is required for intersection checking. "
                          "Please install it in your Houdini Python environment.")
    if len(verts_a) == 0 or len(verts_b) == 0:
        return False

    tree = KDTree(verts_b)
    dists, _ = tree.query(verts_a, k=1, workers=-1)
    return bool(np.any(dists < threshold))


def filter_by_intersection(selected_frame_indices: list[int],
                            frames: list[int],
                            bones: list,
                            rest_positions: np.ndarray,
                            weights: np.ndarray,
                            rest_xforms_inv: np.ndarray,
                            world_xforms: np.ndarray,
                            body_part_vertex_map: dict[str, np.ndarray],
                            intersection_pairs: list[tuple[str, str]],
                            threshold: float,
                            action: int) -> tuple[list[int], list[int]]:
    """Filter or flag selected frame indices based on self-intersection.

    Args:
        selected_frame_indices: Indices into *frames* list to evaluate.
        frames:        Full list of sampled frame numbers.
        bones:         List of hou.ObjNode (for transform lookup if needed).
        rest_positions: float32 [V, 3] rest-pose vertex positions.
        weights:       float32 [V, B] skinning weights.
        rest_xforms_inv: float32 [B, 4, 4] inverse rest transforms.
        world_xforms:  float32 [F, B, 4, 4] all-frames world transforms.
        body_part_vertex_map: { part_name: vertex_indices }.
        intersection_pairs: List of (part_a, part_b) tuples to check.
        threshold:     Penetration depth threshold.
        action:        0 = exclude intersecting poses, 1 = flag only.

    Returns:
        (valid_indices, flagged_indices) — both are subsets of selected_frame_indices.
    """
    valid: list[int] = []
    flagged: list[int] = []

    for sel_idx in selected_frame_indices:
        frame_world_xforms = world_xforms[sel_idx]  # [B, 4, 4]
        deformed = compute_lbs(rest_positions, weights, rest_xforms_inv, frame_world_xforms)

        intersecting = False
        for part_a, part_b in intersection_pairs:
            verts_a_idx = body_part_vertex_map.get(part_a, np.array([], dtype=np.int32))
            verts_b_idx = body_part_vertex_map.get(part_b, np.array([], dtype=np.int32))
            if len(verts_a_idx) == 0 or len(verts_b_idx) == 0:
                continue
            verts_a = deformed[verts_a_idx]
            verts_b = deformed[verts_b_idx]
            if check_intersection_pair(verts_a, verts_b, threshold):
                intersecting = True
                break

        if intersecting:
            flagged.append(sel_idx)
            if action == 1:  # flag only — keep the pose
                valid.append(sel_idx)
            # action == 0 → exclude → do not append to valid
        else:
            valid.append(sel_idx)

    return valid, flagged
