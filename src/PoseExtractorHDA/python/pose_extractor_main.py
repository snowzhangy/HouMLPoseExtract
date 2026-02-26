"""
pose_extractor_main.py
HDA button callback orchestrator and cook-time output builder.

Two-pass architecture:
  Pass 1 (button click)  → extract_poses(kwargs)
  Pass 2 (cook time)     → cook(kwargs)
"""
from __future__ import annotations

import fnmatch
import json
import sys
import os

import numpy as np

# ---------------------------------------------------------------------------
# Allow the HDA's Python module to import its sibling modules.
# ---------------------------------------------------------------------------
try:
    _HERE = os.path.dirname(__file__)
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
except NameError:
    # Inside Houdini HDA PythonModule, __file__ is not defined.
    pass

import bone_utils          # noqa: E402
import fps_sampler         # noqa: E402
import pose_remixing       # noqa: E402
import lbs_intersection    # noqa: E402
import skeleton_geo        # noqa: E402

try:
    import hou
except ImportError:
    hou = None

# Re-entrancy guard — prevents recursive cook when geometryAtFrame()
# on upstream nodes triggers dependency graph re-evaluation.
_cook_in_progress = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_status(node, msg: str) -> None:
    """Update the 'status_message' parameter on *node* (best-effort).

    Skipped when called during cook to avoid dirtying the node and
    triggering infinite re-evaluation.
    """
    if _cook_in_progress:
        return
    try:
        node.parm("status_message").set(str(msg))
    except Exception:
        pass


def _resolve_bone_indices(pattern: str, bone_names: list[str]) -> list[int]:
    """Return indices of bone names matching *pattern* (fnmatch)."""
    return [i for i, n in enumerate(bone_names) if fnmatch.fnmatch(n, pattern)]


def _read_bone_groups(node) -> dict[str, list[int]]:
    """Read bone-group multiparm entries and resolve them to bone-index lists."""
    # Bone names must already be stored in userData at this point
    bone_names_raw = node.userData("bone_names")
    if not bone_names_raw:
        return {}
    bone_names: list[str] = json.loads(bone_names_raw)

    num_groups = node.parm("bone_groups").eval()
    group_defs = []
    for i in range(1, num_groups + 1):
        group_defs.append({
            "name": node.parm(f"bone_group_name{i}").eval(),
            "pattern": node.parm(f"bone_group_pattern{i}").eval(),
        })

    return pose_remixing.build_bone_groups(bone_names, group_defs)


def _read_body_part_defs(node, bone_names: list[str]) -> dict[str, list[str]]:
    """Read body-part multiparm entries and resolve bone name patterns."""
    num_parts = node.parm("body_parts").eval()
    raw_defs = []
    for i in range(1, num_parts + 1):
        raw_defs.append({
            "name": node.parm(f"body_part_name{i}").eval(),
            "bone_pattern": node.parm(f"body_part_bone_pattern{i}").eval(),
        })
    return lbs_intersection.resolve_body_part_defs(bone_names, raw_defs)


def _parse_intersection_pairs(pairs_str: str) -> list[tuple[str, str]]:
    """Parse 'left_arm:torso,right_arm:torso' into list of (str,str) tuples."""
    pairs: list[tuple[str, str]] = []
    for tok in pairs_str.split(","):
        tok = tok.strip()
        if ":" in tok:
            a, b = tok.split(":", 1)
            pairs.append((a.strip(), b.strip()))
    return pairs


def _has_kinefx_skeleton(geo) -> bool:
    """Return True if *geo* has a 'name' point attribute (KineFX skeleton)."""
    return geo is not None and geo.findPointAttrib("name") is not None


def _find_skeleton_source(hda_node):
    """Auto-detect the first KineFX skeleton from wired inputs.

    For multi-output nodes (like fbxcharacterimport) we must use
    ``inputGeometry()`` on a downstream node — it is the only API in
    Houdini 20.5 that correctly resolves which output is wired.

    Search order:
      1. HDA Input 1.
      2. HDA Input 0 (bonedeform) input 2 — animated skeleton.
      3. HDA Input 0 (bonedeform) input 1 — rest skeleton.

    Returns:
        (accessor_node, input_index) so callers can use
        ``accessor_node.inputGeometry(input_index)`` to read skeleton
        geometry at the current frame, or
        ``bone_utils.geo_at_frame_via_input(accessor_node, input_index, frame)``
        to read at a specific frame.
        Returns ``(None, -1)`` if not found.
    """
    candidates = []  # (accessor_node, accessor_input_idx)

    # 1. Direct HDA Input 1
    if hda_node.input(1) is not None:
        candidates.append((hda_node, 1))

    # 2. bonedeform (HDA Input 0) skeleton inputs
    bonedeform = hda_node.input(0)
    if bonedeform is not None:
        for idx in (2, 1):
            try:
                if bonedeform.input(idx) is not None:
                    candidates.append((bonedeform, idx))
            except Exception:
                pass

    for acc_node, acc_idx in candidates:
        try:
            geo = acc_node.inputGeometry(acc_idx)
        except Exception:
            continue
        if _has_kinefx_skeleton(geo):
            return acc_node, acc_idx

    return None, -1


def _find_all_skeleton_inputs(hda_node):
    """Discover all connected KineFX skeleton sources.

    Checks wired inputs 1–3, then any extra animation SOPs listed in the
    ``extra_anim_sources`` multiparm (for when >3 animations are needed).

    Each returned entry is ``(accessor_node, accessor_input_idx)`` where
    ``accessor_input_idx`` is the HDA input index for wired connectors,
    or -1 when the accessor is a referenced SOP (read its own output).

    Falls back to the bonedeform sub-input search if nothing is found.
    """
    sources = []

    # 1. Wired HDA Inputs 1+ (animation skeletons, dynamic like Merge SOP)
    all_inputs = hda_node.inputs()
    for input_idx in range(1, len(all_inputs)):
        if all_inputs[input_idx] is None:
            continue
        try:
            geo = hda_node.inputGeometry(input_idx)
        except Exception:
            geo = None
        if _has_kinefx_skeleton(geo):
            sources.append((hda_node, input_idx))

    # 2. Extra animation sources from multiparm (overflow beyond 3 wired inputs)
    try:
        num_extra = hda_node.parm("extra_anim_sources").eval()
    except Exception:
        num_extra = 0
    for i in range(1, num_extra + 1):
        path = hda_node.parm(f"extra_anim_path{i}").eval()
        if not path:
            continue
        ref_node = hda_node.node(path)
        if ref_node is None:
            continue
        try:
            geo = ref_node.geometry()
        except Exception:
            geo = None
        if _has_kinefx_skeleton(geo):
            # Use (ref_node, -1) — read the node's own output geometry
            sources.append((ref_node, -1))

    if sources:
        return sources

    # Fallback: single-input legacy path (bonedeform sub-inputs)
    acc_node, acc_idx = _find_skeleton_source(hda_node)
    if acc_node is not None:
        return [(acc_node, acc_idx)]
    return []


# ---------------------------------------------------------------------------
# Pass 1 — core extraction pipeline (no UI side-effects)
# ---------------------------------------------------------------------------

def _run_extraction(node, progress_cb=None) -> None:
    """Core extraction pipeline: sample transforms, run FPS, store results.

    This function is called both by the button callback and by auto-extract
    on first cook.  It never calls ``node.cook()`` — results are stored in
    ``node.userData`` for the cook function to read immediately after.

    Args:
        progress_cb: Optional callable(fraction) where fraction is 0.0–1.0.
                     Used by ``hou.InterruptableOperation.updateProgress``.
    """
    # -------------------------------------------------------- parameters
    char_root_node = node.parm("char_root").evalAsNode()
    char_root = char_root_node.path() if char_root_node else ""

    rest_frame = node.parm("rest_frame").eval()
    frame_start = node.parm("frame_start").eval()
    frame_end = node.parm("frame_end").eval()
    frame_step = max(1, node.parm("frame_step").eval())
    num_output_poses = node.parm("num_output_poses").eval()
    diversity_bone_filter = node.parm("diversity_bone_filter").eval() or "*"
    enable_remixing = bool(node.parm("enable_remixing").eval())
    random_seed = node.parm("random_seed").eval()
    check_intersection = bool(node.parm("check_intersection").eval())

    # Progress weighting: sampling=0..0.80, quats=0.80..0.85, FPS=0.85..0.95, rest=0.95..1.0
    def _progress(fraction):
        if progress_cb:
            progress_cb(max(0.0, min(1.0, fraction)))

    # -------------------------------------------- detect skeleton sources
    _set_status(node, "Finding bones...")
    _progress(0.0)
    skel_sources = _find_all_skeleton_inputs(node)
    use_kinefx = len(skel_sources) > 0

    if use_kinefx:
        # Build hierarchy from first input's rest frame (all inputs share same skeleton)
        first_acc_node, first_acc_idx = skel_sources[0]
        if first_acc_idx >= 0:
            rest_skel = bone_utils.get_input_geometry_at_frame(first_acc_node, first_acc_idx, float(rest_frame))
            if rest_skel is None:
                rest_skel = first_acc_node.geometryAtFrame(float(rest_frame))
        else:
            rest_skel = first_acc_node.geometryAtFrame(float(rest_frame))
        if rest_skel is None:
            raise ValueError(f"Could not retrieve skeleton geometry at frame {rest_frame}.")
        bone_names, parent_indices, name_to_idx = bone_utils.build_hierarchy_from_geo(rest_skel)
        num_bones = len(bone_names)
        # Store source list so cook() can look up the right input
        skel_sources_data = [{"path": n.path(), "idx": idx} for n, idx in skel_sources]
        node.setUserData("skel_sources", json.dumps(skel_sources_data))
    elif char_root:
        bones = bone_utils.find_bones(char_root, "*")
        bone_names, parent_indices, name_to_idx = bone_utils.build_hierarchy(bones)
        num_bones = len(bone_names)
        node.setUserData("skel_sources", json.dumps([]))
    else:
        raise ValueError(
            "No skeleton found. Either:\n"
            "  - Connect KineFX skeleton SOPs to Inputs 1..9, or\n"
            "  - Connect a bonedeform SOP (with skeleton inputs) to Input 0, or\n"
            "  - Set the 'Character Root' parameter to an OBJ bone network."
        )

    # Cache bone names early so _read_bone_groups can use them
    node.setUserData("bone_names", json.dumps(bone_names))
    node.setUserData("parent_indices", json.dumps(parent_indices))

    # -------------------------------------------- sample transforms
    frames = list(range(frame_start, frame_end + 1, frame_step))
    if rest_frame not in frames:
        frames_with_rest = sorted(set(frames) | {rest_frame})
    else:
        frames_with_rest = frames

    if use_kinefx:
        # Sample from all connected skeleton inputs and stack
        all_world_xforms = []    # list of [F_i, B, 4, 4] arrays
        source_map = []          # [(input_source_idx, frame_num), ...] per row in stacked array

        num_src = len(skel_sources)
        for src_i, (acc_node, acc_idx) in enumerate(skel_sources):
            _set_status(node, f"Sampling input {src_i+1}/{num_src}: "
                        f"{len(frames_with_rest)} frames × {num_bones} bones...")
            # Per-source progress: spread 0.0–0.80 across all sources
            src_base = 0.80 * src_i / num_src
            src_span = 0.80 / num_src

            def _frame_progress(fi, total, _base=src_base, _span=src_span):
                _progress(_base + _span * fi / max(1, total))

            if acc_idx >= 0:
                xforms_i = bone_utils.sample_world_transforms_from_geo(
                    acc_node, acc_idx, frames_with_rest,
                    bone_names=bone_names, status_node=node,
                    progress_cb=_frame_progress,
                )
            else:
                xforms_i = bone_utils.sample_world_transforms_from_node(
                    acc_node, frames_with_rest,
                    bone_names=bone_names, status_node=node,
                    progress_cb=_frame_progress,
                )
            all_world_xforms.append(xforms_i)
            for f in frames_with_rest:
                source_map.append((src_i, f))

        world_xforms = np.concatenate(all_world_xforms, axis=0)  # [F_total, B, 4, 4]
        num_inputs = len(skel_sources)
        frames_per_input = len(frames_with_rest)

        # Build row indices for the sampling (non-rest) frames per input
        frame_to_row_in_input = {f: i for i, f in enumerate(frames_with_rest)}
        sampling_rows = []
        source_map_sampling = []  # [(input_source_idx, frame_num), ...] for sampling rows only
        for src_i in range(num_inputs):
            offset = src_i * frames_per_input
            for f in frames:
                sampling_rows.append(offset + frame_to_row_in_input[f])
                source_map_sampling.append((src_i, f))
    else:
        _set_status(node, f"Sampling {len(frames_with_rest)} frames × {num_bones} bones...")
        world_xforms = bone_utils.sample_world_transforms(
            bones, frames_with_rest, status_node=node,
            progress_cb=lambda fi, total: _progress(0.80 * fi / max(1, total)),
        )  # [F_all, B, 4, 4]

        frame_to_row_in_input = {f: i for i, f in enumerate(frames_with_rest)}
        sampling_rows = [frame_to_row_in_input[f] for f in frames]
        source_map_sampling = [(0, f) for f in frames]

    # ---------------------------------------------- local quaternions
    _set_status(node, "Converting to quaternions...")
    _progress(0.80)
    quats_all = bone_utils.world_to_local_quats(world_xforms, parent_indices)
    quats = quats_all[sampling_rows]  # [F_sampled_total, B, 4]

    # ------------------------------------------------- pose remixing
    if enable_remixing:
        _set_status(node, "Remixing poses...")
        bone_groups = _read_bone_groups(node)
        bone_groups_idx = pose_remixing.build_bone_groups(
            bone_names,
            [{"name": k, "pattern": "*"} for k in bone_groups]
        )
        quats = pose_remixing.remix_poses(quats, bone_groups, random_seed)

    # ----------------------------------------------- FPS selection
    _set_status(node, "Running FPS selection...")
    _progress(0.85)
    diversity_indices = _resolve_bone_indices(diversity_bone_filter, bone_names)
    if not diversity_indices:
        diversity_indices = list(range(len(bone_names)))

    # Find which indices in the sampling pool correspond to the rest_frame
    rest_idx_in_sampling = []
    if rest_frame in frames:
        for i, (src_i, f) in enumerate(source_map_sampling):
            if f == rest_frame and src_i == 0:
                rest_idx_in_sampling.append(i)
                break

    selected = fps_sampler.farthest_point_sampling(
        quats, num_output_poses, diversity_indices, seed_indices=rest_idx_in_sampling
    )

    # ------------------------------------------ intersection filter
    _progress(0.95)
    flagged: list[int] = []
    if check_intersection:
        _set_status(node, "Reading rest-pose geometry...")
        if node.inputs() and node.inputs()[0] is not None:
            rest_geo = node.inputs()[0].geometryAtFrame(float(rest_frame))
        else:
            raise ValueError("Input 0 is required for intersection checking (bonedeform output).")

        rest_positions_list = [list(pt.position()) for pt in rest_geo.points()]
        rest_positions = np.array(rest_positions_list, dtype=np.float32)

        cap_weights, _ = lbs_intersection.read_bone_capture(rest_geo, bones)

        # Use rest transforms from the first input
        rest_row = frame_to_row_in_input[rest_frame]
        rest_xforms = world_xforms[rest_row]           # [B,4,4]
        rest_xforms_inv = np.linalg.inv(rest_xforms)  # [B,4,4]

        body_part_defs = _read_body_part_defs(node, bone_names)
        body_part_map = lbs_intersection.build_body_part_vertex_map(
            cap_weights, name_to_idx, body_part_defs
        )
        pairs_str = node.parm("intersection_pairs").eval()
        intersection_pairs = _parse_intersection_pairs(pairs_str)
        threshold = node.parm("intersection_threshold").eval()
        action = node.parm("intersection_action").eval()

        _set_status(node, "Filtering intersections...")
        world_xforms_sampled = world_xforms[sampling_rows]
        selected, flagged = lbs_intersection.filter_by_intersection(
            selected, [sm[1] for sm in source_map_sampling], bones,
            rest_positions, cap_weights, rest_xforms_inv,
            world_xforms_sampled, body_part_map,
            intersection_pairs, threshold, action
        )

    # ----------------------------------------------- store results
    # New format: selected_poses = [[input_idx, frame], ...]
    selected_poses = [list(source_map_sampling[i]) for i in selected]
    node.setUserData("selected_poses", json.dumps(selected_poses))
    # Clear legacy key so cook() uses the new format
    node.setUserData("selected_frames", "")
    node.parm("num_selected_poses").set(len(selected_poses))

    input_counts = {}
    for src_i, _ in selected_poses:
        input_counts[src_i] = input_counts.get(src_i, 0) + 1
    source_summary = ", ".join(f"input {k+1}: {v}" for k, v in sorted(input_counts.items()))
    _set_status(node, f"Done: {len(selected_poses)} poses selected ({source_summary}).")


# ---------------------------------------------------------------------------
# Pass 1 — button callback (thin wrapper around _run_extraction)
# ---------------------------------------------------------------------------

def extract_poses(kwargs: dict) -> None:
    """Button callback: runs extraction with progress bar, forces recook."""
    node = kwargs["node"]

    try:
        with hou.InterruptableOperation(
            "Extracting Poses",
            long_operation_name="Sampling transforms and running FPS selection",
            open_interrupt_dialog=True,
        ) as operation:
            _run_extraction(node, progress_cb=operation.updateProgress)

        try:
            node.cook(force=True)
        except Exception:
            pass  # Data is saved in userData; cook will succeed on next frame change

    except hou.OperationInterrupted:
        _set_status(node, "Extraction cancelled.")
    except Exception as exc:
        _set_status(node, f"ERROR: {exc}")
        raise


# ---------------------------------------------------------------------------
# Pass 2 — cook callback
# ---------------------------------------------------------------------------

def cook(kwargs: dict) -> None:
    """Cook-time function: maps current output frame → original animation frame
    and builds skeleton geometry for that pose.

    IMPORTANT: This must be lightweight.  Never run extraction or set
    parameters here — that destabilises the dependency graph and crashes
    during ROP rendering.
    """
    global _cook_in_progress

    node = kwargs["node"]
    geo = kwargs["geo"]

    # Broad re-entrancy guard
    if _cook_in_progress:
        return
    _cook_in_progress = True
    try:
        _cook_inner(node, geo)
    except Exception:
        geo.clear()
    finally:
        _cook_in_progress = False


def _cook_inner(node, geo) -> None:
    """Actual cook logic, called under the _cook_in_progress guard."""

    raw_poses = node.userData("selected_poses")
    raw_bones = node.userData("bone_names")
    raw_parents = node.userData("parent_indices")

    if not raw_poses or not raw_bones or not raw_parents:
        # No extraction data — output empty geometry.
        # User must click "Re-Extract Poses" before rendering.
        geo.clear()
        return

    selected_poses: list[list[int]] = json.loads(raw_poses)
    bone_names: list[str] = json.loads(raw_bones)
    parent_indices: list[int] = json.loads(raw_parents)

    if not selected_poses:
        geo.clear()
        return

    # Map output frame → (input_source_idx, actual_frame)
    out_frame = int(hou.frame())
    pose_idx = max(0, min(out_frame - 1, len(selected_poses) - 1))
    input_source_idx, actual_frame = selected_poses[pose_idx]

    # -------------------------------------------------------- fetch skeleton
    # Read the skeleton at the target frame via the wired input.
    # Use node.inputGeometry() for the current-frame fast path; only
    # fall back to geometryAtFrame() when the target differs.
    current_skel_geo = None

    # Try wired input 1 first (primary animation skeleton)
    try:
        if int(hou.frame()) == actual_frame:
            current_skel_geo = node.inputGeometry(1)
        else:
            current_skel_geo = bone_utils.get_input_geometry_at_frame(
                node, 1, float(actual_frame)
            )
    except Exception:
        pass

    # Fallback: check stored skel_sources paths
    if current_skel_geo is None or not _has_kinefx_skeleton(current_skel_geo):
        raw_skel_sources = node.userData("skel_sources") or "[]"
        skel_sources_data = json.loads(raw_skel_sources)

        if skel_sources_data and input_source_idx < len(skel_sources_data):
            entry = skel_sources_data[input_source_idx]
            ref_node = hou.node(entry["path"])
            if ref_node is not None:
                try:
                    current_skel_geo = ref_node.geometryAtFrame(float(actual_frame))
                except Exception:
                    pass

    # Fallback: bonedeform sub-inputs
    if current_skel_geo is None or not _has_kinefx_skeleton(current_skel_geo):
        acc_node, acc_idx = _find_skeleton_source(node)
        if acc_node is not None:
            try:
                if acc_idx >= 0:
                    current_skel_geo = bone_utils.get_input_geometry_at_frame(
                        acc_node, acc_idx, float(actual_frame)
                    )
                else:
                    current_skel_geo = acc_node.geometryAtFrame(float(actual_frame))
            except Exception:
                pass

    if current_skel_geo is not None and _has_kinefx_skeleton(current_skel_geo):
        world_xforms = _read_world_xforms_from_geo(current_skel_geo, bone_names)
    else:
        # OBJ workflow
        char_root_node = node.parm("char_root").evalAsNode()
        if char_root_node is None:
            geo.clear()
            return
        try:
            bones = bone_utils.find_bones(char_root_node.path(), "*")
            world_xforms = bone_utils.sample_world_transforms(bones, [actual_frame])[0]
        except Exception:
            geo.clear()
            return

    # ----------------------------------------------------- build geometry
    skeleton_geo.build_skeleton_geometry(
        geo, None, bone_names, parent_indices, world_xforms
    )


def _read_world_xforms_from_geo(skel_geo, bone_names: list[str]) -> np.ndarray:
    """Extract world-space 4x4 matrices from a KineFX skeleton geometry."""
    num_bones = len(bone_names)
    world_xforms = np.zeros((num_bones, 4, 4), dtype=np.float64)
    for bi in range(num_bones):
        world_xforms[bi] = np.eye(4, dtype=np.float64)

    attr_world = skel_geo.findPointAttrib("worldtransform")
    attr_local = skel_geo.findPointAttrib("transform")
    attr_name = skel_geo.findPointAttrib("name")

    name_to_target_idx = {name: i for i, name in enumerate(bone_names)}

    for pt in skel_geo.points():
        if attr_name is not None:
            name = pt.stringAttribValue(attr_name)
            bi = name_to_target_idx.get(name, -1)
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
            continue

        if m is not None:
            m_val = m.asTuple() if hasattr(m, "asTuple") else m
            arr = np.array(m_val, dtype=np.float64).reshape(4, 4)
            if not np.any(np.isnan(arr)) and not np.any(np.isinf(arr)):
                world_xforms[bi] = arr

    return world_xforms
