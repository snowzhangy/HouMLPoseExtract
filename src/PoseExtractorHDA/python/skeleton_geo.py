"""
skeleton_geo.py
Build KineFX-compatible skeleton geometry for the HDA's cook-time output.
"""
from __future__ import annotations

import numpy as np

try:
    import hou
except ImportError:
    hou = None  # test context


def build_skeleton_geometry(geo,
                              bones: list,
                              bone_names: list[str],
                              parent_indices: list[int],
                              world_xforms_at_frame: np.ndarray) -> None:
    """Populate *geo* with a KineFX-compatible skeleton for one frame.

    Creates:
    - One point per bone at the joint's world-space origin.
    - Point attributes: name (string), transform (9-float Matrix3), parentidx (int).
    - Open polyline primitives connecting each child to its parent.

    Args:
        geo:                   hou.Geometry to populate (cleared first).
        bones:                 Unused (kept for API compat).
        bone_names:            List of bone name strings.
        parent_indices:        parent_indices[i] = index of parent, -1 for roots.
        world_xforms_at_frame: float64 [N_bones, 4, 4] world transforms this frame.
    """
    geo.clear()

    num_bones = len(bone_names)
    if world_xforms_at_frame.shape[0] < num_bones:
        num_bones = world_xforms_at_frame.shape[0]

    if num_bones == 0:
        return

    # ------------------------------------------------------------------ attrs
    identity3 = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    identity4 = (1.0, 0.0, 0.0, 0.0,
                 0.0, 1.0, 0.0, 0.0,
                 0.0, 0.0, 1.0, 0.0,
                 0.0, 0.0, 0.0, 1.0)

    geo.addAttrib(hou.attribType.Point, "name", "")
    geo.addAttrib(hou.attribType.Point, "transform", identity3)
    geo.addAttrib(hou.attribType.Point, "worldtransform", identity4)
    geo.addAttrib(hou.attribType.Point, "parentidx", -1)

    # ------------------------------------------------------------ create points
    point_list = []
    for bi in range(num_bones):
        pt = geo.createPoint()
        m = world_xforms_at_frame[bi]  # [4, 4]

        # Sanitise the entire matrix once
        if np.any(np.isnan(m)) or np.any(np.isinf(m)):
            m = np.eye(4, dtype=np.float64)

        pos_vals = (float(m[3, 0]), float(m[3, 1]), float(m[3, 2]))
        pt.setPosition(hou.Vector3(pos_vals[0], pos_vals[1], pos_vals[2]))

        pt.setAttribValue("name", bone_names[bi])
        pt.setAttribValue("parentidx", parent_indices[bi])

        # Local 3x3 rotation
        rot3 = m[:3, :3]
        xform3 = (
            float(rot3[0, 0]), float(rot3[0, 1]), float(rot3[0, 2]),
            float(rot3[1, 0]), float(rot3[1, 1]), float(rot3[1, 2]),
            float(rot3[2, 0]), float(rot3[2, 1]), float(rot3[2, 2]),
        )
        pt.setAttribValue("transform", xform3)

        # Full 4x4 world transform (required by FBX ROP)
        xform4 = tuple(float(m[r, c]) for r in range(4) for c in range(4))
        pt.setAttribValue("worldtransform", xform4)

        point_list.append(pt)

    # ----------------------------------------------- polyline bone connections
    for bi in range(num_bones):
        pidx = parent_indices[bi]
        if pidx < 0 or pidx >= num_bones:
            continue
        poly = geo.createPolygon(is_closed=False)
        poly.addVertex(point_list[pidx])
        poly.addVertex(point_list[bi])


def set_intersection_flag(geo, is_intersecting: bool) -> None:
    """Add or update the detail attribute 'intersect' (int) on *geo*.

    Args:
        geo:             hou.Geometry to annotate.
        is_intersecting: True → attribute value 1; False → 0.
    """
    attr = geo.findGlobalAttrib("intersect")
    if attr is None:
        attr = geo.addAttrib(hou.attribType.Global, "intersect", 0)
    geo.setGlobalAttribValue("intersect", int(is_intersecting))
