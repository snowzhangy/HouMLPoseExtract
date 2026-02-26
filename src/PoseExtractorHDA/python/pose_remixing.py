"""
pose_remixing.py
Bone-group shuffling augmentation — inspired by UE5 MLDeformer pose remixing.
Pure numpy — no Houdini dependency.
"""
from __future__ import annotations

import numpy as np


def remix_poses(quats: np.ndarray,
                bone_groups: dict[str, list[int]],
                random_seed: int) -> np.ndarray:
    """Create augmented poses by shuffling bone-group rotations across frames.

    For each bone group, the frame assignments for those bones are independently
    permuted (Fisher-Yates shuffle), producing synthetic combinations of poses
    that did not appear in the original animation.

    Args:
        quats:       [N_frames, N_bones, 4] float32 — source quaternions.
                     This array is NOT modified; a new array is returned.
        bone_groups: Dict mapping group_name → list of bone indices (axis-1).
        random_seed: Integer seed for reproducibility.

    Returns:
        New float32 array of shape [N_frames, N_bones, 4] with shuffled groups.
    """
    n_frames = quats.shape[0]
    quats_out = quats.copy()

    rng = np.random.default_rng(random_seed)

    for group_name, bone_indices in bone_groups.items():
        if len(bone_indices) == 0:
            continue
        # Independent Fisher-Yates permutation for this group
        perm = rng.permutation(n_frames)   # shape [N_frames]
        # For each output frame i, take bone-group rotations from source frame perm[i]
        quats_out[:, bone_indices, :] = quats[perm, :, :][:, bone_indices, :]

    return quats_out


def build_bone_groups(bone_names: list[str],
                      group_defs: list[dict[str, str]]) -> dict[str, list[int]]:
    """Resolve fnmatch patterns into lists of bone indices.

    Args:
        bone_names: Ordered list of all bone names.
        group_defs: List of {'name': str, 'pattern': str} dicts from HDA multiparm.

    Returns:
        Dict { group_name: [bone_indices] }.
    """
    import fnmatch

    groups: dict[str, list[int]] = {}
    for gdef in group_defs:
        name = gdef.get("name", "")
        pattern = gdef.get("pattern", "")
        if not name or not pattern:
            continue
        indices = [
            i for i, bname in enumerate(bone_names)
            if fnmatch.fnmatch(bname, pattern)
        ]
        if indices:
            groups[name] = indices

    return groups
