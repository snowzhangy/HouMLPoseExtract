"""
fps_sampler.py
Farthest Point Sampling over joint quaternion distances.
Matches the algorithm used in UE5's MLDeformer TrainingDataProcessor.
Pure numpy — no Houdini dependency.
"""
from __future__ import annotations

import numpy as np


def farthest_point_sampling(quats: np.ndarray,
                             num_output: int,
                             diversity_bone_indices: list[int],
                             seed_indices: list[int] | None = None) -> list[int]:
    """Select *num_output* maximally-diverse frames via Farthest Point Sampling.

    Args:
        quats:                  [N_frames, N_bones, 4] float32 quaternions (xyzw).
        num_output:             Target number of selected frames.
        diversity_bone_indices: Indices into axis-1 of *quats* used for distance.
        seed_indices:           List of frame indices to force-select first (e.g. [0] for rest).

    Returns:
        List of selected frame indices (length <= num_output).
    """
    n_frames = quats.shape[0]
    num_output = min(num_output, n_frames)

    if len(diversity_bone_indices) == 0:
        return list(np.linspace(0, n_frames - 1, num_output, dtype=int))

    div_quats = quats[:, diversity_bone_indices, :].reshape(n_frames, -1).astype(np.float64)
    
    selected: list[int] = []
    selected_set: set[int] = set()

    # 1. Start with seed indices (e.g. Rest Pose)
    if seed_indices:
        for idx in seed_indices:
            if idx < n_frames and idx not in selected_set:
                selected.append(idx)
                selected_set.add(idx)
    
    # If no seeds provided, start with the frame farthest from the zero vector
    if not selected:
        dist_to_zero = np.mean(div_quats ** 2, axis=1)
        first_idx = int(np.argmax(dist_to_zero))
        selected.append(first_idx)
        selected_set.add(first_idx)

    # 2. Initialize distances from the current selection
    # dist_to_selected[i] = min distance from frame i to ANY already selected frame
    dist_to_selected = np.full(n_frames, np.inf, dtype=np.float64)
    for idx in selected:
        diff = div_quats - div_quats[idx]
        new_dist = np.mean(diff ** 2, axis=1)
        dist_to_selected = np.minimum(dist_to_selected, new_dist)

    # 3. Fill the rest of the slots
    while len(selected) < num_output:
        # Pick the frame farthest from all already-selected frames
        dist_to_selected[list(selected_set)] = -1.0
        next_idx = int(np.argmax(dist_to_selected))
        
        if dist_to_selected[next_idx] < 0:
            break # No more frames to pick

        selected.append(next_idx)
        selected_set.add(next_idx)

        # Update minimum distances: compare every frame against the new selection
        diff = div_quats - div_quats[next_idx]
        new_dist = np.mean(diff ** 2, axis=1)
        dist_to_selected = np.minimum(dist_to_selected, new_dist)

    return selected[:num_output]


def compute_distance_matrix(quats_a: np.ndarray, quats_b: np.ndarray) -> float:
    """Compute mean-squared distance between two quaternion pose vectors.

    Args:
        quats_a: [N_bones, 4] float32.
        quats_b: [N_bones, 4] float32.

    Returns:
        Scalar distance value (float).
    """
    diff = quats_a.astype(np.float64) - quats_b.astype(np.float64)
    return float(np.mean(diff ** 2))
