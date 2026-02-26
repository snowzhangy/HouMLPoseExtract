"""
test_pose_remixing.py
Unit tests for pose_remixing.remix_poses and build_bone_groups.
No Houdini dependency required.
"""
from __future__ import annotations

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "PoseExtractorHDA", "python"))

import pose_remixing


class TestBuildBoneGroups:

    def test_basic_pattern(self):
        bone_names = ["spine", "left_arm", "left_elbow", "right_arm", "head"]
        group_defs = [
            {"name": "left", "pattern": "left_*"},
            {"name": "right", "pattern": "right_*"},
        ]
        groups = pose_remixing.build_bone_groups(bone_names, group_defs)
        assert groups["left"] == [1, 2]
        assert groups["right"] == [3]

    def test_wildcard_all(self):
        bone_names = ["a", "b", "c"]
        group_defs = [{"name": "all", "pattern": "*"}]
        groups = pose_remixing.build_bone_groups(bone_names, group_defs)
        assert groups["all"] == [0, 1, 2]

    def test_no_match(self):
        bone_names = ["spine", "head"]
        group_defs = [{"name": "arm", "pattern": "arm_*"}]
        groups = pose_remixing.build_bone_groups(bone_names, group_defs)
        assert "arm" not in groups  # empty groups are excluded

    def test_missing_name_skipped(self):
        bone_names = ["a", "b"]
        group_defs = [{"name": "", "pattern": "*"}]
        groups = pose_remixing.build_bone_groups(bone_names, group_defs)
        assert len(groups) == 0


class TestRemixPoses:

    def _make_quats(self, n_frames: int, n_bones: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.standard_normal((n_frames, n_bones, 4)).astype(np.float32)

    def test_output_shape(self):
        quats = self._make_quats(50, 10)
        groups = {"left": [0, 1], "right": [2, 3]}
        out = pose_remixing.remix_poses(quats, groups, random_seed=42)
        assert out.shape == quats.shape

    def test_does_not_modify_input(self):
        quats = self._make_quats(30, 6)
        original = quats.copy()
        groups = {"grp": [0, 1]}
        pose_remixing.remix_poses(quats, groups, random_seed=0)
        np.testing.assert_array_equal(quats, original)

    def test_non_group_bones_unchanged(self):
        """Bones NOT in any group must be bit-identical to the input."""
        quats = self._make_quats(20, 8)
        groups = {"left": [0, 1, 2]}  # bones 3-7 not in any group
        out = pose_remixing.remix_poses(quats, groups, random_seed=7)
        np.testing.assert_array_equal(quats[:, 3:, :], out[:, 3:, :])

    def test_group_bones_permuted(self):
        """Group bones in the output must be a permutation of source frames."""
        n_frames = 50
        quats = self._make_quats(n_frames, 4)
        groups = {"g": [0]}
        out = pose_remixing.remix_poses(quats, groups, random_seed=1)
        # Sort source and output slices for bone 0; they should be equal as sets
        src_sorted = np.sort(quats[:, 0, :].reshape(-1))
        out_sorted = np.sort(out[:, 0, :].reshape(-1))
        np.testing.assert_allclose(src_sorted, out_sorted, atol=1e-6)

    def test_reproducibility(self):
        quats = self._make_quats(40, 6)
        groups = {"g1": [0, 1], "g2": [2, 3]}
        out1 = pose_remixing.remix_poses(quats, groups, random_seed=99)
        out2 = pose_remixing.remix_poses(quats, groups, random_seed=99)
        np.testing.assert_array_equal(out1, out2)

    def test_different_seeds_differ(self):
        quats = self._make_quats(40, 6)
        groups = {"g": [0, 1, 2, 3]}
        out1 = pose_remixing.remix_poses(quats, groups, random_seed=1)
        out2 = pose_remixing.remix_poses(quats, groups, random_seed=2)
        assert not np.array_equal(out1, out2)

    def test_empty_groups(self):
        quats = self._make_quats(20, 5)
        out = pose_remixing.remix_poses(quats, {}, random_seed=0)
        np.testing.assert_array_equal(quats, out)
