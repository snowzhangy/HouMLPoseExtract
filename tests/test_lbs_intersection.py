"""
test_lbs_intersection.py
Unit tests for LBS deformation and intersection detection.
No Houdini dependency — uses mock_hou for any hou-typed stubs.
"""
from __future__ import annotations

import sys
import os
import pytest
import numpy as np

# Paths
TESTS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(TESTS_DIR, "..", "src", "PoseExtractorHDA", "python")
FIXTURES_DIR = os.path.join(TESTS_DIR, "fixtures")

sys.path.insert(0, SRC_DIR)
sys.path.insert(0, FIXTURES_DIR)

# Install mock hou before importing lbs_intersection
import mock_hou as _mock_hou
_mock_hou.install()

import lbs_intersection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _identity_batch(n: int) -> np.ndarray:
    """Return n identity 4×4 matrices stacked as [n, 4, 4]."""
    return np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))


def _make_unit_cube_verts(n: int = 8) -> np.ndarray:
    """Return the 8 corners of a unit cube centred at origin."""
    corners = np.array([
        [-0.5, -0.5, -0.5],
        [+0.5, -0.5, -0.5],
        [-0.5, +0.5, -0.5],
        [+0.5, +0.5, -0.5],
        [-0.5, -0.5, +0.5],
        [+0.5, -0.5, +0.5],
        [-0.5, +0.5, +0.5],
        [+0.5, +0.5, +0.5],
    ], dtype=np.float32)
    return corners


# ---------------------------------------------------------------------------
# compute_lbs tests
# ---------------------------------------------------------------------------

class TestComputeLBS:

    def test_identity_transforms_no_deformation(self):
        """With identity transforms and rest_inv = I, LBS output == input."""
        V, B = 10, 3
        rest_positions = np.random.default_rng(0).random((V, 3)).astype(np.float32)
        weights = np.zeros((V, B), dtype=np.float32)
        weights[:, 0] = 1.0  # all vertices 100% on bone 0

        rest_inv = _identity_batch(B)
        world_xforms = _identity_batch(B)

        result = lbs_intersection.compute_lbs(rest_positions, weights, rest_inv, world_xforms)
        np.testing.assert_allclose(result, rest_positions, atol=1e-5)

    def test_translation_single_bone(self):
        """Translate bone 0 by (1, 2, 3) — all vertices should move by same amount."""
        V, B = 5, 1
        rest_positions = np.zeros((V, 3), dtype=np.float32)
        weights = np.ones((V, B), dtype=np.float32)

        rest_inv = _identity_batch(B)

        world_xforms = _identity_batch(B)
        tx, ty, tz = 1.0, 2.0, 3.0
        world_xforms[0, 3, 0] = tx
        world_xforms[0, 3, 1] = ty
        world_xforms[0, 3, 2] = tz

        result = lbs_intersection.compute_lbs(rest_positions, weights, rest_inv, world_xforms)
        expected = np.array([[tx, ty, tz]] * V, dtype=np.float32)
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_output_shape(self):
        V, B = 20, 4
        rest_positions = np.random.default_rng(1).random((V, 3)).astype(np.float32)
        weights = np.ones((V, B), dtype=np.float32) / B
        rest_inv = _identity_batch(B)
        world_xforms = _identity_batch(B)
        result = lbs_intersection.compute_lbs(rest_positions, weights, rest_inv, world_xforms)
        assert result.shape == (V, 3)

    def test_two_bone_blend(self):
        """A vertex half-weighted between two translated bones lands in the middle."""
        V, B = 1, 2
        rest_positions = np.zeros((V, 3), dtype=np.float32)
        weights = np.array([[0.5, 0.5]], dtype=np.float32)

        rest_inv = _identity_batch(B)
        world_xforms = _identity_batch(B)
        world_xforms[0, 3, 0] = 2.0   # bone 0 at x=2
        world_xforms[1, 3, 0] = 0.0   # bone 1 at x=0

        result = lbs_intersection.compute_lbs(rest_positions, weights, rest_inv, world_xforms)
        # Expected: 0.5 * [2,0,0] + 0.5 * [0,0,0] = [1,0,0]
        np.testing.assert_allclose(result[0], [1.0, 0.0, 0.0], atol=1e-5)


# ---------------------------------------------------------------------------
# check_intersection_pair tests
# ---------------------------------------------------------------------------

class TestCheckIntersectionPair:

    def test_non_intersecting_cubes(self):
        """Two cubes 10 units apart should NOT intersect."""
        a = _make_unit_cube_verts()
        b = _make_unit_cube_verts() + np.array([10.0, 0.0, 0.0])
        assert not lbs_intersection.check_intersection_pair(a, b, threshold=0.01)

    def test_fully_overlapping(self):
        """Two identical vertex sets are fully intersecting."""
        a = _make_unit_cube_verts()
        b = a.copy()
        assert lbs_intersection.check_intersection_pair(a, b, threshold=0.01)

    def test_threshold_sensitivity(self):
        """Cubes exactly 0.5 units apart: detected at threshold=0.6, not at 0.4.

        Cube A spans x in [-0.5, 0.5], cube B shifted by 1.5 spans x in [1.0, 2.0].
        Closest vertex pair distance = 1.0 - 0.5 = 0.5.
        """
        a = _make_unit_cube_verts()
        b = _make_unit_cube_verts() + np.array([1.5, 0.0, 0.0])  # nearest face gap = 0.5
        assert lbs_intersection.check_intersection_pair(a, b, threshold=0.6)
        assert not lbs_intersection.check_intersection_pair(a, b, threshold=0.4)

    def test_empty_verts_a(self):
        b = _make_unit_cube_verts()
        assert not lbs_intersection.check_intersection_pair(np.zeros((0, 3)), b, 0.01)

    def test_empty_verts_b(self):
        a = _make_unit_cube_verts()
        assert not lbs_intersection.check_intersection_pair(a, np.zeros((0, 3)), 0.01)


# ---------------------------------------------------------------------------
# build_body_part_vertex_map tests
# ---------------------------------------------------------------------------

class TestBuildBodyPartVertexMap:

    def test_basic_assignment(self):
        """Vertices dominated by bone 0 are assigned to 'part_a'."""
        V, B = 6, 3
        # All weight on bone 0
        weights = np.zeros((V, B), dtype=np.float32)
        weights[:3, 0] = 1.0   # vertices 0-2 → bone 0
        weights[3:, 1] = 1.0   # vertices 3-5 → bone 1

        bone_name_to_idx = {"bone0": 0, "bone1": 1, "bone2": 2}
        body_part_defs = {
            "part_a": ["bone0"],
            "part_b": ["bone1"],
        }
        result = lbs_intersection.build_body_part_vertex_map(weights, bone_name_to_idx, body_part_defs)

        assert set(result["part_a"].tolist()) == {0, 1, 2}
        assert set(result["part_b"].tolist()) == {3, 4, 5}

    def test_missing_bone_graceful(self):
        """Unknown bone names are silently ignored."""
        weights = np.ones((4, 2), dtype=np.float32) * 0.5
        bone_name_to_idx = {"b0": 0, "b1": 1}
        body_part_defs = {"part": ["b0", "nonexistent_bone"]}
        result = lbs_intersection.build_body_part_vertex_map(weights, bone_name_to_idx, body_part_defs)
        assert len(result["part"]) > 0

    def test_low_weight_excluded(self):
        """Vertices with weights below threshold are excluded from a part."""
        V, B = 4, 2
        weights = np.zeros((V, B), dtype=np.float32)
        weights[0, 0] = 0.005  # below default threshold of 0.01
        weights[1, 0] = 0.02   # above threshold
        bone_name_to_idx = {"b0": 0, "b1": 1}
        body_part_defs = {"part": ["b0"]}
        result = lbs_intersection.build_body_part_vertex_map(
            weights, bone_name_to_idx, body_part_defs, weight_threshold=0.01
        )
        assert 0 not in result["part"].tolist()
        assert 1 in result["part"].tolist()


# ---------------------------------------------------------------------------
# resolve_body_part_defs tests
# ---------------------------------------------------------------------------

class TestResolveBodyPartDefs:

    def test_pattern_matching(self):
        bone_names = ["left_arm", "left_elbow", "right_arm", "spine"]
        raw_defs = [
            {"name": "left", "bone_pattern": "left_*"},
            {"name": "right", "bone_pattern": "right_*"},
        ]
        result = lbs_intersection.resolve_body_part_defs(bone_names, raw_defs)
        assert result["left"] == ["left_arm", "left_elbow"]
        assert result["right"] == ["right_arm"]

    def test_empty_pattern_skipped(self):
        bone_names = ["a", "b"]
        raw_defs = [{"name": "p", "bone_pattern": ""}]
        result = lbs_intersection.resolve_body_part_defs(bone_names, raw_defs)
        assert len(result) == 0
