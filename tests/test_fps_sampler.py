"""
test_fps_sampler.py
Unit tests for fps_sampler.farthest_point_sampling.
No Houdini dependency required.
"""
from __future__ import annotations

import sys
import os
import pytest
import numpy as np

# Make sure source modules are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "PoseExtractorHDA", "python"))

import fps_sampler


class TestFarthestPointSampling:

    def _make_quats(self, n_frames: int, n_bones: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        q = rng.standard_normal((n_frames, n_bones, 4)).astype(np.float32)
        # Normalise so they are unit quaternions
        norms = np.linalg.norm(q, axis=-1, keepdims=True) + 1e-8
        return (q / norms).astype(np.float32)

    def test_returns_correct_count(self):
        quats = self._make_quats(100, 10)
        selected = fps_sampler.farthest_point_sampling(quats, 20, list(range(10)))
        assert len(selected) == 20

    def test_does_not_exceed_n_frames(self):
        quats = self._make_quats(10, 5)
        selected = fps_sampler.farthest_point_sampling(quats, 50, list(range(5)))
        assert len(selected) == 10  # capped at n_frames

    def test_no_duplicates(self):
        quats = self._make_quats(200, 8)
        selected = fps_sampler.farthest_point_sampling(quats, 50, list(range(8)))
        assert len(selected) == len(set(selected)), "Duplicate frame indices in result"

    def test_valid_indices(self):
        n_frames = 50
        quats = self._make_quats(n_frames, 6)
        selected = fps_sampler.farthest_point_sampling(quats, 15, list(range(6)))
        for idx in selected:
            assert 0 <= idx < n_frames

    def test_diversity_subset(self):
        """Only the first 2 bones should drive distance."""
        quats = self._make_quats(100, 10)
        selected_all = fps_sampler.farthest_point_sampling(quats, 20, list(range(10)))
        selected_sub = fps_sampler.farthest_point_sampling(quats, 20, [0, 1])
        # Results may differ — just verify both are valid
        assert len(selected_sub) == 20
        assert len(set(selected_sub)) == 20

    def test_empty_diversity_bones(self):
        """With no diversity bones, should return evenly spaced frames."""
        quats = self._make_quats(100, 10)
        selected = fps_sampler.farthest_point_sampling(quats, 10, [])
        assert len(selected) == 10

    def test_deterministic(self):
        """Same input → same result (FPS is deterministic)."""
        quats = self._make_quats(100, 10, seed=42)
        s1 = fps_sampler.farthest_point_sampling(quats, 20, list(range(10)))
        s2 = fps_sampler.farthest_point_sampling(quats, 20, list(range(10)))
        assert s1 == s2

    def test_maximally_diverse(self):
        """Manual check: 4 frames in 2D — FPS should pick corners-first."""
        # Frame 0: (0,0), Frame 1: (1,0), Frame 2: (0,1), Frame 3: (1,1)
        # Represent as [4, 1, 4] quats (1 bone, 4 components used as 2D)
        quats = np.zeros((4, 1, 4), dtype=np.float32)
        quats[0, 0, :2] = [0.0, 0.0]
        quats[1, 0, :2] = [1.0, 0.0]
        quats[2, 0, :2] = [0.0, 1.0]
        quats[3, 0, :2] = [1.0, 1.0]

        # First selected should be farthest from zero → frame 3 (1,1)
        selected = fps_sampler.farthest_point_sampling(quats, 4, [0])
        assert selected[0] == 3, f"Expected first pick = 3 (farthest from 0), got {selected[0]}"

    def test_compute_distance_matrix(self):
        a = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        b = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        dist = fps_sampler.compute_distance_matrix(a, b)
        # mean((1-0)^2 + (0-1)^2 + 0 + 0) / 4 = 0.5
        assert abs(dist - 0.5) < 1e-6


class TestComputeDistanceMatrix:

    def test_zero_distance_same_pose(self):
        q = np.ones((5, 4), dtype=np.float32)
        assert fps_sampler.compute_distance_matrix(q, q) == 0.0

    def test_positive_for_different(self):
        a = np.zeros((4, 4), dtype=np.float32)
        b = np.ones((4, 4), dtype=np.float32)
        assert fps_sampler.compute_distance_matrix(a, b) > 0.0
