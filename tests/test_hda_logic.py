"""
test_hou_logic.py
Verify that the logic using hou.geometryAtTime (replacing ContextManager) works.
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

# Install mock hou
import mock_hou
mock_hou.install()

import bone_utils

class MockParm:
    def __init__(self, val):
        self._val = val
    def eval(self):
        return self._val

class _MockConnection:
    """Stub for hou.NodeConnection returned by inputConnections()."""
    def __init__(self, input_idx, upstream_node, output_idx=0):
        self._input_idx = input_idx
        self._upstream = upstream_node
        self._output_idx = output_idx
    def inputIndex(self):
        return self._input_idx
    def inputNode(self):
        return self._upstream
    def outputIndex(self):
        return self._output_idx

class MockNode:
    def __init__(self):
        self._geo = mock_hou.Geometry()
        # Add a point so it has some geometry
        self._geo.createPoint()

    def input(self, index):
        return self

    def inputConnections(self):
        return [_MockConnection(0, self)]

    def inputGeometry(self, index):
        return self._geo

    def geometry(self):
        return self._geo

    def geometryAtFrame(self, f, output_index=0):
        return self._geo

def test_sample_world_transforms_from_geo():
    accessor = MockNode()
    frames = [1, 2, 3]
    # accessor_node + input_index pattern; MockNode.inputGeometry returns identity geo
    result = bone_utils.sample_world_transforms_from_geo(accessor, 0, frames)

    assert result.shape == (3, 1, 4, 4)
    assert np.allclose(result[0], np.eye(4))

def test_matrix_to_quat_identity():
    mats = np.eye(3, dtype=np.float32).reshape(1, 3, 3)
    q = bone_utils.matrix_to_quat(mats)
    # Identity matrix should be qw=1, xyz=0. xyzw order -> [0,0,0,1]
    expected = np.array([[0, 0, 0, 1]], dtype=np.float32)
    np.testing.assert_allclose(q, expected, atol=1e-6)

def test_matrix_to_quat_90_x():
    # 90 degrees around X axis
    # [1, 0, 0]
    # [0, 0, -1]
    # [0, 1, 0]
    mats = np.array([[
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ]], dtype=np.float32)
    q = bone_utils.matrix_to_quat(mats)
    # Half angle = 45 deg. sin(45) = 0.707, cos(45) = 0.707
    # q = [sin(th/2)*ax, sin(th/2)*ay, sin(th/2)*az, cos(th/2)]
    # For -90 deg around X: [sin(-45), 0, 0, cos(-45)] = [-0.707, 0, 0, 0.707]
    val = np.sqrt(2)/2
    expected = np.array([[-val, 0, 0, val]], dtype=np.float32)
    np.testing.assert_allclose(q, expected, atol=1e-6)
