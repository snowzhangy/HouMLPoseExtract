"""
mock_hou.py
Minimal stub of the hou module for offline unit testing.
Only the symbols actually used by the pose extractor modules are stubbed.
"""
from __future__ import annotations

import math
import sys
from typing import Any


# ---------------------------------------------------------------------------
# hou.Vector3
# ---------------------------------------------------------------------------

class Vector3:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z

    def __iter__(self):
        return iter((self.x, self.y, self.z))

    def __repr__(self):
        return f"Vector3({self.x}, {self.y}, {self.z})"


# ---------------------------------------------------------------------------
# hou.Matrix3 / hou.Matrix4
# ---------------------------------------------------------------------------

class Matrix3:
    def __init__(self, *args):
        if len(args) == 1 and hasattr(args[0], '__len__'):
            # hou.Matrix3((a,b,c,d,e,f,g,h,i)) — single tuple/list of 9
            self._data = list(args[0])
        elif len(args) == 9:
            self._data = list(args)
        else:
            self._data = [1, 0, 0, 0, 1, 0, 0, 0, 1]

    def asTuple(self):
        return tuple(self._data)


class Matrix4:
    """Identity 4×4 by default; can be initialised from a flat 16-element tuple."""

    def __init__(self, *args):
        if len(args) == 1 and hasattr(args[0], '__len__'):
            # hou.Matrix4((a,b,...,p)) — single tuple/list of 16
            self._data = list(args[0])
        elif len(args) == 16:
            self._data = list(args)
        else:
            self._data = [
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1,
            ]

    def asTuple(self):
        return tuple(self._data)

    def setTranslation(self, pos: Vector3) -> None:
        self._data[12] = pos.x
        self._data[13] = pos.y
        self._data[14] = pos.z

    @classmethod
    def identity(cls) -> "Matrix4":
        return cls()


# ---------------------------------------------------------------------------
# hou.attribType
# ---------------------------------------------------------------------------

class attribType:
    Point = "point"
    Prim = "prim"
    Vertex = "vertex"
    Global = "global"


# ---------------------------------------------------------------------------
# hou.attribTypeInfo
# ---------------------------------------------------------------------------

class attribTypeInfo:
    Matrix3 = "matrix3"
    Matrix4 = "matrix4"
    None_ = "none"


# ---------------------------------------------------------------------------
# hou.Geometry helpers
# ---------------------------------------------------------------------------

class _Attrib:
    def __init__(self, name: str, default):
        self.name = name
        self.default = default
        self._options: dict[str, Any] = {}

    def setOption(self, name: str, value: Any) -> None:
        self._options[name] = value

    def option(self, name: str) -> Any:
        return self._options.get(name)


class _Point:
    def __init__(self, position: Vector3, number: int = 0):
        self._position = position
        self._attribs: dict[str, Any] = {}
        self._number = number

    def position(self) -> Vector3:
        return self._position

    def setPosition(self, pos: Vector3) -> None:
        self._position = pos

    def number(self) -> int:
        return self._number

    def attribValue(self, name: str) -> Any:
        return self._attribs.get(name)

    def setAttribValue(self, name: str, value: Any) -> None:
        self._attribs[name] = value


class _Vertex:
    def __init__(self, pt: _Point):
        self._point = pt

    def point(self) -> _Point:
        return self._point


class _Polygon:
    def __init__(self):
        self._verts: list[_Vertex] = []
        self._attribs: dict[str, Any] = {}

    def addVertex(self, point: _Point) -> None:
        self._verts.append(_Vertex(point))

    def vertices(self) -> list[_Vertex]:
        return list(self._verts)

    def setAttribValue(self, name: str, value: Any) -> None:
        self._attribs[name] = value


class Geometry:
    """Minimal hou.Geometry stub."""

    def __init__(self):
        self._points: list[_Point] = []
        self._prims: list[_Polygon] = []
        self._attribs: dict[str, _Attrib] = {}
        self._global_attribs: dict[str, Any] = {}

    def clear(self) -> None:
        self._points.clear()
        self._prims.clear()
        self._attribs.clear()
        self._global_attribs.clear()

    def points(self) -> list[_Point]:
        return list(self._points)

    def createPoint(self) -> _Point:
        pt = _Point(Vector3(), number=len(self._points))
        self._points.append(pt)
        return pt

    def prims(self) -> list[_Polygon]:
        return list(self._prims)

    def createPolygon(self, is_closed: bool = True) -> _Polygon:
        poly = _Polygon()
        self._prims.append(poly)
        return poly

    def addAttrib(self, attrib_type, name: str, default) -> _Attrib:
        attr = _Attrib(name, default)
        self._attribs[name] = attr
        return attr

    def findPointAttrib(self, name: str) -> _Attrib | None:
        return self._attribs.get(name)

    def findGlobalAttrib(self, name: str) -> _Attrib | None:
        return self._attribs.get(f"__global__{name}")

    def setGlobalAttribValue(self, name: str, value: Any) -> None:
        self._global_attribs[name] = value

    def getGlobalAttribValue(self, name: str) -> Any:
        return self._global_attribs.get(name)


# ---------------------------------------------------------------------------
# hou.ObjNode stub
# ---------------------------------------------------------------------------

class ObjNode:
    """Minimal stub for Houdini OBJ nodes."""

    def __init__(self, name: str, type_name: str = "bone",
                 world_xform: "Matrix4 | None" = None,
                 parent_node: "ObjNode | None" = None,
                 children: "list[ObjNode] | None" = None):
        self._name = name
        self._type_name = type_name
        self._world_xform = world_xform or Matrix4()
        self._parent_node = parent_node
        self._children: list[ObjNode] = children or []

    def name(self) -> str:
        return self._name

    def path(self) -> str:
        if self._parent_node is not None:
            return self._parent_node.path() + "/" + self._name
        return "/" + self._name

    def type(self) -> "ObjNode":
        return self

    def children(self) -> list["ObjNode"]:
        return self._children

    def parent(self) -> "ObjNode | None":
        return self._parent_node

    def input(self, index: int) -> "ObjNode | None":
        """Mock implementation: return self or None."""
        return self

    def inputGeometry(self, index: int) -> Geometry:
        return self.geometry()

    def geometry(self) -> Geometry:
        return Geometry()

    def geometryAtFrame(self, f: float) -> Geometry:
        return self.geometry()

    def worldTransformAtTime(self, t: float) -> Matrix4:
        return self._world_xform

    # Used by type().name()
    _type_name_val: str = ""

    def __getattr__(self, item):
        if item == "_type_name_val":
            return "bone"
        raise AttributeError(item)


# Make type().name() work
class _TypeProxy:
    def __init__(self, name: str):
        self._name = name

    def name(self) -> str:
        return self._name


# Patch ObjNode.type() to return _TypeProxy
_original_ObjNode_type = ObjNode.type


def _patched_type(self):
    return _TypeProxy(self._type_name)


ObjNode.type = _patched_type  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# hou module-level functions
# ---------------------------------------------------------------------------

def node(path: str) -> ObjNode | None:
    """Return None unless overridden in a test."""
    return _node_registry.get(path)


_node_registry: dict[str, ObjNode] = {}


def register_node(path: str, obj_node: ObjNode) -> None:
    """Test helper: register a node at a path so hou.node(path) returns it."""
    _node_registry[path] = obj_node


def frame() -> float:
    return _current_frame


_current_frame: float = 1.0


def set_frame(f: float) -> None:
    global _current_frame
    _current_frame = f


def setFrame(f: float) -> None:
    """Alias matching real Houdini's hou.setFrame()."""
    set_frame(f)


def frameToTime(f: float, fps: float = 24.0) -> float:
    return (f - 1.0) / fps


def timeToFrame(t: float, fps: float = 24.0) -> float:
    return t * fps + 1.0


# ---------------------------------------------------------------------------
# Install as 'hou' in sys.modules so imports resolve
# ---------------------------------------------------------------------------

def install() -> None:
    """Install this stub as sys.modules['hou']."""
    sys.modules["hou"] = sys.modules[__name__]  # type: ignore[assignment]
