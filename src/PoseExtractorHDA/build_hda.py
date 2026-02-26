"""
build_hda.py
Assembles PoseExtractor.hda from the source Python modules in python/.
"""
from __future__ import annotations

import os
import pathlib
import textwrap
import re

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
THIS_DIR = pathlib.Path(__file__).parent.resolve()
PYTHON_DIR = THIS_DIR / "python"
OUTPUT_HDA = THIS_DIR / "PoseExtractor.hda"

# Source module files in dependency order
MODULE_FILES = [
    "bone_utils.py",
    "fps_sampler.py",
    "pose_remixing.py",
    "lbs_intersection.py",
    "skeleton_geo.py",
    "pose_extractor_main.py",
]


# ---------------------------------------------------------------------------
# HDA parameter template
# ---------------------------------------------------------------------------

def _get_parameter_template_group():
    """Build the complete hou.ParmTemplateGroup for PoseExtractor."""
    import hou

    templates = []

    # ---------------------------------------------------------------- Tab: Sampling
    sampling_folder = hou.FolderParmTemplate("sampling", "Sampling",
                                              folder_type=hou.folderType.Tabs)
    sampling_folder.addParmTemplate(
        hou.StringParmTemplate("char_root", "Character Root (OBJ Fallback)", 1,
                                default_value=("",),
                                string_type=hou.stringParmType.NodeReference))
    sampling_folder.addParmTemplate(
        hou.IntParmTemplate("rest_frame", "Rest Frame", 1, default_value=(0,)))
    sampling_folder.addParmTemplate(
        hou.IntParmTemplate("frame_start", "Frame Start", 1,
                             default_value=(1,),
                             naming_scheme=hou.parmNamingScheme.Base1,
                             help="Use $FSTART to match scene range."))
    sampling_folder.addParmTemplate(
        hou.IntParmTemplate("frame_end", "Frame End", 1,
                             default_value=(240,),
                             help="Use $FEND to match scene range."))
    sampling_folder.addParmTemplate(
        hou.IntParmTemplate("frame_step", "Frame Step", 1, default_value=(1,),
                             min_is_strict=True, min=1))
    sampling_folder.addParmTemplate(
        hou.IntParmTemplate("num_output_poses", "Num Output Poses", 1,
                             default_value=(2000,), min=1))
    # Additional animation sources (multiparm, like Object Merge)
    extra_anims_multi = hou.FolderParmTemplate(
        "extra_anim_sources", "Extra Animation Sources",
        folder_type=hou.folderType.MultiparmBlock)
    extra_anims_multi.addParmTemplate(
        hou.StringParmTemplate("extra_anim_path#", "Animation SOP #", 1,
                                default_value=("",),
                                string_type=hou.stringParmType.NodeReference,
                                help="Path to an additional animation skeleton SOP."))
    sampling_folder.addParmTemplate(extra_anims_multi)
    templates.append(sampling_folder)

    # ----------------------------------------------------------- Tab: Diversity Bones
    div_folder = hou.FolderParmTemplate("diversity", "Diversity Bones",
                                         folder_type=hou.folderType.Tabs)
    div_folder.addParmTemplate(
        hou.StringParmTemplate("diversity_bone_filter", "Bone Filter", 1,
                                default_value=("*",),
                                help="fnmatch pattern; matched bones drive FPS distance."))
    templates.append(div_folder)

    # ----------------------------------------------------------- Tab: Pose Remixing
    remix_folder = hou.FolderParmTemplate("remixing", "Pose Remixing",
                                           folder_type=hou.folderType.Tabs)
    remix_folder.addParmTemplate(
        hou.ToggleParmTemplate("enable_remixing", "Enable Remixing", default_value=False))
    remix_folder.addParmTemplate(
        hou.IntParmTemplate("random_seed", "Random Seed", 1, default_value=(777,)))

    bone_groups_multi = hou.FolderParmTemplate(
        "bone_groups", "Bone Groups",
        folder_type=hou.folderType.MultiparmBlock)
    bone_groups_multi.addParmTemplate(
        hou.StringParmTemplate("bone_group_name#", "Group Name #", 1,
                                default_value=("",)))
    bone_groups_multi.addParmTemplate(
        hou.StringParmTemplate("bone_group_pattern#", "Bone Pattern #", 1,
                                default_value=("",)))
    remix_folder.addParmTemplate(bone_groups_multi)
    templates.append(remix_folder)

    # ---------------------------------------------------------- Tab: Self-Intersection
    isect_folder = hou.FolderParmTemplate("intersection", "Self-Intersection",
                                           folder_type=hou.folderType.Tabs)
    isect_folder.addParmTemplate(
        hou.ToggleParmTemplate("check_intersection", "Check Intersection",
                                default_value=False))
    isect_folder.addParmTemplate(
        hou.FloatParmTemplate("intersection_threshold", "Threshold", 1,
                               default_value=(0.01,), min=0.0))
    isect_folder.addParmTemplate(
        hou.MenuParmTemplate("intersection_action", "Action",
                              menu_items=("0", "1"),
                              menu_labels=("Exclude Poses", "Flag Only"),
                              default_value=0))

    body_parts_multi = hou.FolderParmTemplate(
        "body_parts", "Body Parts",
        folder_type=hou.folderType.MultiparmBlock)
    body_parts_multi.addParmTemplate(
        hou.StringParmTemplate("body_part_name#", "Part Name #", 1,
                                default_value=("",)))
    body_parts_multi.addParmTemplate(
        hou.StringParmTemplate("body_part_bone_pattern#", "Bone Pattern #", 1,
                                default_value=("",)))
    isect_folder.addParmTemplate(body_parts_multi)
    isect_folder.addParmTemplate(
        hou.StringParmTemplate("intersection_pairs", "Intersection Pairs", 1,
                                default_value=("",),
                                help="Comma-separated pairs: left_arm:torso,right_arm:torso"))
    templates.append(isect_folder)

    # ----------------------------------------------------------------- Tab: Status
    status_folder = hou.FolderParmTemplate("status", "Status",
                                            folder_type=hou.folderType.Tabs)
    # Button
    btn = hou.ButtonParmTemplate(
        "process_button", "Re-Extract Poses",
        script_callback="hou.phm().extract_poses(kwargs)",
        script_callback_language=hou.scriptLanguage.Python)
    status_folder.addParmTemplate(btn)
    status_folder.addParmTemplate(
        hou.StringParmTemplate("status_message", "Status", 1,
                                default_value=("Ready.",),
                                is_label_hidden=False))
    status_folder.addParmTemplate(
        hou.IntParmTemplate("num_selected_poses", "Selected Poses", 1,
                             default_value=(0,)))
    templates.append(status_folder)

    ptg = hou.ParmTemplateGroup(templates)
    return ptg


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build(output_path: str | None = None) -> None:
    """Assemble and save PoseExtractor.hda."""
    import hou

    out = pathlib.Path(output_path) if output_path else OUTPUT_HDA
    out.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ python module
    python_sections: list[str] = [
        "from __future__ import annotations",
        "import sys",
        "import numpy as np",
        "",
        "class _NamespaceProxy:",
        "    def __init__(self, g):",
        "        self._g = g",
        "    def __getattr__(self, name):",
        "        if name in self._g:",
        "            return self._g[name]",
        "        raise AttributeError(f\"module '{__name__}' has no attribute '{name}'\")",
        "",
        "this = _NamespaceProxy(globals())",
    ]
    
    siblings = [f"{f.replace('.py', '')}" for f in MODULE_FILES]
    for s in siblings:
        python_sections.append(f"{s} = this")
    
    for fname in MODULE_FILES:
        src_path = PYTHON_DIR / fname
        if not src_path.exists():
            raise FileNotFoundError(f"Source file not found: {src_path}")
        
        lines = src_path.read_text(encoding="utf-8").splitlines()
        new_lines = []
        for line in lines:
            if line.startswith("from __future__"):
                continue
            is_sibling = False
            for s in siblings:
                if re.match(fr"^\s*(import\s+{s}|from\s+{s}\s+import)", line):
                    is_sibling = True
                    break
            if is_sibling:
                new_lines.append(f"# (inlined) {line}")
            else:
                new_lines.append(line)

        src = "\n".join(new_lines)
        python_sections.append(f"# ===== {fname} =====\n{src}")

    python_module_code = "\n\n".join(python_sections)

    # Cook script
    # Inside a Python SOP, 'kwargs' is not a global. 
    # We must build it and ensure 'node' points to the HDA.
    cook_script = textwrap.dedent("""\
        internal_sop = hou.pwd()
        hda_node = internal_sop.parent()
        
        # Package the objects the cook function expects
        # We pass the HDA node so the script reads HDA parameters
        pass_kwargs = {
            'node': hda_node,
            'geo': internal_sop.geometry()
        }
        
        hda_node.hm().cook(pass_kwargs)
    """)

    # ------------------------------------------------------------------ create HDA
    obj_net = hou.node("/obj")
    if obj_net is None:
        obj_net = hou.node("/").createNode("objnet", "tmp_build")

    geo = obj_net.createNode("geo", "pose_extractor_tmp")
    subnet = geo.createNode("subnet", "PoseExtractor")
    
    # Wire only Input 0 (rest mesh) to the internal Python SOP.
    # Animation skeleton inputs (1..N) are accessed via the parent HDA node
    # at cook time, so they don't need direct wiring here.
    python_sop = subnet.createNode("python", "pose_extractor_core")
    python_sop.setInput(0, subnet.indirectInputs()[0])
    
    # Add Rig Visualizer for professional joint display
    try:
        viz = subnet.createNode("rigvisualizer", "joint_viz")
        viz.setInput(0, python_sop)
        # Enable joint and bone display by default
        viz.parm("showjoints").set(True)
        viz.parm("showbones").set(True)
        viz.parm("jointscale").set(0.2)
        last_node = viz
    except Exception:
        # Fallback if rigvisualizer is not found
        last_node = python_sop

    output_node = subnet.createNode("output", "output0")
    output_node.setInput(0, last_node)
    
    python_sop.parm("python").set(cook_script)

    subnet.setSelected(True, clear_all_selected=True)

    operator_name = "PoseExtractor"
    operator_label = "Pose Extractor"

    hda_node = subnet.createDigitalAsset(
        name=operator_name,
        hda_file_name=str(out),
        description=operator_label,
        min_num_inputs=2,
        max_num_inputs=2,
    )

    defn = hda_node.type().definition()
    ptg = _get_parameter_template_group()
    defn.setParmTemplateGroup(ptg)
    defn.addSection("PythonModule", python_module_code)

    # Input 0 = rest mesh, Input 1 = animation skeleton(s)
    defn.addSection("InputNames",
                     "Rest Mesh / BoneDeform\t"
                     "Animation Skeleton")

    defn.save(str(out))

    print(f"[build_hda] Saved: {out}")

    hda_node.destroy()
    geo.destroy()


if __name__ == "__main__":
    build()
