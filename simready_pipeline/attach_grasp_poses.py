import os
import sys
import pickle
import re
import json
import traceback

_SCRIPT_BASE = os.path.dirname(os.path.abspath(__file__))
REVIEWER_BASE = os.environ.get(
    "REVIEWER_BASE",
    os.path.join(_SCRIPT_BASE, "reviewer_subset")
    if os.path.isdir(os.path.join(_SCRIPT_BASE, "reviewer_subset"))
    else _SCRIPT_BASE,
)
SIM_READY_DIRNAME = os.environ.get("REVIEWER_SIM_READY_DIRNAME", "sim_ready")
REPORT_DIR = os.path.abspath(os.environ.get("PIPELINE_REPORT_DIR", os.path.join(REVIEWER_BASE, ".pipeline_reports")))

def _resolve_dir(env_name, relative_candidates):
    if os.environ.get(env_name):
        return os.path.abspath(os.environ[env_name])
    for rel in relative_candidates:
        candidate = os.path.abspath(os.path.join(REVIEWER_BASE, *rel))
        if os.path.isdir(candidate):
            return candidate
    return os.path.abspath(os.path.join(REVIEWER_BASE, *relative_candidates[0]))


CODE_BASE = _resolve_dir("REVIEWER_CODE_BASE", [
    ("template_rich_knowledge", "code"),
    ("..", "template_rich_knowledge", "code"),
    ("..", "..", "template_rich_knowledge", "code"),
    ("..", "..", "..", "template_rich_knowledge", "code"),
])
CLASSES_BASE = _resolve_dir("REVIEWER_CLASSES_DIR", [
    ("classes",),
    ("..", "classes"),
    ("..", "..", "classes"),
    ("..", "..", "..", "classes"),
])
PIP_PACKAGES = _resolve_dir("REVIEWER_PIP_PACKAGES", [
    ("pip_packages",),
    ("..", "pip_packages"),
    ("..", "..", "pip_packages"),
    ("..", "..", "..", "pip_packages"),
])
if PIP_PACKAGES not in sys.path:
    sys.path.insert(0, PIP_PACKAGES)

import numpy as np
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

from pxr import Gf, Sdf, Usd, UsdGeom


def _install_lightweight_utils_module():
    import types

    def get_rodrigues_matrix(axis, angle):
        axis = np.asarray(axis, dtype=np.float64)
        norm = np.linalg.norm(axis)
        if norm == 0:
            return np.eye(3)
        axis = axis / norm
        x, y, z = axis
        c = np.cos(angle)
        s = np.sin(angle)
        C = 1.0 - c
        return np.array([
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
        ], dtype=np.float64)

    def apply_transformation(vertices, position, rotation, rotation_order="XYZ", offset_first=False):
        from scipy.spatial.transform import Rotation as Rot

        vertices = np.asarray(vertices, dtype=np.float64)
        position = np.asarray(position, dtype=np.float64)
        rotation = np.asarray(rotation, dtype=np.float64)
        rot_mat = Rot.from_euler(rotation_order, rotation, degrees=False).as_matrix()
        if offset_first:
            return (vertices + position) @ rot_mat.T
        return vertices @ rot_mat.T + position

    def adjust_position_from_rotation(position, rotation, rotation_order="XYZ"):
        return list(apply_transformation(
            np.asarray(position, dtype=np.float64)[None, :], [0, 0, 0], rotation, rotation_order)[0])

    def list_add(list1, list2):
        return [a + b for a, b in zip(list1, list2)]

    utils_module = types.ModuleType("utils")
    utils_module.apply_transformation = apply_transformation
    utils_module.adjust_position_from_rotation = adjust_position_from_rotation
    utils_module.list_add = list_add
    utils_module.get_rodrigues_matrix = get_rodrigues_matrix
    utils_module.COLOR20 = []
    sys.modules["utils"] = utils_module
    return utils_module


def _install_optional_dependency_shims():
    import types

    if "open3d" not in sys.modules:
        sys.modules["open3d"] = types.ModuleType("open3d")

BODY_TEMPLATE_NAMES = {
    "Multilevel_Body", "Cylindrical_Body", "Cuboidal_Body",
    "Cylindrical_body", "Cuboidal_body",
    "Double_Layer_Barrel", "Storagefurniture_body",
}


def _safe_prim_name(name):
    out = re.sub(r"[^A-Za-z0-9_]", "_", str(name))
    if not out or out[0].isdigit():
        out = f"part_{out}"
    return out


def _load_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, list):
        if len(data) == 0:
            raise ValueError(f"empty pkl list: {pkl_path}")
        data = data[0]
    if not isinstance(data, dict) or "conceptualization" not in data:
        raise ValueError(f"invalid conceptualization format: {pkl_path}")
    return data


def _extract_category(object_name):
    parts = object_name.split("_")
    for p in parts:
        if p in ("Bottle", "Box", "Cup", "Mug", "Shampoo", "Sanitizer",
                  "Gluestick", "Ruler", "Pen", "USB", "Chair", "KitchenPot",
                  "Bowl", "Clip", "Doorhandle", "Knife", "Lighter", "Mouse",
                  "Pliers", "Scissors", "Spoon", "Switch", "Table",
                  "Hanger", "Foldingrack", "StorageFurniture"):
            return p
    return None


CATEGORY_NORMALIZE = {"Cup": "Mug", "Mug": "Mug", "Sanitizer": "Shampoo", "Hanger": "Foldingrack"}

CATEGORY_INFO = {
    "Bottle": {"dir": "Bottle", "root_path": "/Bottle", "usd_filename": "Bottle.usd"},
    "Box": {"dir": "Box", "root_path": "/Box", "usd_filename": "Box.usd"},
    "Mug": {"dir": "Mug", "root_path": "/Mug", "usd_filename": "Mug.usd"},
    "Shampoo": {"dir": "Shampoo", "root_path": "/Shampoo", "usd_filename": "Shampoo.usd"},
    "Gluestick": {"dir": "Gluestick", "root_path": "/Gluestick", "usd_filename": "Gluestick.usd"},
    "Ruler": {"dir": "Ruler", "root_path": "/Ruler", "usd_filename": "Ruler.usd"},
    "Pen": {"dir": "Pen", "root_path": "/Pen", "usd_filename": "Pen.usd"},
    "USB": {"dir": "USB", "root_path": "/USB", "usd_filename": "USB.usd"},
    "Chair": {"dir": "Chair", "root_path": "/Chair", "usd_filename": "Chair.usd"},
    "KitchenPot": {"dir": "KitchenPot", "root_path": "/KitchenPot", "usd_filename": "KitchenPot.usd"},
    "Doorhandle": {"dir": "Doorhandle", "root_path": "/Doorhandle", "usd_filename": "Doorhandle.usd"},
    "Knife": {"dir": "Knife", "root_path": "/Knife", "usd_filename": "Knife.usd"},
    "Pliers": {"dir": "Pliers", "root_path": "/Pliers", "usd_filename": "Pliers.usd"},
    "Scissors": {"dir": "Scissors", "root_path": "/Scissors", "usd_filename": "Scissors.usd"},
    "Switch": {"dir": "Switch", "root_path": "/Switch", "usd_filename": "Switch.usd"},
    "Table": {"dir": "Table", "root_path": "/Table", "usd_filename": "Table.usd"},
    "Foldingrack": {"dir": "Foldingrack", "root_path": "/Foldingrack", "usd_filename": "Foldingrack.usd"},
    "Bowl": {"dir": "Bowl", "root_path": "/Bowl", "usd_filename": "Bowl.usd"},
    "Clip": {"dir": "Clip", "root_path": "/Clip", "usd_filename": "Clip.usd"},
    "Lighter": {"dir": "Lighter", "root_path": "/Lighter", "usd_filename": "Lighter.usd"},
    "Mouse": {"dir": "Mouse", "root_path": "/Mouse", "usd_filename": "Mouse.usd"},
    "Spoon": {"dir": "Spoon", "root_path": "/Spoon", "usd_filename": "Spoon.usd"},
    "StorageFurniture": {"dir": "StorageFurniture", "root_path": "/StorageFurniture", "usd_filename": "StorageFurniture.usd"},
}

SHAMPOO_TEMPLATE_ALIASES = {
    "Cylindrical_Body": "Cylindrical_body",
    "Cuboidal_Body": "Cuboidal_body",
    "Toothpaste_Body": "Toothpaste_body",
    "Regular_Nozzle": "Regular_nozzle",
    "Cylindrical_Cap": "Cylindrical_cap",
    "Regular_Cap": "Regular_cap",
}


def _get_or_create_parent(stage, prim_path):
    parts = str(prim_path).lstrip("/").split("/")
    current = ""
    for part in parts[:-1]:
        parent = f"/{part}" if not current else f"{current}/{part}"
        if not stage.GetPrimAtPath(parent):
            UsdGeom.Xform.Define(stage, parent)
        current = parent
    return prim_path


def _reset_grasps(stage, grasp_root):
    _get_or_create_parent(stage, grasp_root)
    grasp_parent = UsdGeom.Xform.Define(stage, grasp_root).GetPrim()
    for child in list(grasp_parent.GetChildren()):
        stage.RemovePrim(child.GetPath())
    return grasp_parent


def _grasp_count(stage, root_path):
    return sum(
        1 for prim in stage.Traverse()
        if str(prim.GetPath()).startswith(f"{root_path}/grasps/")
        and prim.HasAttribute("grasp:pose_matrix")
    )


def _import_category_modules(category_dir):
    import importlib
    import importlib.util

    for mod_name in list(sys.modules.keys()):
        if mod_name in ("concept_template", "geometry_template", "knowledge_definitions",
                         "knowledge_utils", "utils", "base_template"):
            del sys.modules[mod_name]

    code_dir = os.path.join(CODE_BASE, category_dir)
    class_dir = os.path.join(CLASSES_BASE, category_dir)
    for p in (code_dir, class_dir, PIP_PACKAGES):
        while p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, PIP_PACKAGES)
    if category_dir == "StorageFurniture":
        # The reviewer pkl uses the expanded constructor in classes/.
        sys.path.insert(0, code_dir)
        sys.path.insert(0, class_dir)
    else:
        sys.path.insert(0, class_dir)
        sys.path.insert(0, code_dir)

    _install_optional_dependency_shims()
    utils_module = _install_lightweight_utils_module()

    concept_template = importlib.import_module("concept_template")
    geometry_template = importlib.import_module("geometry_template")
    if category_dir == "StorageFurniture":
        def load_code_module(name):
            spec = importlib.util.spec_from_file_location(name, os.path.join(code_dir, f"{name}.py"))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module
        knowledge_definitions = load_code_module("knowledge_definitions")
        knowledge_utils = load_code_module("knowledge_utils")
        if not hasattr(knowledge_definitions, "Drawer_with_U_handle"):
            knowledge_definitions.Drawer_with_U_handle = type("Drawer_with_U_handle", (), {})
    else:
        knowledge_definitions = importlib.import_module("knowledge_definitions")
        knowledge_utils = importlib.import_module("knowledge_utils")

    for p in (code_dir, class_dir):
        if p in sys.path:
            sys.path.remove(p)
    return concept_template, geometry_template, knowledge_definitions, knowledge_utils, utils_module


def _build_template_lookup(concept_template, geometry_template):
    lookup = {}
    for mod in (concept_template, geometry_template):
        for name in dir(mod):
            if not name.startswith("_"):
                obj = getattr(mod, name)
                if isinstance(obj, type):
                    lookup[name] = obj
    return lookup


def _find_body_usd_offset(stage, root_path, pkl_body_template, pkl_body_pos):
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Xform":
            continue
        pname = prim.GetName()
        if any(kw in str(prim.GetPath()) for kw in ("virtual", "joints", "visuals", "collisions", "Looks", "grasps")):
            continue
        if pname in ("Bottle", "Box", "Mug", "Shampoo", "Gluestick", "Ruler", "Pen"):
            continue
        if pname.startswith("_"):
            continue

        matched = False
        if pname == pkl_body_template:
            matched = True
        elif pname.lower() == pkl_body_template.lower():
            matched = True
        elif pname.replace("_", "").lower() == pkl_body_template.replace("_", "").lower():
            matched = True

        if not matched and pkl_body_template:
            matched = pname.lower().startswith(pkl_body_template.lower() + "_arm")

        if not matched:
            continue

        t = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(prim).ExtractTranslation())
        usd_body_pos = t

        delta_x = usd_body_pos[0] - pkl_body_pos[0]
        delta_z = usd_body_pos[2] - pkl_body_pos[1]
        return delta_x, delta_z, usd_body_pos

    return 0.0, 0.0, None


def _concept_to_usd_pos(pos_xyz, delta_x, delta_z):
    return np.array([pos_xyz[0] + delta_x, -pos_xyz[2], pos_xyz[1] + delta_z], dtype=np.float64)


def _concept_to_usd_quat(q_xyzw):
    return np.array([q_xyzw[0], -q_xyzw[2], q_xyzw[1], q_xyzw[3]], dtype=np.float64)


def _concept_to_usd_vec(v_xyz):
    return np.array([v_xyz[0], -v_xyz[2], v_xyz[1]], dtype=np.float64)


def _concept_to_usd_matrix(M_concept, delta_x, delta_z):
    M_swap = np.array([
        [1, 0, 0, delta_x],
        [0, 0, -1, 0],
        [0, 1, 0, delta_z],
        [0, 0, 0, 1],
    ], dtype=np.float64)
    M = np.array(M_concept, dtype=np.float64)
    if M.shape != (4, 4):
        M_4x4 = np.eye(4)
        M_4x4[:3, :3] = M[:3, :3]
        M_4x4[:3, 3] = M[:3, 3]
        M = M_4x4
    return M_swap @ M


def _sample_params(manip_params_size):
    if manip_params_size == 0:
        return [None]
    if manip_params_size == 1:
        return [(0.0,), (-0.5,), (0.5,)]
    if manip_params_size == 2:
        return [(0.0, 0.0), (-0.5, 0.0), (0.5, 0.0), (0.0, -0.5), (0.0, 0.5)]
    if manip_params_size == 3:
        return [(0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (-0.5, 0.0, 1.0), (0.5, 0.0, 1.0), (0.0, 0.5, -1.0)]
    return [tuple([0.0] * manip_params_size)]


def _sample_component_grasp_params(component, manip_params_size):
    if getattr(component, "semantic", None) in ("Body", "Cover"):
        return [(0.0,), (0.5,), (1.0,), (1.5,)]
    return _sample_params(manip_params_size)


def _normalize_grasp_spec(obj, spec):
    """Validate a grasp spec and convert object-local poses to world poses."""
    if not isinstance(spec, dict):
        return None
    normalized = dict(spec)
    try:
        width = float(normalized["grasp_width"])
        if not np.isfinite(width) or width <= 0.0:
            return None
        normalized["grasp_width"] = width
        if "world_position" in normalized and "world_rotation" in normalized:
            position = np.asarray(normalized["world_position"], dtype=np.float64)
            rotation = np.asarray(normalized["world_rotation"], dtype=np.float64)
            return normalized if position.shape == (3,) and rotation.shape == (4,) and np.all(np.isfinite(position)) and np.all(np.isfinite(rotation)) and np.isclose(np.linalg.norm(rotation), 1.0, atol=1e-5) else None

        local_position = np.asarray(normalized["local_position"], dtype=np.float64)
        local_rotation = np.asarray(normalized["local_rotation"], dtype=np.float64)
        if (local_position.shape != (3,) or local_rotation.shape != (3, 3)
                or not np.all(np.isfinite(local_position)) or not np.all(np.isfinite(local_rotation))
                or not np.allclose(local_rotation.T @ local_rotation, np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(local_rotation), 1.0, atol=1e-5)):
            return None

        from scipy.spatial.transform import Rotation as Rot
        object_rotation = Rot.from_euler("xyz", obj.rotation, degrees=False).as_matrix()
        world_rotation = object_rotation @ local_rotation
        world_position = object_rotation @ local_position + np.asarray(obj.position, dtype=np.float64)
        world_matrix = np.eye(4, dtype=np.float64)
        world_matrix[:3, :3] = world_rotation
        world_matrix[:3, 3] = world_position
        normalized.update({
            "world_position": world_position,
            "world_rotation": Rot.from_matrix(world_rotation).as_quat(),
            "world_transformation_matrix": world_matrix,
            "world_approach_direction": world_rotation[:, 2],
            "world_finger_closing_direction": world_rotation[:, 0],
        })
        if "local_force_direction" in normalized:
            force = object_rotation @ np.asarray(normalized["local_force_direction"], dtype=np.float64)
            if force.shape != (3,) or not np.all(np.isfinite(force)) or np.linalg.norm(force) <= 0.0:
                return None
            normalized["world_force_direction"] = force / np.linalg.norm(force)
        return normalized
    except (KeyError, TypeError, ValueError):
        return None


def _add_grasp_to_stage(stage, grasp_root, grasp_idx, spec, delta_x, delta_z, scale,
                         component=None, params=None, extra_attrs=None):

    if component is not None:
        semantic_name = _safe_prim_name(component["semantic"]).lower()
        grasp_path = f"{grasp_root}/{semantic_name}_{component['index']}_grasp_{grasp_idx}"
    else:
        grasp_path = f"{grasp_root}/grasp_{grasp_idx}"

    g_xform = UsdGeom.Xform.Define(stage, grasp_path)
    g_xform.ClearXformOpOrder()

    usd_pos = _concept_to_usd_pos(spec["world_position"], delta_x, delta_z) * scale
    g_xform.AddTranslateOp().Set(Gf.Vec3d(float(usd_pos[0]), float(usd_pos[1]), float(usd_pos[2])))

    usd_q = _concept_to_usd_quat(np.asarray(spec["world_rotation"], dtype=np.float64))
    g_xform.AddOrientOp().Set(Gf.Quatf(float(usd_q[3]), float(usd_q[0]), float(usd_q[1]), float(usd_q[2])))

    prim = g_xform.GetPrim()

    if component is not None:
        prim.CreateAttribute("grasp:component", Sdf.ValueTypeNames.String).Set(str(component["semantic"]))

    approach = _concept_to_usd_vec(spec.get("world_approach_direction", [0.0, 0.0, 1.0]))
    prim.CreateAttribute("grasp:approach", Sdf.ValueTypeNames.Vector3f).Set(
        Gf.Vec3f(float(approach[0]), float(approach[1]), float(approach[2])))

    if "world_finger_closing_direction" in spec:
        fc = _concept_to_usd_vec(spec["world_finger_closing_direction"])
        prim.CreateAttribute("grasp:finger_closing", Sdf.ValueTypeNames.Vector3f).Set(
            Gf.Vec3f(float(fc[0]), float(fc[1]), float(fc[2])))

    if "grasp_width" in spec:
        prim.CreateAttribute("grasp:width", Sdf.ValueTypeNames.Float).Set(
            float(spec["grasp_width"] * scale))

    if "manip_params_size" in spec:
        prim.CreateAttribute("grasp:manip_params_size", Sdf.ValueTypeNames.Int).Set(
            int(spec["manip_params_size"]))

    if params is not None:
        prim.CreateAttribute("grasp:manipulation_params", Sdf.ValueTypeNames.FloatArray).Set(
            [float(x) for x in params])

    if extra_attrs:
        for attr_name, attr_val in extra_attrs.items():
            if isinstance(attr_val, (list, tuple)) and len(attr_val) == 3:
                vec = _concept_to_usd_vec(attr_val)
                prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Vector3f).Set(
                    Gf.Vec3f(float(vec[0]), float(vec[1]), float(vec[2])))

    t_mat = _concept_to_usd_matrix(
        spec.get("world_transformation_matrix", np.eye(4)), delta_x, delta_z)
    t_mat[:3, 3] *= scale
    prim.CreateAttribute("grasp:pose_matrix", Sdf.ValueTypeNames.FloatArray).Set(
        t_mat.flatten().tolist())

    return grasp_idx + 1


def process_bottle_objects(object_dirs):
    ct, gt, kd, ku, _ = _import_category_modules("Bottle")
    lookup = _build_template_lookup(ct, gt)
    cfg = CATEGORY_INFO["Bottle"]

    for obj_dir in object_dirs:
        pkl_dir = os.path.join(obj_dir, "conceptualization")
        if not os.path.isdir(pkl_dir):
            print(f"  [SKIP] No conceptualization for {os.path.basename(obj_dir)}")
            continue
        pkl_files = [f for f in os.listdir(pkl_dir) if f.endswith(".pkl")]
        if not pkl_files:
            print(f"  [SKIP] No pkl for {os.path.basename(obj_dir)}")
            continue
        pkl_path = os.path.join(pkl_dir, pkl_files[0])

        usd_path = os.path.join(obj_dir, SIM_READY_DIRNAME, cfg["usd_filename"])
        if not os.path.exists(usd_path):
            print(f"  [SKIP] No USD: {usd_path}")
            continue

        print(f"  Processing: {os.path.basename(obj_dir)}")

        concept_data = _load_pkl(pkl_path)

        body_template = None
        body_pos = None
        grasp_targets = []
        for c in concept_data["conceptualization"]:
            template_name = c.get("template")
            if template_name is None:
                continue
            cls = lookup.get(template_name)
            if cls is None:
                continue
            obj = cls(**c["parameters"])
            pos = np.array(c["parameters"].get("position", [0, 0, 0]), dtype=np.float64)
            if template_name in BODY_TEMPLATE_NAMES and body_template is None:
                body_template = template_name
                body_pos = pos
            if kd.get_grasp_spec(obj) is not None:
                grasp_targets.append(obj)

        if not grasp_targets:
            print(f"    [SKIP] No grasp target objects found")
            continue

        stage = Usd.Stage.Open(usd_path)
        if not stage:
            print(f"    [ERROR] Cannot open USD: {usd_path}")
            continue

        delta_x, delta_z, usd_body = _find_body_usd_offset(
            stage, cfg["root_path"], body_template if body_template else "", body_pos if body_pos is not None else np.zeros(3))
        if usd_body is None:
            print(f"    [WARN] No body Xform found in USD, using Δx=0, Δz=0")
            delta_x, delta_z = 0.0, 0.0

        grasp_root = f"{cfg['root_path']}/grasps"
        _reset_grasps(stage, grasp_root)

        count = 0
        for obj in grasp_targets:
            default_spec = kd.get_grasp_spec(obj)
            if default_spec is None:
                continue
            manip_params_size = int(default_spec["manip_params_size"])
            if manip_params_size == 1:
                test_params_list = [(0.0,), (0.5,), (1.0,), (1.5,), (2.0,)]
            elif manip_params_size == 2:
                test_params_list = [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (1.5, 0.0), (2.0, 0.0), (0.0, 1.0)]
            else:
                test_params_list = [tuple([0.0] * manip_params_size)]

            for params in test_params_list:
                spec = kd.get_grasp_spec(obj, manipulation_params=params)
                if spec is None:
                    continue
                count = _add_grasp_to_stage(stage, grasp_root, count, spec, delta_x, delta_z, 1.0)

        stage.GetRootLayer().Save()
        print(f"    Added {count} grasps  (Δx={delta_x:.6f}, Δz={delta_z:.6f})")


def process_mug_objects(object_dirs):
    ct, gt, kd, ku, _ = _import_category_modules("Mug")
    lookup = _build_template_lookup(ct, gt)
    cfg = CATEGORY_INFO["Mug"]

    mug_handle_types = set(k for k in lookup if "Handle" in k)
    test_params = [(-2.8, -1, 0), (-2.8, 1, 0), (-1.0, -1, 0), (-1.0, 1, 0)]

    for obj_dir in object_dirs:
        pkl_dir = os.path.join(obj_dir, "conceptualization")
        if not os.path.isdir(pkl_dir):
            print(f"  [SKIP] No conceptualization for {os.path.basename(obj_dir)}")
            continue
        pkl_files = [f for f in os.listdir(pkl_dir) if f.endswith(".pkl")]
        if not pkl_files:
            print(f"  [SKIP] No pkl for {os.path.basename(obj_dir)}")
            continue
        pkl_path = os.path.join(pkl_dir, pkl_files[0])

        usd_path = os.path.join(obj_dir, SIM_READY_DIRNAME, cfg["usd_filename"])
        if not os.path.exists(usd_path):
            print(f"  [SKIP] No USD: {usd_path}")
            continue

        print(f"  Processing: {os.path.basename(obj_dir)}")

        concept_data = _load_pkl(pkl_path)

        body_template = None
        body_pos = None
        handle_objs = []
        for c in concept_data["conceptualization"]:
            template_name = c.get("template")
            if template_name is None:
                continue
            if template_name in BODY_TEMPLATE_NAMES and body_template is None:
                body_template = template_name
                body_pos = np.array(c["parameters"].get("position", [0, 0, 0]), dtype=np.float64)
                continue
            if template_name not in mug_handle_types:
                continue
            cls = lookup.get(template_name)
            if cls is None:
                continue
            obj = cls(**c["parameters"])
            handle_objs.append(obj)

        if not handle_objs:
            print(f"    [SKIP] No handle objects found")
            continue

        stage = Usd.Stage.Open(usd_path)
        if not stage:
            print(f"    [ERROR] Cannot open USD: {usd_path}")
            continue

        delta_x, delta_z, usd_body = _find_body_usd_offset(
            stage, cfg["root_path"], body_template if body_template else "", body_pos if body_pos is not None else np.zeros(3))
        if usd_body is None:
            print(f"    [WARN] No body Xform found in USD, using Δx=0, Δz=0")
            delta_x, delta_z = 0.0, 0.0

        grasp_root = f"{cfg['root_path']}/grasps"
        _reset_grasps(stage, grasp_root)

        count = 0
        for h_obj in handle_objs:
            for p1, p2, p3 in test_params:
                spec = kd.get_grasp_spec(h_obj, manipulation_params=(p1, p2, p3))
                if spec is None:
                    continue
                if "world_position" not in spec or "world_rotation" not in spec:
                    continue
                count = _add_grasp_to_stage(stage, grasp_root, count, spec, delta_x, delta_z, 1.0)

        stage.GetRootLayer().Save()
        print(f"    Added {count} grasps  (Δx={delta_x:.6f}, Δz={delta_z:.6f})")


def process_shampoo_objects(object_dirs):
    ct, gt, kd, ku, utils = _import_category_modules("Shampoo")
    lookup = _build_template_lookup(ct, gt)
    cfg = CATEGORY_INFO["Shampoo"]

    for obj_dir in object_dirs:
        pkl_dir = os.path.join(obj_dir, "conceptualization")
        if not os.path.isdir(pkl_dir):
            print(f"  [SKIP] No conceptualization for {os.path.basename(obj_dir)}")
            continue
        pkl_files = [f for f in os.listdir(pkl_dir) if f.endswith(".pkl")]
        if not pkl_files:
            print(f"  [SKIP] No pkl for {os.path.basename(obj_dir)}")
            continue
        pkl_path = os.path.join(pkl_dir, pkl_files[0])

        usd_path = os.path.join(obj_dir, SIM_READY_DIRNAME, cfg["usd_filename"])
        if not os.path.exists(usd_path):
            print(f"  [SKIP] No USD: {usd_path}")
            continue

        print(f"  Processing: {os.path.basename(obj_dir)}")

        concept_data = _load_pkl(pkl_path)

        body_template = None
        body_pos = None
        grasp_target_objs = []
        for c in concept_data["conceptualization"]:
            template_name = c.get("template")
            if template_name is None:
                continue

            class_name = SHAMPOO_TEMPLATE_ALIASES.get(template_name, template_name)
            cls = lookup.get(class_name)
            if cls is None:
                continue

            params = dict(c["parameters"])
            if class_name == "Regular_nozzle":
                parts_params = list(params.get("parts_params", []))
                other_size_values = []
                for key in ("nozzle_size", "nozzle_length", "nozzle_offset"):
                    value = params.get(key)
                    if isinstance(value, (list, tuple)):
                        other_size_values.extend(abs(float(v)) for v in value)
                if parts_params and other_size_values:
                    max_part_param = max(abs(float(v)) for v in parts_params)
                    max_other_param = max(other_size_values)
                    if max_part_param > 1.0 and max_other_param < 1.0:
                        params["parts_params"] = [float(v) * 0.001 for v in parts_params]

            obj = cls(**params)

            if class_name in BODY_TEMPLATE_NAMES and body_template is None:
                body_template = class_name
                body_pos = np.array(params.get("position", [0, 0, 0]), dtype=np.float64)

            if kd.get_grasp_spec(obj) is not None:
                grasp_target_objs.append(obj)

        if not grasp_target_objs:
            print(f"    [SKIP] No grasp target objects found")
            continue

        stage = Usd.Stage.Open(usd_path)
        if not stage:
            print(f"    [ERROR] Cannot open USD: {usd_path}")
            continue

        delta_x, delta_z, usd_body = _find_body_usd_offset(
            stage, cfg["root_path"], body_template if body_template else "", body_pos if body_pos is not None else np.zeros(3))
        if usd_body is None:
            print(f"    [WARN] No body Xform found in USD, using Δx=0, Δz=0")
            delta_x, delta_z = 0.0, 0.0

        grasp_root = f"{cfg['root_path']}/grasps"
        _reset_grasps(stage, grasp_root)

        count = 0
        for obj in grasp_target_objs:
            default_spec = kd.get_grasp_spec(obj)
            if default_spec is None:
                continue
            params_list = _sample_component_grasp_params(obj, int(default_spec["manip_params_size"]))
            for params in params_list:
                spec = kd.get_grasp_spec(obj, manipulation_params=params)
                if spec is None:
                    continue
                if "world_position" not in spec or "world_rotation" not in spec:
                    continue
                count = _add_grasp_to_stage(stage, grasp_root, count, spec, delta_x, delta_z, 1.0,
                                              params=params)

        stage.GetRootLayer().Save()
        print(f"    Added {count} grasps  (Δx={delta_x:.6f}, Δz={delta_z:.6f})")


def process_generic_objects(object_dirs, category_name):
    cfg = CATEGORY_INFO[category_name]
    try:
        ct, gt, kd, ku, utils = _import_category_modules(cfg["dir"])
    except ImportError as exc:
        print(f"  [SKIP] No grasp pose module for {category_name}: {exc}")
        return
    lookup = _build_template_lookup(ct, gt)

    for obj_dir in object_dirs:
        pkl_dir = os.path.join(obj_dir, "conceptualization")
        if not os.path.isdir(pkl_dir):
            print(f"  [SKIP] No conceptualization for {os.path.basename(obj_dir)}")
            continue
        pkl_files = [f for f in os.listdir(pkl_dir) if f.endswith(".pkl")]
        if not pkl_files:
            print(f"  [SKIP] No pkl for {os.path.basename(obj_dir)}")
            continue
        pkl_path = os.path.join(pkl_dir, pkl_files[0])

        usd_path = os.path.join(obj_dir, SIM_READY_DIRNAME, cfg["usd_filename"])
        if not os.path.exists(usd_path):
            print(f"  [SKIP] No USD: {usd_path}")
            continue

        print(f"  Processing: {os.path.basename(obj_dir)}")

        concept_data = _load_pkl(pkl_path)

        body_template = None
        body_pos = None
        components = []
        for idx, c in enumerate(concept_data["conceptualization"]):
            template_name = c.get("template")
            if template_name is None:
                continue
            cls = lookup.get(template_name)
            if cls is None:
                continue
            try:
                obj = cls(**c["parameters"])
            except Exception as exc:
                print(f"    [WARN] Cannot instantiate {template_name} in {os.path.basename(obj_dir)}: {exc}")
                continue
            if template_name in BODY_TEMPLATE_NAMES and body_template is None:
                body_template = template_name
                body_pos = np.array(c["parameters"].get("position", [0, 0, 0]), dtype=np.float64)
            semantic = getattr(obj, "semantic", template_name)
            components.append({
                "index": idx, "template": template_name, "semantic": semantic, "obj": obj,
            })

        stage = Usd.Stage.Open(usd_path)
        if not stage:
            print(f"    [ERROR] Cannot open USD: {usd_path}")
            continue

        delta_x, delta_z, usd_body = _find_body_usd_offset(
            stage, cfg["root_path"], body_template if body_template else "", body_pos if body_pos is not None else np.zeros(3))
        if usd_body is None:
            print(f"    [WARN] No body Xform found in USD, using Δx=0, Δz=0")
            delta_x, delta_z = 0.0, 0.0

        grasp_root = f"{cfg['root_path']}/grasps"
        _reset_grasps(stage, grasp_root)

        grasp_idx = 0
        for component in components:
            obj = component["obj"]
            try:
                default_spec = kd.get_grasp_spec(obj)
            except Exception as exc:
                print(f"    [WARN] Cannot get default grasp for {component['template']}: {exc}")
                continue
            if default_spec is None:
                continue

            params_list = ([(0.0, 0.0, 0.0, 0.0)] if category_name == "StorageFurniture"
                           else _sample_component_grasp_params(obj, int(default_spec["manip_params_size"])))
            for params in params_list:
                try:
                    spec = kd.get_grasp_spec(obj, manipulation_params=params)
                except Exception as exc:
                    print(f"    [WARN] Cannot get grasp for {component['template']} params={params}: {exc}")
                    continue
                spec = _normalize_grasp_spec(obj, spec)
                if spec is None:
                    continue
                grasp_idx = _add_grasp_to_stage(
                    stage, grasp_root, grasp_idx, spec, delta_x, delta_z, 1.0,
                    component=component, params=params)

        stage.GetRootLayer().Save()
        print(f"    Added {grasp_idx} grasps  (Δx={delta_x:.6f}, Δz={delta_z:.6f})")


def process_box_objects(object_dirs):
    ct, gt, kd, ku, _ = _import_category_modules("Box")
    lookup = _build_template_lookup(ct, gt)
    cfg = CATEGORY_INFO["Box"]

    RegularCover = lookup.get("Regular_Cover")
    FourfoldCover = lookup.get("Fourfold_Cover")
    CuboidalBody = lookup.get("Cuboidal_Body")

    def _grasp_params_for_component(obj):
        if RegularCover is not None and isinstance(obj, RegularCover):
            return [(0.0, 0.0), (-0.5, -0.5), (0.5, 0.5), (-0.5, 0.5), (0.5, -0.5)]
        if FourfoldCover is not None and isinstance(obj, FourfoldCover):
            return [
                (0.0, 0.0, -1.0), (0.0, 0.0, 1.0),
                (-0.5, -0.5, -1.0), (0.5, 0.5, -1.0),
                (-0.5, 0.5, 1.0), (0.5, -0.5, 1.0),
            ]
        if CuboidalBody is not None and isinstance(obj, CuboidalBody):
            return [(0.0, 0.0)]
        return _sample_params(0)

    for obj_dir in object_dirs:
        pkl_dir = os.path.join(obj_dir, "conceptualization")
        if not os.path.isdir(pkl_dir):
            print(f"  [SKIP] No conceptualization for {os.path.basename(obj_dir)}")
            continue
        pkl_files = [f for f in os.listdir(pkl_dir) if f.endswith(".pkl")]
        if not pkl_files:
            print(f"  [SKIP] No pkl for {os.path.basename(obj_dir)}")
            continue
        pkl_path = os.path.join(pkl_dir, pkl_files[0])

        usd_path = os.path.join(obj_dir, SIM_READY_DIRNAME, cfg["usd_filename"])
        if not os.path.exists(usd_path):
            print(f"  [SKIP] No USD: {usd_path}")
            continue

        print(f"  Processing: {os.path.basename(obj_dir)}")

        concept_data = _load_pkl(pkl_path)

        body_template = None
        body_pos = None
        components = []
        for idx, c in enumerate(concept_data["conceptualization"]):
            template_name = c.get("template")
            if template_name is None:
                continue
            cls = lookup.get(template_name)
            if cls is None:
                continue
            obj = cls(**c["parameters"])
            if template_name in BODY_TEMPLATE_NAMES and body_template is None:
                body_template = template_name
                body_pos = np.array(c["parameters"].get("position", [0, 0, 0]), dtype=np.float64)
            semantic = getattr(obj, "semantic", template_name)
            components.append({
                "index": idx, "template": template_name, "semantic": semantic, "obj": obj,
            })

        stage = Usd.Stage.Open(usd_path)
        if not stage:
            print(f"    [ERROR] Cannot open USD: {usd_path}")
            continue

        delta_x, delta_z, usd_body = _find_body_usd_offset(
            stage, cfg["root_path"], body_template if body_template else "", body_pos if body_pos is not None else np.zeros(3))
        if usd_body is None:
            print(f"    [WARN] No body Xform found in USD, using Δx=0, Δz=0")
            delta_x, delta_z = 0.0, 0.0

        grasp_root = f"{cfg['root_path']}/grasps"
        _reset_grasps(stage, grasp_root)

        count = 0
        for component in components:
            obj = component["obj"]
            for params in _grasp_params_for_component(obj):
                spec = kd.get_grasp_spec(obj, manipulation_params=params)
                if spec is None:
                    continue
                extra = {}
                if "world_force_direction" in spec:
                    extra["grasp:force_direction"] = spec["world_force_direction"]
                count = _add_grasp_to_stage(
                    stage, grasp_root, count, spec, delta_x, delta_z, 1.0,
                    component=component, params=params, extra_attrs=extra)

        stage.GetRootLayer().Save()
        print(f"    Added {count} grasps  (Δx={delta_x:.6f}, Δz={delta_z:.6f})")


def main():
    target_path = os.path.join(_SCRIPT_BASE, "config", "target_batches.json")
    target_batches = None
    attach_batches = None
    if os.path.exists(target_path):
        with open(target_path, "r", encoding="utf-8") as f:
            target_batches = set(json.load(f))
    if os.environ.get("ATTACH_BATCHES"):
        attach_batches = {
            x.strip() for x in os.environ["ATTACH_BATCHES"].split(",")
            if x.strip()
        }

    all_dirs = sorted(
        d for d in os.listdir(REVIEWER_BASE)
        if os.path.isdir(os.path.join(REVIEWER_BASE, d)) and d.startswith("batch_")
    )
    if attach_batches is not None:
        all_dirs = [d for d in all_dirs if d in attach_batches]
    elif target_batches is not None:
        all_dirs = [d for d in all_dirs if d in target_batches]
    print(
        f"[INFO] Target filter: "
        f"{len(attach_batches) if attach_batches is not None else len(target_batches) if target_batches is not None else 'ALL'} "
        f"targets; matched {len(all_dirs)} directories")

    category_objects = {}
    skipped = []
    for d in all_dirs:
        cat = _extract_category(d)
        if cat is None:
            skipped.append(d)
            continue
        cat = CATEGORY_NORMALIZE.get(cat, cat)
        if cat not in CATEGORY_INFO:
            skipped.append(d)
            continue
        category_objects.setdefault(cat, []).append(os.path.join(REVIEWER_BASE, d))

    requested_categories = os.environ.get("ATTACH_CATEGORIES")
    if requested_categories:
        requested = {
            CATEGORY_NORMALIZE.get(c.strip(), c.strip())
            for c in requested_categories.split(",")
            if c.strip()
        }
        category_objects = {k: v for k, v in category_objects.items() if k in requested}
        print(f"[INFO] Category filter: {sorted(requested)}")
    print(f"[INFO] Categories to process: {', '.join(f'{k}:{len(v)}' for k, v in sorted(category_objects.items())) or 'none'}")
    if skipped:
        print("[SKIPPED objects (no grasp pose code)]:")
        for s in skipped:
            cat = _extract_category(s) or "UNKNOWN"
            print(f"  - {s}  (category: {cat})")
        print()

    PROCESSORS = {
        "Bottle": process_bottle_objects,
        "Mug": process_mug_objects,
        "Shampoo": process_shampoo_objects,
        "Gluestick": lambda dirs: process_generic_objects(dirs, "Gluestick"),
        "Ruler": lambda dirs: process_generic_objects(dirs, "Ruler"),
        "Pen": lambda dirs: process_generic_objects(dirs, "Pen"),
        "USB": lambda dirs: process_generic_objects(dirs, "USB"),
        "Chair": lambda dirs: process_generic_objects(dirs, "Chair"),
        "KitchenPot": lambda dirs: process_generic_objects(dirs, "KitchenPot"),
        "Doorhandle": lambda dirs: process_generic_objects(dirs, "Doorhandle"),
        "Knife": lambda dirs: process_generic_objects(dirs, "Knife"),
        "Pliers": lambda dirs: process_generic_objects(dirs, "Pliers"),
        "Scissors": lambda dirs: process_generic_objects(dirs, "Scissors"),
        "Switch": lambda dirs: process_generic_objects(dirs, "Switch"),
        "Table": lambda dirs: process_generic_objects(dirs, "Table"),
        "Foldingrack": lambda dirs: process_generic_objects(dirs, "Foldingrack"),
        "Bowl": lambda dirs: process_generic_objects(dirs, "Bowl"),
        "Clip": lambda dirs: process_generic_objects(dirs, "Clip"),
        "Lighter": lambda dirs: process_generic_objects(dirs, "Lighter"),
        "Mouse": lambda dirs: process_generic_objects(dirs, "Mouse"),
        "Spoon": lambda dirs: process_generic_objects(dirs, "Spoon"),
        "StorageFurniture": lambda dirs: process_generic_objects(dirs, "StorageFurniture"),
        "Box": process_box_objects,
    }

    for cat in sorted(category_objects.keys()):
        obj_dirs = category_objects[cat]
        print(f"\n{'='*60}")
        print(f"Category: {cat} ({len(obj_dirs)} objects)")
        print(f"{'='*60}")

        processor = PROCESSORS.get(cat)
        if processor:
            try:
                processor(obj_dirs)
            except Exception:
                traceback.print_exc()
                raise
        else:
            print(f"  [SKIP] No processor for {cat}")

    unavailable = {"Bowl", "Mouse", "Spoon"}
    report = {}
    for batch in all_dirs:
        cat = CATEGORY_NORMALIZE.get(_extract_category(batch), _extract_category(batch))
        cfg = CATEGORY_INFO.get(cat)
        row = {"category": cat, "bound_count": 0}
        if cfg is None:
            row["status"] = "no_template_definition"
        else:
            usd_path = os.path.join(REVIEWER_BASE, batch, SIM_READY_DIRNAME, cfg["usd_filename"])
            stage = Usd.Stage.Open(usd_path) if os.path.exists(usd_path) else None
            root = stage.GetPrimAtPath(cfg["root_path"]) if stage else None
            if not root:
                row["status"] = "target_prim_missing"
            else:
                row["bound_count"] = _grasp_count(stage, cfg["root_path"])
                row["status"] = "defined" if row["bound_count"] else (
                    "definition_import_error" if cat in unavailable else "no_effective_definition")
        report[batch] = row
    report_dir = REPORT_DIR
    os.makedirs(report_dir, exist_ok=True)
    with open(os.path.join(report_dir, "grasp_attachment_report.json"), "w", encoding="utf-8") as f:
        json.dump({"sim_ready_dirname": SIM_READY_DIRNAME, "assets": report}, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("All done!")


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
