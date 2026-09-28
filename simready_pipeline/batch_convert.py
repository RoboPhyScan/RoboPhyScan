import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import omni.kit.commands
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf
from collision_vhacd import prepare_vhacd_urdf
from usd_reference_repair import repair_usd_references

ROOT = Path(__file__).resolve().parent
ASSET_ROOT = Path(os.environ.get("REVIEWER_BASE", ROOT)).resolve()
CONFIG_DIR = ROOT / "config"
TARGET_BATCHES_PATH = CONFIG_DIR / "target_batches.json"
MODULAR_WORKSPACE = ROOT
MODULAR_SELECTOR_PATH = CONFIG_DIR / "modular_assets.json"
MODULAR_BINDINGS_PATH = CONFIG_DIR / "interaction_bindings.json"
URDF_SOURCE_DIRNAME = os.environ.get("REVIEWER_URDF_SOURCE_DIRNAME", "urdf_source")
FINAL_ASSET_DIRNAME = os.environ.get("REVIEWER_SIM_READY_DIRNAME", "sim_ready")
REPORT_DIR = Path(os.environ.get("PIPELINE_REPORT_DIR", ASSET_ROOT / ".pipeline_reports")).resolve()


def load_target_batches():
    if not TARGET_BATCHES_PATH.exists():
        return None
    with open(TARGET_BATCHES_PATH, "r", encoding="utf-8") as f:
        return set(json.load(f))

TARGET_BINDING_OVERRIDES = {
    "batch_02_0043_Bottle_009_Mushroom_Meat_Sauce": {
        "Multilevel_Body": ["Multilevel_Body", "closure"],
    },
    "batch_01_0009_Bottle_008_mug_thermos": {
        "Cylindrical_Lid_virtual_prismatic": ["Cylindrical_Lid"],
        "Cylindrical_Lid_base": ["Cylindrical_Lid"],
        "Cylindrical_Connector": ["Cylindrical_Lid"],
        "Cylindrical_Lid_cover": ["Cylindrical_Lid"],
    },
    "batch_02_0003_Doorhandle_001_Bathroom_Door_Handle": {
        "Regular_handle_base_0": ["Regular_handle_0"],
        "Regular_handle_grip_0": ["Regular_handle_1"],
        "Regular_handle_base_1": ["Regular_handle_2"],
        "Regular_handle_grip_1": ["Regular_handle_1"],
    },
    "batch_02_0034_KitchenPot_001_Floral_Enamel_Pot": {
        "Multilevel_Tophandle": ["Cylindrical_Body_Cap"],
        "Trifold_Sidehandle": ["Cylindrical_Body_Cap"],
    },
    "batch_01_0015_Pliers_001_industrial_needlenose_pliers": {
        "Rear_Curved_Handle_left": ["Rear_Curved_Handle_0"],
        "Rear_Curved_Handle_right": ["Rear_Curved_Handle_1"],
        "Cusp_Gripper_left": ["Cusp_Gripper_0", "Pivot_Joint_0"],
        "Cusp_Gripper_right": ["Cusp_Gripper_1", "Pivot_Joint_1"],
    },
    "batch_02_0033_Ruler_002_Ultraman_Themed_Rulers": {
        "Regular_shaft": ["Regular_shaft", "regular_shaft_2"],
        "Asymmetrical_body_Arm1": ["Asymmetrical_body_Arm1", "symmetrical_body"],
        "Asymmetrical_body_Arm2": ["Asymmetrical_body_Arm2", "symmetrical_body"],
    },
    "batch_01_0012_Scissors_001_black": {
        "Curved_Blade_left": ["Curved_Blade_0"],
        "Curved_Blade_right": ["Curved_Blade_1"],
        "Half_Ring_Handle_left": ["Half_Ring_Handle_0"],
        "Half_Ring_Handle_right": ["Half_Ring_Handle_1"],
        "Cylindrical_Shaft": ["Cylindrical_Shaft_0", "Cylindrical_Shaft_1"],
        "Double_Cuboidal_Shaft_left_outer": ["Cylindrical_Shaft_0"],
        "Double_Cuboidal_Shaft_right_outer": ["Cylindrical_Shaft_1"],
    },
    "batch_02_0044_Scissors_001_Safety_Kids_Scissors": {
        "Curved_Blade_left": ["Curved_Blade_0"],
        "Curved_Blade_right": ["Curved_Blade_1"],
        "Half_Ring_Handle_left": ["Half_Ring_Handle_0"],
        "Half_Ring_Handle_right": ["Half_Ring_Handle_1"],
        "Cylindrical_Shaft": ["Cylindrical_Shaft_0", "Cylindrical_Shaft_1"],
        "Double_Cuboidal_Shaft_left_outer": ["Cylindrical_Shaft_0"],
        "Double_Cuboidal_Shaft_right_outer": ["Cylindrical_Shaft_1"],
    },
    "batch_02_0016_Bottle_004_Cute_Animal_Thermos": {
        "Cylindrical_Lid_base": ["Cylindrical_Lid"],
        "Cylindrical_Lid_cover": ["Cylindrical_Lid"],
    },
    "batch_02_0017_Bottle_005_Disney_Mickey_Thermos": {
        "Cylindrical_Lid_base": ["Cylindrical_Lid"],
        "Cylindrical_Lid_cover": ["Cylindrical_Lid"],
    },
    "batch_01_0044_Table_001_white_study": {
        "Cable_stayed_leg_p1_id0": ["Cable_stayed_leg"],
        "Cable_stayed_leg_p1_id1": ["Cable_stayed_leg"],
        "Cable_stayed_leg_p1_id2": ["Cable_stayed_leg"],
        "Cable_stayed_leg_p1_id3": ["Cable_stayed_leg"],
        "Cable_stayed_leg_p1_id4": ["Cable_stayed_leg"],
        "Cable_stayed_leg_p1_id5": ["Cable_stayed_leg"],
        "Regular_partition_p2_id0": ["Regular_desktop"],
    },
}

for _shampoo_batch in (
    "batch_01_0027_Shampoo_001_lux_shower_gel",
    "batch_01_0032_Sanitizer_001_liquid_soap",
    "batch_02_0007_Shampoo_001_Enchanteur_Body_Wash",
    "batch_02_0008_Shampoo_002_Dettol_Hand_Wash",
    "batch_02_0009_Shampoo_003_Dettol_Foam_Hand_Wash",
):
    TARGET_BINDING_OVERRIDES.setdefault(_shampoo_batch, {}).update({
        "Regular_nozzle_virtual_prismatic": ["Regular_nozzle"],
        "Regular_nozzle_Head_virtual_prismatic": ["Regular_nozzle_Head"],
    })

TARGET_BINDING_OVERRIDES.setdefault("batch_01_0035_Lighter_001_chofn", {}).update({
    "Cambered_Nozzle": ["Cambered_Nozzle"],
})
for _thermos_batch in (
    "batch_02_0016_Bottle_004_Cute_Animal_Thermos",
    "batch_02_0017_Bottle_005_Disney_Mickey_Thermos",
):
    TARGET_BINDING_OVERRIDES.setdefault(_thermos_batch, {}).update({
        "Cylindrical_Connector": ["Cylindrical_Connector"],
        "Cylindrical_Button": ["Cylindrical_Button"],
    })
TARGET_BINDING_OVERRIDES.setdefault("batch_02_0045_Pen_001_Multi-color_Ballpoint_Pens", {}).update({
    "Cylindrical_Refill": ["Cylindrical_Refill"],
    "Trifold_Clip": ["Trifold_Clip"],
})

REBOUND_JOINT_CONTROLS = {
    "batch_01_0035_Lighter_001_chofn": {
        "Cambered_Body_to_L_Shaped_Button": {
            "drive_type": "linear",
            "target_position": 0.0,
            "lower_limit": 0.0,
            "upper_limit": 0.004,
            "stiffness": 2000.0,
            "damping": 6.0,
            "max_force": 10.0,
        },
    },
    "batch_02_0003_Doorhandle_001_Bathroom_Door_Handle": {
        "Regular_handle_base_0_to_Regular_handle_grip_0": {
            "drive_type": "angular",
            "target_from_limit": "farther_from_zero",
            "stiffness": 12.0,
            "damping": 1.5,
            "max_force": 8.0,
        },
        "Regular_handle_base_1_to_Regular_handle_grip_1": {
            "drive_type": "angular",
            "target_from_limit": "farther_from_zero",
            "stiffness": 12.0,
            "damping": 1.5,
            "max_force": 8.0,
        },
    },
    "batch_01_0031_Clip_001_plastic_hanging": {
        "Regular_lever_to_Regular_lever_handle_left": {
            "drive_type": "angular",
            "target_position": 0.0,
            "stiffness": 4.0,
            "damping": 0.4,
            "max_force": 4.0,
        },
        "Regular_lever_to_Regular_lever_handle_right": {
            "drive_type": "angular",
            "target_position": 0.0,
            "stiffness": 4.0,
            "damping": 0.4,
            "max_force": 4.0,
        },
    },
    "batch_02_0020_Clip_001_Large_Laundry_Clips": {
        "Regular_lever_to_Regular_lever_handle_left": {
            "drive_type": "angular",
            "target_position": 0.0,
            "stiffness": 5.0,
            "damping": 0.5,
            "max_force": 5.0,
        },
        "Regular_lever_to_Regular_lever_handle_right": {
            "drive_type": "angular",
            "target_position": 0.0,
            "stiffness": 5.0,
            "damping": 0.5,
            "max_force": 5.0,
        },
    },
    "batch_02_0021_Clip_002_Yellow_Hair_Clips": {
        "Regular_lever_to_Regular_lever_handle_left": {
            "drive_type": "angular",
            "target_position": 0.0,
            "stiffness": 3.0,
            "damping": 0.35,
            "max_force": 3.0,
        },
        "Regular_lever_to_Regular_lever_handle_right": {
            "drive_type": "angular",
            "target_position": 0.0,
            "stiffness": 3.0,
            "damping": 0.35,
            "max_force": 3.0,
        },
    },
    "batch_01_0027_Shampoo_001_lux_shower_gel": {
        "Cylindrical_body_to_Regular_nozzle_virtual_prismatic": {
            "drive_type": "linear",
            "target_position": 0.0,
            "stiffness": 8000.0,
            "damping": 350.0,
            "max_force": 2000.0,
        },
        "Regular_nozzle_to_Regular_nozzle_Head_virtual_prismatic": {
            "drive_type": "linear",
            "target_position": 0.0,
            "stiffness": 8000.0,
            "damping": 350.0,
            "max_force": 2000.0,
        },
        "Regular_nozzle_virtual_prismatic_to_Regular_nozzle": {
            "drive_type": "angular",
            "target_position": 0.0,
            "stiffness": 18.0,
            "damping": 2.0,
            "max_force": 10.0,
        },
        "Regular_nozzle_Head_virtual_prismatic_to_Regular_nozzle_Head": {
            "drive_type": "angular",
            "target_position": 0.0,
            "stiffness": 10.0,
            "damping": 1.2,
            "max_force": 6.0,
        },
    },
    "batch_01_0032_Sanitizer_001_liquid_soap": {
        "Cylindrical_body_to_Regular_nozzle_virtual_prismatic": {
            "drive_type": "linear",
            "target_position": 0.0,
            "stiffness": 8000.0,
            "damping": 350.0,
            "max_force": 2000.0,
        },
        "Regular_nozzle_to_Regular_nozzle_Head_virtual_prismatic": {
            "drive_type": "linear",
            "target_position": 0.0,
            "stiffness": 8000.0,
            "damping": 350.0,
            "max_force": 2000.0,
        },
        "Regular_nozzle_virtual_prismatic_to_Regular_nozzle": {
            "drive_type": "angular",
            "target_position": 0.0,
            "stiffness": 18.0,
            "damping": 2.0,
            "max_force": 10.0,
        },
        "Regular_nozzle_Head_virtual_prismatic_to_Regular_nozzle_Head": {
            "drive_type": "angular",
            "target_position": 0.0,
            "stiffness": 10.0,
            "damping": 1.2,
            "max_force": 6.0,
        },
    },
    "batch_02_0007_Shampoo_001_Enchanteur_Body_Wash": {
        "Cylindrical_body_to_Regular_nozzle_virtual_prismatic": {
            "drive_type": "linear",
            "target_position": 0.0,
            "stiffness": 8000.0,
            "damping": 350.0,
            "max_force": 2000.0,
        },
        "Regular_nozzle_to_Regular_nozzle_Head_virtual_prismatic": {
            "drive_type": "linear",
            "target_position": 0.0,
            "stiffness": 8000.0,
            "damping": 350.0,
            "max_force": 2000.0,
        },
        "Regular_nozzle_virtual_prismatic_to_Regular_nozzle": {
            "drive_type": "angular",
            "target_position": 0.0,
            "stiffness": 18.0,
            "damping": 2.0,
            "max_force": 10.0,
        },
        "Regular_nozzle_Head_virtual_prismatic_to_Regular_nozzle_Head": {
            "drive_type": "angular",
            "target_position": 0.0,
            "stiffness": 10.0,
            "damping": 1.2,
            "max_force": 6.0,
        },
    },
    "batch_02_0008_Shampoo_002_Dettol_Hand_Wash": {
        "Cylindrical_body_to_Regular_nozzle_virtual_prismatic": {
            "drive_type": "linear",
            "target_position": 0.0,
            "stiffness": 8000.0,
            "damping": 350.0,
            "max_force": 2000.0,
        },
        "Regular_nozzle_to_Regular_nozzle_Head_virtual_prismatic": {
            "drive_type": "linear",
            "target_position": 0.0,
            "stiffness": 8000.0,
            "damping": 350.0,
            "max_force": 2000.0,
        },
        "Regular_nozzle_virtual_prismatic_to_Regular_nozzle": {
            "drive_type": "angular",
            "target_position": 0.0,
            "stiffness": 18.0,
            "damping": 2.0,
            "max_force": 10.0,
        },
        "Regular_nozzle_Head_virtual_prismatic_to_Regular_nozzle_Head": {
            "drive_type": "angular",
            "target_position": 0.0,
            "stiffness": 10.0,
            "damping": 1.2,
            "max_force": 6.0,
        },
    },
    "batch_02_0009_Shampoo_003_Dettol_Foam_Hand_Wash": {
        "Cylindrical_body_to_Regular_nozzle_virtual_prismatic": {
            "drive_type": "linear",
            "target_position": 0.0,
            "stiffness": 8000.0,
            "damping": 350.0,
            "max_force": 2000.0,
        },
        "Regular_nozzle_to_Regular_nozzle_Head_virtual_prismatic": {
            "drive_type": "linear",
            "target_position": 0.0,
            "stiffness": 8000.0,
            "damping": 350.0,
            "max_force": 2000.0,
        },
        "Regular_nozzle_virtual_prismatic_to_Regular_nozzle": {
            "drive_type": "angular",
            "target_position": 0.0,
            "stiffness": 18.0,
            "damping": 2.0,
            "max_force": 10.0,
        },
        "Regular_nozzle_Head_virtual_prismatic_to_Regular_nozzle_Head": {
            "drive_type": "angular",
            "target_position": 0.0,
            "stiffness": 10.0,
            "damping": 1.2,
            "max_force": 6.0,
        },
    },
}

FIXED_JOINT_LIMITS = {}

ASSEMBLY_PRISMATIC_LOCKS = {}

IGNORE_WORLD_POSE_SINGLE_LINK_BATCHES = {
    "batch_01_0028_Ruler_001_measuring_tool",
}


def load_item_components():
    with open(CONFIG_DIR / "item_components.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_modular_bindings():
    if not MODULAR_SELECTOR_PATH.exists() or not MODULAR_BINDINGS_PATH.exists():
        return {}, {"attach_grasp_poses": True}
    selector = json.loads(MODULAR_SELECTOR_PATH.read_text(encoding="utf-8"))
    binding_data = json.loads(MODULAR_BINDINGS_PATH.read_text(encoding="utf-8"))
    enabled = {
        item["batch"]: set(item.get("interaction_ids", []))
        for item in selector.get("assets", [])
        if item.get("enabled", False)
    }
    all_bindings = {item["id"]: item for item in binding_data.get("interactions", [])}
    selected = {}
    for batch, interaction_ids in enabled.items():
        missing = sorted(interaction_ids - all_bindings.keys())
        if missing:
            raise ValueError(f"modular_assets.json 引用了不存在的 interaction id: {missing}")
        selected[batch] = [all_bindings[item_id] for item_id in sorted(interaction_ids)]
    return selected, selector.get("pipeline", {})


def package_modular_asset(usd_path, batch_name, interactions):
    if not interactions:
        return None
    src_root = MODULAR_WORKSPACE
    package_root = MODULAR_WORKSPACE / "modular_interactions"
    for import_root in (src_root, package_root):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))
    from modular_interactions.package_asset_behaviors import package_asset

    resolved = json.loads(json.dumps(interactions))
    for interaction in resolved:
        interaction["target"]["usd"] = f"./{Path(usd_path).name}"
    return package_asset(Path(usd_path), resolved, asset_id=batch_name)


def run_grasp_for_batches(batches_by_root, sim_ready_dirname):
    if not batches_by_root:
        return 0
    result_code = 0
    for reviewer_base, batch_names in sorted(batches_by_root.items()):
        env = os.environ.copy()
        env["ATTACH_BATCHES"] = ",".join(sorted(batch_names))
        env["REVIEWER_BASE"] = reviewer_base
        env["REVIEWER_SIM_READY_DIRNAME"] = sim_ready_dirname
        command = [sys.executable, str(ROOT / "attach_grasp_poses.py")]
        result = subprocess.run(command, cwd=ROOT, env=env, check=False)
        result_code = result_code or int(result.returncode)
    return result_code


def _relativize_report(value):
    if isinstance(value, dict):
        return {key: _relativize_report(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_relativize_report(item) for item in value]
    if isinstance(value, str):
        try:
            return str(Path(value).resolve().relative_to(ASSET_ROOT)).replace("\\", "/")
        except (OSError, ValueError):
            return value
    return value


def write_batch_report(records, grasp_code, sim_ready_dirname):
    report_dir = REPORT_DIR
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / "vhacd_usd_batch_report.json"
    existing = {}
    if report_path.exists():
        try:
            existing = {item["batch"]: item for item in json.loads(
                report_path.read_text(encoding="utf-8")).get("assets", []) if item.get("batch")}
        except (OSError, json.JSONDecodeError):
            pass
    existing.update({item["batch"]: _relativize_report(item) for item in records if item.get("batch")})
    grasp_path = report_dir / "grasp_attachment_report.json"
    if grasp_path.exists():
        try:
            grasp_rows = json.loads(grasp_path.read_text(encoding="utf-8")).get("assets", {})
            for batch, grasp in grasp_rows.items():
                if batch in existing:
                    existing[batch]["grasp"] = _relativize_report(grasp)
        except (OSError, json.JSONDecodeError):
            pass
    report = {
        "output_dirname": sim_ready_dirname,
        "grasp_exit_code": grasp_code,
        "assets": [existing[name] for name in sorted(existing)],
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def discover_existing_outputs(sim_ready_dirname, excluded_batches):
    target_batches = load_target_batches()
    only_batches = None
    if os.environ.get("BATCH_CONVERT_ONLY"):
        only_batches = {x.strip() for x in os.environ["BATCH_CONVERT_ONLY"].split(",") if x.strip()}
    result = {}

    def maybe_add(item_dir):
        if not item_dir.is_dir() or not item_dir.name.startswith("batch_"):
            return
        if only_batches is not None and item_dir.name not in only_batches:
            return
        if only_batches is None and target_batches is not None and item_dir.name not in target_batches:
            return
        if item_dir.name in excluded_batches:
            return
        if any((item_dir / sim_ready_dirname).glob("*.usd")):
            result.setdefault(str(item_dir.parent.resolve()), set()).add(item_dir.name)

    for item_dir in sorted(ASSET_ROOT.iterdir()):
        maybe_add(item_dir)
    return result


def discover_urdfs():
    urdfs = []
    target_batches = load_target_batches()
    only_batches = None
    if os.environ.get("BATCH_CONVERT_ONLY"):
        only_batches = {x.strip() for x in os.environ["BATCH_CONVERT_ONLY"].split(",") if x.strip()}
    seen = set()

    def maybe_add_item_dir(item_dir):
        if not item_dir.is_dir() or not item_dir.name.startswith("batch_"):
            return
        vhacd_mode = os.environ.get("BATCH_CONVERT_VHACD", "1") != "0"
        if only_batches is not None and item_dir.name not in only_batches:
            return
        if only_batches is None and target_batches is not None and item_dir.name not in target_batches:
            return
        urdf_source = item_dir / URDF_SOURCE_DIRNAME
        if not urdf_source.is_dir():
            return
        for urdf_file in sorted(urdf_source.glob("*.urdf")):
            if urdf_file.stem.endswith("_no_world"):
                continue
            key = str(urdf_file.resolve())
            if key in seen:
                continue
            seen.add(key)
            batch_name = item_dir.name
            output_dir = item_dir / FINAL_ASSET_DIRNAME
            dest = output_dir / urdf_file.with_suffix(".usd").name
            if os.environ.get("BATCH_CONVERT_RESUME") == "1" and dest.exists():
                continue
            urdfs.append((str(urdf_file), str(dest), batch_name))

    for item_dir in sorted(ASSET_ROOT.iterdir()):
        maybe_add_item_dir(item_dir)
    return urdfs


def stage_urdf_source(urdf_path, output_dir):
    """Keep source URDFs immutable while retaining the importer's local mesh layout."""
    urdf_path = Path(urdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    staged_urdf = output_dir / urdf_path.name
    shutil.copy2(urdf_path, staged_urdf)
    source_meshes = urdf_path.parent / "meshes"
    if source_meshes.is_dir():
        shutil.copytree(source_meshes, output_dir / "meshes", dirs_exist_ok=True)
    return staged_urdf


def rpy_to_matrix(rpy):
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def matrix_to_rpy(matrix):
    r20 = matrix[2][0]
    if abs(r20) < 1.0 - 1e-9:
        pitch = math.asin(-r20)
        roll = math.atan2(matrix[2][1], matrix[2][2])
        yaw = math.atan2(matrix[1][0], matrix[0][0])
    else:
        pitch = math.pi / 2 if r20 <= -1.0 else -math.pi / 2
        roll = 0.0
        yaw = math.atan2(-matrix[0][1], matrix[1][1])
    return [roll, pitch, yaw]


def transform_from_origin(origin):
    xyz = [0.0, 0.0, 0.0]
    rpy = [0.0, 0.0, 0.0]
    if origin is not None:
        xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()]
        rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()]
    return xyz, rpy_to_matrix(rpy)


def invert_transform(xyz, rot):
    inv_rot = [[rot[j][i] for j in range(3)] for i in range(3)]
    inv_xyz = [-sum(inv_rot[i][j] * xyz[j] for j in range(3)) for i in range(3)]
    return inv_xyz, inv_rot


def compose_transform(a_xyz, a_rot, b_xyz, b_rot):
    xyz = [
        a_xyz[i] + sum(a_rot[i][j] * b_xyz[j] for j in range(3))
        for i in range(3)
    ]
    rot = [
        [sum(a_rot[i][k] * b_rot[k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]
    return xyz, rot


def relative_transform(parent_xyz, parent_rot, child_xyz, child_rot):
    inv_xyz, inv_rot = invert_transform(parent_xyz, parent_rot)
    return compose_transform(inv_xyz, inv_rot, child_xyz, child_rot)


def strip_world_link(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    world_transform = None
    world_joints = []

    for joint in root.findall("joint"):
        parent = joint.find("parent")
        if parent is not None and parent.get("link") == "world":
            world_joints.append(joint)
            origin = joint.find("origin")
            if world_transform is None and origin is not None:
                xyz_str = origin.get("xyz", "0 0 0")
                rpy_str = origin.get("rpy", "0 0 0")
                xyz = [float(v) for v in xyz_str.split()]
                rpy = [float(v) for v in rpy_str.split()]
                world_transform = (xyz, rpy)

    removed = False

    for link in root.findall("link"):
        if link.get("name") == "world":
            root.remove(link)
            removed = True
            break

    joints_to_remove = list(world_joints)
    if len(world_joints) > 1:
        primary = world_joints[0]
        primary_child = primary.find("child").get("link")
        primary_xyz, primary_rot = transform_from_origin(primary.find("origin"))

        for joint in world_joints[1:]:
            child = joint.find("child")
            if child is None:
                continue
            child_name = child.get("link")
            child_xyz, child_rot = transform_from_origin(joint.find("origin"))
            rel_xyz, rel_rot = relative_transform(primary_xyz, primary_rot, child_xyz, child_rot)
            rel_rpy = matrix_to_rpy(rel_rot)

            new_joint = ET.Element("joint", {
                "name": f"{primary_child}_to_{child_name}_world_fixed",
                "type": "fixed",
            })
            ET.SubElement(new_joint, "parent", {"link": primary_child})
            ET.SubElement(new_joint, "child", {"link": child_name})
            ET.SubElement(new_joint, "origin", {
                "xyz": " ".join(f"{v:.9f}" for v in rel_xyz),
                "rpy": " ".join(f"{v:.9f}" for v in rel_rpy),
            })
            root.append(new_joint)

    for j in joints_to_remove:
        root.remove(j)
        removed = True

    if not removed:
        return None, None

    orig = Path(urdf_path)
    clean_path = orig.parent / f"{orig.stem}_no_world.urdf"
    tree.write(str(clean_path), encoding="unicode")
    return str(clean_path), world_transform


def sanitize_name(name):
    return name.replace(" ", "_").replace("-", "_")


def average_coefficients(coeff_list):
    valid = [c for c in coeff_list if c is not None]
    if not valid:
        return None
    return {
        "kinetic_friction_coefficient": sum(c["kinetic_friction_coefficient"] for c in valid) / len(valid),
        "static_friction_coefficient": sum(c["static_friction_coefficient"] for c in valid) / len(valid),
    }


def build_target_coefficients(batch_name, components):
    source_names = TARGET_BINDING_OVERRIDES.get(batch_name, {})
    result = {}
    for link_name, json_names in source_names.items():
        coeff_list = []
        for jn in json_names:
            if jn in components:
                coeff_list.append(components[jn])
        avg = average_coefficients(coeff_list)
        if avg:
            result[link_name] = avg
    for key, coeffs in components.items():
        if key not in result:
            result[key] = coeffs
    return result


def apply_friction_materials(stage, item_components, batch_name):
    if batch_name not in item_components:
        return 0, []

    components = item_components[batch_name]
    target_coeffs = build_target_coefficients(batch_name, components)

    root_prim = stage.GetDefaultPrim()
    if not root_prim:
        print("  NO ROOT PRIM")
        return 0, 0
    root_path = root_prim.GetPath()

    looks_path = root_path.AppendChild("Looks")
    if not stage.GetPrimAtPath(looks_path):
        UsdGeom.Scope.Define(stage, looks_path)

    bound_count = 0
    missing_paths = []

    for prim in stage.Traverse():
        path = prim.GetPath()
        if prim.GetTypeName() != "Xform":
            continue
        if str(path).startswith(str(looks_path)):
            continue

        has_rigid = False
        try:
            apis = prim.GetAppliedSchemas() or []
            has_rigid = "PhysicsRigidBodyAPI" in apis
        except Exception:
            pass
        if not has_rigid:
            continue
        if not _has_collision_descendant(prim):
            continue

        link_name = prim.GetName()
        coeffs = target_coeffs.get(link_name)

        if coeffs is None:
            missing_paths.append(str(path))
            continue

        material_name = f"{sanitize_name(link_name)}_PhysicsMaterial"
        material_path = looks_path.AppendChild(material_name)

        if stage.GetPrimAtPath(material_path):
            stage.RemovePrim(material_path)

        material = UsdShade.Material.Define(stage, material_path)
        material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        material_api.CreateStaticFrictionAttr().Set(float(coeffs["static_friction_coefficient"]))
        material_api.CreateDynamicFrictionAttr().Set(float(coeffs["kinetic_friction_coefficient"]))

        binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
        binding_api.Bind(
            material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            materialPurpose="physics",
        )
        bound_count += 1

    return bound_count, missing_paths


def apply_root_pose(stage, world_transform, batch_name):
    xyz, rpy = world_transform or ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    art_root = None
    for prim in stage.Traverse():
        try:
            apis = prim.GetAppliedSchemas() or []
        except Exception:
            continue
        if "PhysicsArticulationRootAPI" in apis:
            art_root = prim
            break

    # Single fixed-link imports may not expose PhysicsArticulationRootAPI.
    # The default prim is still the correct root for the stripped world pose.
    single_link_import = art_root is None
    if single_link_import:
        art_root = stage.GetDefaultPrim()
        if not art_root:
            return 0

    parent = art_root.GetParent()
    target = parent if parent and parent.GetPath() != Sdf.Path("/") else art_root

    # Keep the stage/default prim in the world frame. Isaac Sim may create new
    # objects under it, so rotating it would also rotate objects such as a ground plane.
    if single_link_import:
        for child in art_root.GetChildren():
            if any(prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in Usd.PrimRange(child)):
                target = child
                break

    xformable = UsdGeom.Xformable(target)
    xformable.ClearXformOpOrder()

    if single_link_import and batch_name in IGNORE_WORLD_POSE_SINGLE_LINK_BATCHES:
        xyz, rpy = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    applied = 0
    translate_op = None
    if any(abs(v) > 1e-6 for v in xyz):
        translate_op = xformable.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(*xyz))
        applied = 1
    if any(abs(v) > 1e-6 for v in rpy):
        xformable.AddRotateXYZOp().Set(Gf.Vec3f(
            math.degrees(rpy[0]), math.degrees(rpy[1]), math.degrees(rpy[2])
        ))
        applied = 1

    if single_link_import:
        bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"])
        min_z = bbox.ComputeWorldBound(art_root).ComputeAlignedRange().GetMin()[2]
        if abs(min_z) > 1e-6:
            if translate_op is None:
                translate_op = xformable.AddTranslateOp()
                translate_op.Set(Gf.Vec3d(0.0, 0.0, -min_z))
            else:
                translate_op.Set(translate_op.Get() + Gf.Vec3d(0.0, 0.0, -min_z))
            applied = 1

    return applied


def remove_drives_from_stage(stage):
    count = 0
    for prim in stage.Traverse():
        apis = []
        try:
            apis = prim.GetAppliedSchemas() or []
        except Exception:
            pass
        if "PhysicsDriveAPI:angular" in apis:
            prim.RemoveAPI(UsdPhysics.DriveAPI, "angular")
            count += 1
        if "PhysicsDriveAPI:linear" in apis:
            prim.RemoveAPI(UsdPhysics.DriveAPI, "linear")
            count += 1
    return count


def apply_joint_smoothing(usd_dest_path):
    robot_name = Path(usd_dest_path).stem
    sim_ready = Path(usd_dest_path).parent
    phys_config = sim_ready / "configuration" / f"{robot_name}_physics.usd"

    total = 0

    stage = Usd.Stage.Open(usd_dest_path)
    if stage:
        total += remove_drives_from_stage(stage)
        stage.Save()

    if phys_config.exists():
        try:
            phys_stage = Usd.Stage.Open(str(phys_config))
            if phys_stage:
                n = remove_drives_from_stage(phys_stage)
                if n > 0:
                    phys_stage.Save()
                total += n
        except Exception as e:
            print(f"  WARNING: phys cfg: {e}")

    return total


def has_nonworld_fixed_joint(urdf_path):
    root = ET.parse(urdf_path).getroot()
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        if joint.get("type") == "fixed" and parent is not None and parent.get("link") != "world":
            return True
    return False


def _iter_subtree_with_proxies(prim):
    """Read referenced importer geometry without attempting to edit instance proxies."""
    stack = [prim]
    predicate = Usd.TraverseInstanceProxies()
    while stack:
        current = stack.pop()
        yield current
        stack.extend(current.GetFilteredChildren(predicate))


def _collision_prims(prim, container=None):
    result = []
    for current in _iter_subtree_with_proxies(prim):
        path = str(current.GetPath())
        if container and f"/{container}/" not in path:
            continue
        if "PhysicsCollisionAPI" in (current.GetAppliedSchemas() or []):
            result.append(current)
    return sorted(result, key=lambda item: str(item.GetPath()))


def _has_collision_descendant(prim):
    return bool(_collision_prims(prim))


def validate_vhacd_colliders(stage, manifest_path):
    """Validate importer-authored VHACD colliders; never create a fallback mesh."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    expected = {item["link"]: item["hulls"] for item in manifest.get("meshes", [])}
    rigid_by_name = {
        prim.GetName(): prim for prim in stage.Traverse()
        if "PhysicsRigidBodyAPI" in (prim.GetAppliedSchemas() or [])
    }
    errors = []
    actual = {}
    for link_name, hull_count in expected.items():
        rigid = rigid_by_name.get(link_name)
        if rigid is None:
            errors.append(f"missing rigid body: {link_name}")
            continue
        colliders = _collision_prims(rigid, "collisions")
        actual[link_name] = len(colliders)
        if len(colliders) != hull_count:
            errors.append(f"{link_name}: expected {hull_count} collisions, found {len(colliders)}")
        for collider in colliders:
            schemas = collider.GetAppliedSchemas() or []
            enabled = collider.GetAttribute("physics:collisionEnabled")
            approximation = collider.GetAttribute("physics:approximation")
            if "PhysicsMeshCollisionAPI" not in schemas:
                errors.append(f"{collider.GetPath()}: missing PhysicsMeshCollisionAPI")
            elif not enabled or enabled.Get() is not True:
                errors.append(f"{collider.GetPath()}: collision is not enabled")
            elif not approximation or approximation.Get() != UsdPhysics.Tokens.convexHull:
                errors.append(f"{collider.GetPath()}: approximation is not convexHull")
    return {"expected": expected, "actual": actual, "errors": errors}


def _first_mesh_descendant(prim):
    for current in _iter_subtree_with_proxies(prim):
        if current.IsA(UsdGeom.Mesh):
            return current
    return None


def _mesh_signature(mesh_prim, rigid_prim, xform_cache):
    mesh = UsdGeom.Mesh(mesh_prim)
    points = mesh.GetPointsAttr().Get() or []
    counts = mesh.GetFaceVertexCountsAttr().Get() or []
    indices = mesh.GetFaceVertexIndicesAttr().Get() or []
    if not points or not counts or not indices:
        return None
    rigid_inverse = xform_cache.GetLocalToWorldTransform(rigid_prim).GetInverse()
    mesh_to_rigid = xform_cache.GetLocalToWorldTransform(mesh_prim) * rigid_inverse
    transformed = [mesh_to_rigid.Transform(point) for point in points]
    return (
        tuple((round(point[0], 6), round(point[1], 6), round(point[2], 6)) for point in transformed),
        tuple(counts),
        tuple(indices),
    )


def migrate_legacy_vhacd_colliders(stage, manifest_path):
    """Remove only duplicate repair hulls after proving importer hulls are identical."""
    validation = validate_vhacd_colliders(stage, manifest_path)
    report = {
        "manifest_hulls": sum(validation["expected"].values()),
        "standard_collision_hulls": sum(validation["actual"].values()),
        "legacy_repair_hulls": 0,
        "geometry_matched": False,
        "removed_colliders": 0,
        "remaining_colliders": 0,
        "schema_errors": validation["errors"],
        "status": "validation_failed" if validation["errors"] else "already_clean",
    }
    if validation["errors"]:
        return report

    rigid_by_name = {
        prim.GetName(): prim for prim in stage.Traverse()
        if "PhysicsRigidBodyAPI" in (prim.GetAppliedSchemas() or [])
    }
    removals = []
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for link_name, expected_hulls in validation["expected"].items():
        rigid = rigid_by_name[link_name]
        legacy_root = stage.GetPrimAtPath(rigid.GetPath().AppendChild("colliders"))
        if not legacy_root:
            continue
        if legacy_root.IsInstanceProxy() or legacy_root.GetPrimStack()[0].layer != stage.GetRootLayer():
            report["schema_errors"].append(f"{legacy_root.GetPath()}: not authored in root layer")
            continue
        legacy_hulls = list(legacy_root.GetChildren())
        report["legacy_repair_hulls"] += len(legacy_hulls)
        standard_hulls = _collision_prims(rigid, "collisions")
        if len(legacy_hulls) != expected_hulls:
            report["schema_errors"].append(
                f"{link_name}: expected {expected_hulls} legacy hulls, found {len(legacy_hulls)}"
            )
            continue
        for standard, legacy in zip(standard_hulls, legacy_hulls):
            standard_mesh = _first_mesh_descendant(standard)
            legacy_mesh = _first_mesh_descendant(legacy)
            if not standard_mesh or not legacy_mesh:
                report["schema_errors"].append(f"{link_name}: missing mesh for duplicate comparison")
                break
            if _mesh_signature(standard_mesh, rigid, xform_cache) != _mesh_signature(legacy_mesh, rigid, xform_cache):
                report["schema_errors"].append(f"{link_name}: standard and legacy hull geometry differ")
                break
        else:
            removals.append(legacy_root.GetPath())

    if report["schema_errors"]:
        report["status"] = "comparison_failed"
        report["remaining_colliders"] = report["legacy_repair_hulls"]
        return report

    report["geometry_matched"] = bool(removals)
    for path in removals:
        stage.RemovePrim(path)
    report["removed_colliders"] = report["legacy_repair_hulls"]
    report["status"] = "migrated" if removals else "already_clean"
    return report


def apply_rebound_controls_to_stage(stage, batch_name):
    controls = REBOUND_JOINT_CONTROLS.get(batch_name)
    if not controls:
        return 0

    count = 0
    for prim in stage.Traverse():
        control = controls.get(prim.GetName())
        if control is None:
            continue
        target_position = control.get("target_position", 0.0)
        if control.get("target_from_limit") == "farther_from_zero":
            if control["drive_type"] == "angular":
                joint = UsdPhysics.RevoluteJoint(prim)
            else:
                joint = UsdPhysics.PrismaticJoint(prim)
            lower = joint.GetLowerLimitAttr().Get()
            upper = joint.GetUpperLimitAttr().Get()
            if lower is not None and upper is not None:
                target_position = lower if abs(float(lower)) >= abs(float(upper)) else upper
        elif control.get("target_from_limit") in ("lower", "upper"):
            if control["drive_type"] == "angular":
                joint = UsdPhysics.RevoluteJoint(prim)
            else:
                joint = UsdPhysics.PrismaticJoint(prim)
            if control["target_from_limit"] == "lower":
                limit = joint.GetLowerLimitAttr().Get()
            else:
                limit = joint.GetUpperLimitAttr().Get()
            if limit is not None:
                target_position = float(limit)
        elif "target_from_local_pos0_axis" in control:
            joint = UsdPhysics.Joint(prim)
            local_pos0 = joint.GetLocalPos0Attr().Get()
            if local_pos0 is not None:
                scale = float(control.get("target_scale", 1.0))
                target_position = float(local_pos0[int(control["target_from_local_pos0_axis"])]) * scale
        drive = UsdPhysics.DriveAPI.Apply(prim, control["drive_type"])
        drive.CreateTargetPositionAttr().Set(float(target_position))
        drive.CreateStiffnessAttr().Set(float(control["stiffness"]))
        drive.CreateDampingAttr().Set(float(control["damping"]))
        drive.CreateMaxForceAttr().Set(float(control["max_force"]))
        if "lower_limit" in control or "upper_limit" in control:
            joint = UsdPhysics.PrismaticJoint(prim)
            if "lower_limit" in control:
                joint.CreateLowerLimitAttr().Set(float(control["lower_limit"]))
            if "upper_limit" in control:
                joint.CreateUpperLimitAttr().Set(float(control["upper_limit"]))
        count += 1
    return count


def apply_rebound_controls(usd_dest_path, batch_name):
    robot_name = Path(usd_dest_path).stem
    sim_ready = Path(usd_dest_path).parent
    phys_config = sim_ready / "configuration" / f"{robot_name}_physics.usd"

    total = 0
    for path in (Path(usd_dest_path), phys_config):
        if not path.exists():
            continue
        stage = Usd.Stage.Open(str(path))
        if not stage:
            continue
        count = apply_rebound_controls_to_stage(stage, batch_name)
        if count:
            stage.Save()
            total += count
    return total


def remove_drive_properties(prim, drive_type):
    prefix = f"drive:{drive_type}:"
    removed = 0
    for attr in list(prim.GetAttributes()):
        if attr.GetName().startswith(prefix):
            attr.Block()
            removed += 1
    if f"PhysicsDriveAPI:{drive_type}" in (prim.GetAppliedSchemas() or []):
        prim.RemoveAPI(UsdPhysics.DriveAPI, drive_type)
        removed += 1
    return removed


def apply_assembly_locks_to_stage(stage, batch_name):
    locks = ASSEMBLY_PRISMATIC_LOCKS.get(batch_name)
    if not locks:
        return 0

    count = 0
    for prim in stage.Traverse():
        lock_position = locks.get(prim.GetName())
        if lock_position is None:
            continue
        remove_drive_properties(prim, "linear")
        prim.CreateAttribute("physics:lowerLimit", Sdf.ValueTypeNames.Float).Set(float(lock_position))
        prim.CreateAttribute("physics:upperLimit", Sdf.ValueTypeNames.Float).Set(float(lock_position))
        prim.CreateAttribute("state:linear:physics:position", Sdf.ValueTypeNames.Float).Set(float(lock_position))
        prim.CreateAttribute("state:linear:physics:velocity", Sdf.ValueTypeNames.Float).Set(0.0)
        count += 1
    return count


def apply_assembly_locks(usd_dest_path, batch_name):
    robot_name = Path(usd_dest_path).stem
    sim_ready = Path(usd_dest_path).parent
    phys_config = sim_ready / "configuration" / f"{robot_name}_physics.usd"

    total = 0
    for path in (Path(usd_dest_path), phys_config):
        if not path.exists():
            continue
        stage = Usd.Stage.Open(str(path))
        if not stage:
            continue
        stage.SetEditTarget(stage.GetRootLayer())
        count = apply_assembly_locks_to_stage(stage, batch_name)
        if count:
            stage.Save()
            total += count
    return total


def apply_fixed_joint_limits_to_stage(stage, batch_name):
    joints = FIXED_JOINT_LIMITS.get(batch_name)
    if not joints:
        return 0

    count = 0
    for prim in stage.Traverse():
        if prim.GetName() not in joints:
            continue
        remove_drive_properties(prim, "linear")
        remove_drive_properties(prim, "angular")
        prim.CreateAttribute("physics:lowerLimit", Sdf.ValueTypeNames.Float).Set(0.0)
        prim.CreateAttribute("physics:upperLimit", Sdf.ValueTypeNames.Float).Set(0.0)
        prim.CreateAttribute("state:linear:physics:position", Sdf.ValueTypeNames.Float).Set(0.0)
        prim.CreateAttribute("state:linear:physics:velocity", Sdf.ValueTypeNames.Float).Set(0.0)
        prim.CreateAttribute("state:angular:physics:position", Sdf.ValueTypeNames.Float).Set(0.0)
        prim.CreateAttribute("state:angular:physics:velocity", Sdf.ValueTypeNames.Float).Set(0.0)
        count += 1
    return count


def apply_fixed_joint_limits(usd_dest_path, batch_name):
    robot_name = Path(usd_dest_path).stem
    sim_ready = Path(usd_dest_path).parent
    phys_config = sim_ready / "configuration" / f"{robot_name}_physics.usd"

    total = 0
    for path in (Path(usd_dest_path), phys_config):
        if not path.exists():
            continue
        stage = Usd.Stage.Open(str(path))
        if not stage:
            continue
        stage.SetEditTarget(stage.GetRootLayer())
        count = apply_fixed_joint_limits_to_stage(stage, batch_name)
        if count:
            stage.Save()
            total += count
    return total


def repair_existing_vhacd_outputs(item_components):
    """Remove verified duplicate repair hulls without re-importing or VHACD."""
    only = os.environ.get("BATCH_CONVERT_ONLY")
    only_batches = {x.strip() for x in only.split(",") if x.strip()} if only else None
    rows = []
    for batch_dir in sorted(ASSET_ROOT.glob("batch_*")):
        if not batch_dir.is_dir() or (only_batches and batch_dir.name not in only_batches):
            continue
        out_dir = batch_dir / FINAL_ASSET_DIRNAME
        if not out_dir.is_dir():
            continue
        for usd_path in sorted(out_dir.glob("*.usd")):
            robot = usd_path.stem
            urdf = out_dir / f"{robot}.urdf"
            config = out_dir / "configuration"
            base = config / f"{robot}_base.usd"
            physics = config / f"{robot}_physics.usd"
            ref_report = repair_usd_references(base, physics, urdf) if base.exists() else {"status": "missing_base"}
            stage = Usd.Stage.Open(str(usd_path))
            if not stage:
                rows.append({"batch": batch_dir.name, "usd": str(usd_path), "status": "stage_open_failed", "reference": ref_report})
                continue
            manifest_path = out_dir / "vhacd_manifest.json"
            if not manifest_path.exists():
                rows.append({"batch": batch_dir.name, "usd": str(usd_path), "status": "missing_manifest", "reference": ref_report})
                continue
            collision = migrate_legacy_vhacd_colliders(stage, manifest_path)
            if collision["status"] in {"validation_failed", "comparison_failed"}:
                rows.append({"batch": batch_dir.name, "usd": str(usd_path), "status": collision["status"],
                             "reference": ref_report, "collision": collision})
                continue
            bound, missing = apply_friction_materials(stage, item_components, batch_dir.name)
            stage.Save()
            rows.append({"batch": batch_dir.name, "usd": str(usd_path), "status": "ok" if ref_report.get("status") != "error" else "reference_error",
                         "reference": ref_report, "collision": collision,
                         "friction": {"bound": bound, "missing": len(missing), "missing_paths": missing}})
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "existing_vhacd_repair_report.json").write_text(
        json.dumps({"assets": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 1 if any(row["status"] != "ok" for row in rows) else 0


def main():
    item_components = load_item_components()
    if os.environ.get("BATCH_CONVERT_REPAIR_EXISTING", "0") == "1":
        code = repair_existing_vhacd_outputs(item_components)
        if app:
            app.close()
        return code
    modular_bindings, modular_pipeline = load_modular_bindings()
    tasks = discover_urdfs()
    total = len(tasks)
    print(f"Found {total} URDF files\n")

    converted = 0
    total_bound = 0
    total_missing = 0
    total_joints = 0
    modular_bound = 0
    modular_failed = 0
    failed_batches = set()
    records = []

    for i, (urdf_path, dest_path, batch_name) in enumerate(tasks):
        robot_name = Path(urdf_path).stem
        print(f"[{i+1}/{total}] {batch_name} / {robot_name}")

        vhacd_mode = os.environ.get("BATCH_CONVERT_VHACD", "1") != "0"
        working_urdf = stage_urdf_source(urdf_path, Path(dest_path).parent)
        if vhacd_mode:
            concept_json = Path(urdf_path).parent.parent / "conceptualization" / f"{Path(urdf_path).stem}.json"
            if not concept_json.exists():
                concept_json = Path(urdf_path).parent.parent / "conceptualization" / f"{batch_name}.json"
            try:
                import_path, vhacd_manifest = prepare_vhacd_urdf(
                    working_urdf, Path(dest_path).parent, concept_json
                )
            except Exception as exc:
                print(f"  ERROR: VHACD failed: {exc}")
                records.append({
                    "batch": batch_name,
                    "urdf": urdf_path,
                    "usd": dest_path,
                    "status": "vhacd_failed",
                    "error": str(exc),
                })
                failed_batches.add(batch_name)
                continue
        else:
            import_path = working_urdf
            vhacd_manifest = None
        clean_urdf, world_transform = strip_world_link(import_path)
        import_path = clean_urdf if clean_urdf else import_path

        status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
        # VHACD collider links must retain stable USD prim paths across the base/physics layers.
        keep_fixed_joints = vhacd_mode
        import_config.set_merge_fixed_joints(not keep_fixed_joints)
        import_config.set_import_inertia_tensor(True)
        import_config.set_fix_base(False)
        import_config.set_convex_decomp(False if vhacd_mode else True)
        import_config.set_collision_from_visuals(False if vhacd_mode else True)
        import_config.set_distance_scale(1.0)
        import_config.set_make_default_prim(True)
        import_config.set_create_physics_scene(True)
        import_config.set_density(0.0)

        status, prim_path = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=import_path,
            import_config=import_config,
            dest_path=dest_path,
            get_articulation_root=True,
        )

        if not status:
            print(f"  ERROR: import failed")
            records.append({"batch": batch_name, "urdf": urdf_path, "usd": dest_path, "status": "import_failed"})
            failed_batches.add(batch_name)
            continue

        robot_name = Path(dest_path).stem
        config_dir = Path(dest_path).parent / "configuration"
        reference_report = repair_usd_references(
            config_dir / f"{robot_name}_base.usd",
            config_dir / f"{robot_name}_physics.usd",
            import_path,
        )
        if reference_report.get("status") == "error":
            print(f"  ERROR: unresolved USD references: {reference_report['errors']}")
            failed_batches.add(batch_name)
            records.append({"batch": batch_name, "urdf": urdf_path, "usd": dest_path,
                            "status": "reference_repair_failed", "reference": reference_report})
            continue

        stage = Usd.Stage.Open(dest_path)
        if not stage:
            print(f"  ERROR: could not open {dest_path}")
            records.append({"batch": batch_name, "urdf": urdf_path, "usd": dest_path, "status": "stage_open_failed"})
            failed_batches.add(batch_name)
            continue

        pose_fixed = 0
        if clean_urdf:
            pose_fixed = apply_root_pose(stage, world_transform, batch_name)

        collision_validation = (validate_vhacd_colliders(
            stage, Path(dest_path).parent / "vhacd_manifest.json"
        ) if vhacd_mode else {"expected": {}, "actual": {}, "errors": []})
        if collision_validation["errors"]:
            print(f"  ERROR: VHACD collider validation failed: {collision_validation['errors']}")
            failed_batches.add(batch_name)
            records.append({"batch": batch_name, "urdf": urdf_path, "usd": dest_path,
                            "status": "collision_validation_failed", "collision": collision_validation})
            continue

        bound, missing = apply_friction_materials(stage, item_components, batch_name)
        total_bound += bound
        total_missing += len(missing)
        stage.Save()

        if clean_urdf:
            try:
                Path(clean_urdf).unlink()
            except OSError:
                pass

        joint_count = apply_joint_smoothing(dest_path)
        total_joints += joint_count
        lock_count = apply_assembly_locks(dest_path, batch_name)
        fixed_count = apply_fixed_joint_limits(dest_path, batch_name)

        modular_report = None
        if batch_name in modular_bindings:
            try:
                modular_report = package_modular_asset(dest_path, batch_name, modular_bindings[batch_name])
                modular_bound += 1
            except Exception as exc:
                modular_failed += 1
                print(f"  ERROR: modular behavior package failed: {exc}")

        records.append({
            "batch": batch_name,
            "urdf": urdf_path,
            "usd": dest_path,
            "status": "ok" if modular_report is not None or batch_name not in modular_bindings else "modular_failed",
            "friction": {"bound": bound, "missing": len(missing), "missing_paths": missing},
            "collision_validation": collision_validation,
            "reference_repair": reference_report,
            "merge_fixed_joints": not keep_fixed_joints,
            "vhacd_manifest": str(Path(dest_path).parent / "vhacd_manifest.json") if vhacd_manifest else None,
            "vhacd_hulls": sum(item["hulls"] for item in vhacd_manifest["meshes"]) if vhacd_manifest else None,
            "modular": modular_report or {"status": "not_selected"},
        })

        converted += 1
        tags = []
        if clean_urdf:
            tags.append("no-world")
        if pose_fixed:
            tags.append("pose-fixed")
        if lock_count:
            tags.append(f"locked-assembly={lock_count}")
        if fixed_count:
            tags.append(f"fixed-joints={fixed_count}")
        if modular_report:
            tags.append(f"modular={len(modular_report['modules'])}")
        status_str = f" [{' '.join(tags)}]" if tags else ""
        print(f"  OK{status_str} | bound={bound} missing={len(missing)} joints={joint_count}" + (f" hulls={sum(x['hulls'] for x in vhacd_manifest['meshes'])}" if vhacd_manifest else ""))

    print(f"\n=== SUMMARY ===")
    print(f"Converted: {converted}/{total}")
    print(f"Materials bound: {total_bound}")
    print(f"Missing bindings: {total_missing}")
    print(f"Joints smoothed: {total_joints}")
    print(f"Modular behaviors bound: {modular_bound}")
    print(f"Modular behavior failures: {modular_failed}")

    grasp_code = 0
    if modular_pipeline.get("attach_grasp_poses", True):
        output_dirname = FINAL_ASSET_DIRNAME
        grasp_code = run_grasp_for_batches(discover_existing_outputs(output_dirname, failed_batches), output_dirname)
        print(f"Grasp attachment exit code: {grasp_code}")
    else:
        output_dirname = FINAL_ASSET_DIRNAME
    write_batch_report(records, grasp_code, output_dirname)
    if app:
        app.close()
    if modular_failed or grasp_code:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
