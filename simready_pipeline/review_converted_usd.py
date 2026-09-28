import json
import os
import sys
from pathlib import Path

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

from pxr import Usd, UsdPhysics, UsdShade
from usd_reference_repair import repair_usd_references


ROOT = Path(__file__).resolve().parent
REVIEWER_BASE = Path(os.environ.get("REVIEWER_BASE", ROOT)).resolve()
TARGETS = set(json.loads((ROOT / "config" / "target_batches.json").read_text(encoding="utf-8")))
SIM_READY_DIRNAME = os.environ.get("REVIEWER_SIM_READY_DIRNAME", "sim_ready")
REPORT_DIR = Path(os.environ.get("PIPELINE_REPORT_DIR", REVIEWER_BASE / ".pipeline_reports")).resolve()
ALLOWED_DRIVEN_JOINTS = {
    "batch_01_0035_Lighter_001_chofn": {"Cambered_Body_to_L_Shaped_Button"},
    "batch_02_0003_Doorhandle_001_Bathroom_Door_Handle": {
        "Regular_handle_base_0_to_Regular_handle_grip_0",
        "Regular_handle_base_1_to_Regular_handle_grip_1",
    },
    "batch_01_0031_Clip_001_plastic_hanging": {
        "Regular_lever_to_Regular_lever_handle_left",
        "Regular_lever_to_Regular_lever_handle_right",
    },
    "batch_02_0020_Clip_001_Large_Laundry_Clips": {
        "Regular_lever_to_Regular_lever_handle_left",
        "Regular_lever_to_Regular_lever_handle_right",
    },
    "batch_02_0021_Clip_002_Yellow_Hair_Clips": {
        "Regular_lever_to_Regular_lever_handle_left",
        "Regular_lever_to_Regular_lever_handle_right",
    },
    "batch_01_0027_Shampoo_001_lux_shower_gel": {
        "Cylindrical_body_to_Regular_nozzle_virtual_prismatic",
        "Regular_nozzle_to_Regular_nozzle_Head_virtual_prismatic",
        "Regular_nozzle_virtual_prismatic_to_Regular_nozzle",
        "Regular_nozzle_Head_virtual_prismatic_to_Regular_nozzle_Head",
    },
    "batch_01_0032_Sanitizer_001_liquid_soap": {
        "Cylindrical_body_to_Regular_nozzle_virtual_prismatic",
        "Regular_nozzle_to_Regular_nozzle_Head_virtual_prismatic",
        "Regular_nozzle_virtual_prismatic_to_Regular_nozzle",
        "Regular_nozzle_Head_virtual_prismatic_to_Regular_nozzle_Head",
    },
    "batch_02_0007_Shampoo_001_Enchanteur_Body_Wash": {
        "Cylindrical_body_to_Regular_nozzle_virtual_prismatic",
        "Regular_nozzle_to_Regular_nozzle_Head_virtual_prismatic",
        "Regular_nozzle_virtual_prismatic_to_Regular_nozzle",
        "Regular_nozzle_Head_virtual_prismatic_to_Regular_nozzle_Head",
    },
    "batch_02_0008_Shampoo_002_Dettol_Hand_Wash": {
        "Cylindrical_body_to_Regular_nozzle_virtual_prismatic",
        "Regular_nozzle_to_Regular_nozzle_Head_virtual_prismatic",
        "Regular_nozzle_virtual_prismatic_to_Regular_nozzle",
        "Regular_nozzle_Head_virtual_prismatic_to_Regular_nozzle_Head",
    },
    "batch_02_0009_Shampoo_003_Dettol_Foam_Hand_Wash": {
        "Cylindrical_body_to_Regular_nozzle_virtual_prismatic",
        "Regular_nozzle_to_Regular_nozzle_Head_virtual_prismatic",
        "Regular_nozzle_virtual_prismatic_to_Regular_nozzle",
        "Regular_nozzle_Head_virtual_prismatic_to_Regular_nozzle_Head",
    },
}


def report_path(path):
    try:
        return str(Path(path).resolve().relative_to(REVIEWER_BASE)).replace("\\", "/")
    except (OSError, ValueError):
        return Path(path).name


def report_value(value):
    if isinstance(value, dict):
        return {key: report_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [report_value(item) for item in value]
    if isinstance(value, str) and ("/" in value or "\\" in value):
        return report_path(value)
    return value

def usd_files():
    only_batches = None
    if os.environ.get("BATCH_CONVERT_ONLY"):
        only_batches = {x.strip() for x in os.environ["BATCH_CONVERT_ONLY"].split(",") if x.strip()}

    seen = set()

    def maybe_yield_batch_dir(batch_dir):
        if not batch_dir.is_dir() or not batch_dir.name.startswith("batch_"):
            return
        if only_batches is not None and batch_dir.name not in only_batches:
            return
        if only_batches is None and batch_dir.name not in TARGETS:
            return
        sim_ready = batch_dir / SIM_READY_DIRNAME
        if not sim_ready.is_dir():
            return
        for usd_path in sorted(sim_ready.glob("*.usd")):
            key = (batch_dir.name, str(usd_path.resolve()))
            if key in seen:
                continue
            seen.add(key)
            yield batch_dir.name, usd_path

    reviewer_base = REVIEWER_BASE
    if reviewer_base.is_dir():
        for batch_dir in sorted(reviewer_base.iterdir()):
            yield from maybe_yield_batch_dir(batch_dir)
        return

    for batch_dir in sorted(ROOT.iterdir()):
        yield from maybe_yield_batch_dir(batch_dir)


    for parent in sorted(ROOT.iterdir()):
        if not parent.is_dir() or parent.name.startswith(".") or parent.name.startswith("batch_"):
            continue
        if parent.name in {"classes", "template_rich_knowledge", "__pycache__"}:
            continue
        for batch_dir in sorted(parent.iterdir()):
            yield from maybe_yield_batch_dir(batch_dir)


def has_schema(prim, schema):
    try:
        return schema in (prim.GetAppliedSchemas() or [])
    except Exception:
        return False


def has_collision(prim):
    return has_schema(prim, "PhysicsCollisionAPI")


def iter_subtree_with_proxies(prim):
    stack = [prim]
    predicate = Usd.TraverseInstanceProxies()
    while stack:
        current = stack.pop()
        yield current
        stack.extend(current.GetFilteredChildren(predicate))


def has_collision_descendant(prim):
    for current in iter_subtree_with_proxies(prim):
        if has_collision(current):
            return True
    return False


def has_physics_material_binding(prim):
    for current in iter_subtree_with_proxies(prim):
        try:
            binding = UsdShade.MaterialBindingAPI(current).GetDirectBinding(materialPurpose="physics")
            material_path = binding.GetMaterialPath()
            if material_path and not material_path.isEmpty:
                return True
        except Exception:
            pass
    return False


def collider_summary(stage, manifest_path):
    expected = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {item["link"]: item["hulls"] for item in manifest.get("meshes", [])}

    standard = []
    legacy = []
    mesh_schema_missing = []
    wrong_approximation = []
    actual_by_link = {}
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if not has_collision(prim):
            continue
        path = str(prim.GetPath())
        if "/collisions/" in path:
            standard.append(path)
            link_name = next((name for name in expected if f"/{name}/collisions/" in path), None)
            if link_name:
                actual_by_link[link_name] = actual_by_link.get(link_name, 0) + 1
        elif "/colliders/" in path:
            legacy.append(path)
        if expected:
            schemas = prim.GetAppliedSchemas() or []
            if "PhysicsMeshCollisionAPI" not in schemas:
                mesh_schema_missing.append(path)
            approximation = prim.GetAttribute("physics:approximation")
            if not approximation or approximation.Get() != UsdPhysics.Tokens.convexHull:
                wrong_approximation.append(path)

    missing_by_link = {
        name: count - actual_by_link.get(name, 0)
        for name, count in expected.items() if actual_by_link.get(name, 0) != count
    }
    return {
        "standard_collisions": len(standard),
        "legacy_colliders": len(legacy),
        "missing_mesh_collision_api": mesh_schema_missing,
        "wrong_approximation": wrong_approximation,
        "manifest_expected_hulls": sum(expected.values()),
        "manifest_actual_by_link": actual_by_link,
        "manifest_mismatches": missing_by_link,
    }


def review_stage(batch_name, usd_path):
    config = usd_path.parent / "configuration"
    base = config / f"{usd_path.stem}_base.usd"
    physics = config / f"{usd_path.stem}_physics.usd"
    reference_report = (repair_usd_references(base, physics, remove=False)
                        if base.exists() else {"unresolved": [], "errors": ["missing base layer"]})
    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        return {"batch": batch_name, "usd": report_path(usd_path), "error": "cannot open stage"}

    rigid = []
    joints = []
    collision_rigid = []
    friction_bound = 0
    friction_missing_paths = []
    grasps = 0
    fixed_to_world = 0
    locked_or_driven = 0

    for prim in stage.Traverse():
        if has_schema(prim, "PhysicsRigidBodyAPI"):
            rigid.append(str(prim.GetPath()))
            owns_collision = has_collision_descendant(prim)
            if owns_collision:
                collision_rigid.append(str(prim.GetPath()))
            if owns_collision and has_physics_material_binding(prim):
                friction_bound += 1
            elif owns_collision:
                friction_missing_paths.append(str(prim.GetPath()))
        if prim.GetName().lower().startswith("grasp_") or "/grasps/" in str(prim.GetPath()):
            if prim.HasAttribute("grasp:pose_matrix") or prim.HasAttribute("grasp:approach"):
                grasps += 1
        if prim.IsA(UsdPhysics.Joint):
            joints.append(str(prim.GetPath()))
            body0 = UsdPhysics.Joint(prim).GetBody0Rel().GetTargets()
            body1 = UsdPhysics.Joint(prim).GetBody1Rel().GetTargets()
            if not body0 or not body1:
                fixed_to_world += 1
            allowed = prim.GetName() in ALLOWED_DRIVEN_JOINTS.get(batch_name, set())
            if not allowed and (has_schema(prim, "PhysicsDriveAPI:angular") or has_schema(prim, "PhysicsDriveAPI:linear")):
                locked_or_driven += 1


    collider_report = collider_summary(stage, usd_path.parent / "vhacd_manifest.json")
    return {
        "batch": batch_name,
        "usd": report_path(usd_path),
        "rigid_bodies": len(rigid),
        "friction_bound": friction_bound,
        "collision_owning_rigid_bodies": len(collision_rigid),
        "friction_missing_on_collision_rigid_bodies": len(collision_rigid) - friction_bound,
        "friction_missing_paths": friction_missing_paths,
        "unresolved_references": len(reference_report.get("unresolved", [])),
        "unresolved_reference_details": report_value(reference_report.get("unresolved", [])),
        "joints": len(joints),
        "fixed_to_world_or_missing_body": fixed_to_world,
        "driven_joints": locked_or_driven,
        "collisions": collider_report["standard_collisions"],
        "legacy_colliders": collider_report["legacy_colliders"],
        "missing_mesh_collision_api": collider_report["missing_mesh_collision_api"],
        "wrong_approximation": collider_report["wrong_approximation"],
        "manifest_expected_hulls": collider_report["manifest_expected_hulls"],
        "manifest_actual_by_link": collider_report["manifest_actual_by_link"],
        "manifest_mismatches": collider_report["manifest_mismatches"],
        "grasps": grasps,
    }


def main():
    rows = [review_stage(batch, usd_path) for batch, usd_path in usd_files()]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / "conversion_review.json"
    out_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for row in rows:
        flags = []
        if row.get("error"):
            flags.append(row["error"])
        if row.get("rigid_bodies", 0) == 0:
            flags.append("no rigid bodies")
        if row.get("collisions", 0) == 0:
            flags.append("no collisions")
        if row.get("legacy_colliders", 0):
            flags.append("legacy colliders")
        if row.get("missing_mesh_collision_api"):
            flags.append("collision schema missing")
        if row.get("wrong_approximation"):
            flags.append("wrong collision approximation")
        if row.get("manifest_mismatches"):
            flags.append("collision manifest mismatch")
        if row.get("friction_missing_on_collision_rigid_bodies", 0) > 0:
            flags.append("partial friction")
        if row.get("unresolved_references", 0) > 0:
            flags.append("unresolved references")
        if row.get("driven_joints", 0):
            flags.append("driven joints")
        if row.get("fixed_to_world_or_missing_body", 0):
            flags.append("joint missing body target")
        if row.get("grasps", 0) == 0:
            flags.append("no grasps")
        print(f"{Path(row['usd']).name}: rigid={row.get('rigid_bodies', 0)} friction={row.get('friction_bound', 0)} joints={row.get('joints', 0)} collisions={row.get('collisions', 0)} grasps={row.get('grasps', 0)} {'; '.join(flags)}")
    print(f"Wrote {out_path}")
    return 1 if any("error" in r or r.get("unresolved_references", 0) or
                    r.get("friction_missing_on_collision_rigid_bodies", 0) or
                    r.get("legacy_colliders", 0) or r.get("missing_mesh_collision_api") or
                    r.get("wrong_approximation") or r.get("manifest_mismatches") for r in rows) else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        app.close()
