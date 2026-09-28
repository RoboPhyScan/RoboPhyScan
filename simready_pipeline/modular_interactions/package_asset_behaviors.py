from __future__ import annotations

import argparse
import json
import math
import traceback
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from behavior_runtime import _drive_type, _find_prim, _target_from_spec, _to_usd_target, thread_lead_and_ratio
from schema import workspace_root


RUNTIME_PATH = Path(__file__).with_name("behavior_runtime.py")


def render_asset_behavior(asset_id: str, interactions: list[dict]) -> str:
    runtime = RUNTIME_PATH.read_text(encoding="utf-8")
    config = json.dumps([item for item in interactions if item["module"] != "thread_coupling"], ensure_ascii=True, sort_keys=True)
    return runtime + f'''

import carb
import omni.usd
from omni.kit.scripting import BehaviorScript

ASSET_ID = {asset_id!r}
INTERACTIONS = {config}


class AssetInteractionBehavior(BehaviorScript):
    def on_init(self):
        stage = omni.usd.get_context().get_stage()
        self.context = RuntimeContext(stage, self.prim_path)
        self.modules = [create_module(self.context, spec) for spec in INTERACTIONS]
        for module in self.modules:
            module.initialize()
        carb.log_info(f"{{ASSET_ID}}: initialized {{len(self.modules)}} interaction modules")

    def on_play(self):
        self.context.on_play()
        for module in self.modules:
            module.on_play()

    def on_update(self, current_time: float, delta_time: float):
        if delta_time <= 0:
            return
        for module in self.modules:
            module.update(current_time, delta_time)

    def on_pause(self):
        pass

    def on_stop(self):
        for module in self.modules:
            module.on_stop()
        self.context.on_stop()

    def on_destroy(self):
        for module in self.modules:
            module.destroy()
        self.modules.clear()
'''


def _behavior_prim(stage):
    for prim in stage.Traverse():
        if "PhysicsArticulationRootAPI" in (prim.GetAppliedSchemas() or []):
            return prim
    prim = stage.GetDefaultPrim()
    if prim and prim.IsValid():
        return prim
    return next(iter(stage.GetPseudoRoot().GetChildren()), None)


def attach_behavior_script(stage, relative_script_path: str = "./resource/asset_behavior.py") -> str:
    import OmniScriptingSchemaTools
    from pxr import Sdf

    prim = _behavior_prim(stage)
    if prim is None:
        raise RuntimeError("USD has no prim to carry the behavior script")
    OmniScriptingSchemaTools.applyOmniScriptingAPI(stage, prim.GetPath())
    attr = prim.GetAttribute("omni:scripting:scripts")
    if not attr:
        attr = prim.CreateAttribute(
            "omni:scripting:scripts",
            Sdf.ValueTypeNames.AssetArray,
            Sdf.VariabilityVarying,
        )
    attr.Set([Sdf.AssetPath(relative_script_path)])
    return str(prim.GetPath())


def neutralize_magnetic_drives(stage, interactions: list[dict]) -> None:
    from pxr import UsdPhysics

    for spec in interactions:
        if spec.get("module") != "magnetic":
            continue
        prim = _find_prim(stage, spec.get("target", {}))
        if prim is None:
            raise RuntimeError(f"magnetic target not found: {spec.get('id')}")
        drive_type = _drive_type(prim, spec)
        drive = UsdPhysics.DriveAPI.Apply(prim, drive_type)
        closed = float(spec.get("physics", {}).get("closed_position", 0.0))
        drive.CreateTargetPositionAttr().Set(_to_usd_target(closed, drive_type))
        drive.CreateStiffnessAttr().Set(0.0)
        drive.CreateDampingAttr().Set(0.0)
        drive.CreateMaxForceAttr().Set(0.0)


def _owned_targets(spec: dict):
    yield spec["target"]
    if spec.get("module") == "thread_coupling":
        yield _target_from_spec(spec["usd"]["hinge"])
        yield _target_from_spec(spec["usd"]["prismatic"])
        return
    usd = spec.get("usd", {})
    for key in ("support_drives", "disable_drives", "actions"):
        for item in usd.get(key, []):
            if item.get("type", "set_drive_target") == "set_drive_target":
                yield _target_from_spec(item)
    for key in ("trigger_drive", "latch", "guard", "guard_drive"):
        item = usd.get(key)
        if item:
            yield _target_from_spec(item) or spec["target"]


def clear_owned_joint_behaviors(stage, interactions: list[dict]) -> None:
    from pxr import UsdPhysics

    seen = set()
    for spec in interactions:
        for target in _owned_targets(spec):
            prim = _find_prim(stage, target)
            if prim is None or prim.GetPath() in seen:
                continue
            seen.add(prim.GetPath())
            for attr in list(prim.GetAttributes()):
                if attr.GetName().startswith(("drive:linear:", "drive:angular:", "state:linear:", "state:angular:")):
                    attr.Block()
                elif attr.GetName() == "physxJoint:jointFriction":
                    attr.Block()
            for drive_type in ("linear", "angular"):
                if f"PhysicsDriveAPI:{drive_type}" in (prim.GetAppliedSchemas() or []):
                    prim.RemoveAPI(UsdPhysics.DriveAPI, drive_type)


def zero_rigid_body_damping(stage) -> None:
    from pxr import PhysxSchema, UsdPhysics

    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        body.CreateLinearDampingAttr().Set(0.0)
        body.CreateAngularDampingAttr().Set(0.0)


def preset_damping(stage, interactions: list[dict]) -> None:
    from pxr import PhysxSchema

    for spec in interactions:
        if spec.get("module") != "damping":
            continue
        prim = _find_prim(stage, spec["target"])
        if prim is None:
            raise RuntimeError(f"damping target not found: {spec.get('id')}")
        PhysxSchema.PhysxJointAPI.Apply(prim).CreateJointFrictionAttr().Set(
            float(spec["usd"]["joint_friction"])
        )


def preset_thread_couplings(stage, interactions: list[dict]) -> None:
    from pxr import PhysxSchema, Sdf, UsdPhysics

    for spec in interactions:
        if spec.get("module") != "thread_coupling":
            continue
        usd = spec["usd"]
        hinge = _find_prim(stage, _target_from_spec(usd["hinge"]))
        prismatic = _find_prim(stage, _target_from_spec(usd["prismatic"]))
        if hinge is None or prismatic is None:
            raise RuntimeError(f"thread coupling joints not found: {spec.get('id')}")
        if hinge.GetTypeName() != "PhysicsRevoluteJoint" or prismatic.GetTypeName() != "PhysicsPrismaticJoint":
            raise RuntimeError(f"thread coupling needs revolute + prismatic joints: {spec.get('id')}")
        meters_per_unit = float(stage.GetMetadata("metersPerUnit") or 1.0)
        lead_m, ratio = thread_lead_and_ratio(
            usd["pitch_m"], usd["starts"], opening_sign=int(usd.get("opening_sign", 1)),
            meters_per_unit=meters_per_unit,
        )
        backend = usd.get("backend", "rack")
        configured_axis = str(usd.get("axis", "")).upper()
        actual_axis = str(hinge.GetAttribute("physics:axis").Get() or "X").upper()
        if configured_axis and configured_axis != actual_axis:
            raise RuntimeError(
                f"thread axis mismatch for {spec.get('id')}: configured {configured_axis}, USD {actual_axis}"
            )
        if backend == "rack":
            path = Sdf.Path(f"/World/{spec['id']}")
            coupling = PhysxSchema.PhysxPhysicsRackAndPinionJoint.Define(stage, path)
            hinge_joint = UsdPhysics.Joint(hinge)
            prismatic_joint = UsdPhysics.Joint(prismatic)
            coupling.CreateBody0Rel().SetTargets(hinge_joint.GetBody1Rel().GetTargets())
            coupling.CreateBody1Rel().SetTargets(prismatic_joint.GetBody1Rel().GetTargets())
            coupling.CreateHingeRel().SetTargets([hinge.GetPath()])
            coupling.CreatePrismaticRel().SetTargets([prismatic.GetPath()])
            coupling.CreateRatioAttr().Set(ratio)
            coupling.CreateExcludeFromArticulationAttr().Set(True)
        elif backend == "mimic":
            axis = actual_axis
            mimic = PhysxSchema.PhysxMimicJointAPI.Apply(hinge, f"rot{axis}")
            mimic.CreateReferenceJointRel().SetTargets([prismatic.GetPath()])
            # PhysX mimic uses theta_deg = gearing * z_stage_units.  The
            # configured opening sign already belongs in ratio; do not flip it again.
            mimic.CreateGearingAttr().Set(ratio)
            mimic.CreateOffsetAttr().Set(0.0)
        else:
            raise ValueError(f"unknown thread coupling backend: {backend}")
        PhysxSchema.PhysxJointAPI.Apply(hinge).CreateJointFrictionAttr().Set(
            float(usd["joint_friction_coefficient"])
        )
        hinge.SetCustomDataByKey("thread:backend", backend)
        hinge.SetCustomDataByKey("thread:axis", actual_axis)
        hinge.SetCustomDataByKey("thread:opening_sign", int(usd.get("opening_sign", 1)))
        hinge.SetCustomDataByKey("thread:lead_m", lead_m)
        hinge.SetCustomDataByKey("thread:lead_stage_units", lead_m / meters_per_unit)
        hinge.SetCustomDataByKey("thread:hinge_body0", ",".join(str(x) for x in UsdPhysics.Joint(hinge).GetBody0Rel().GetTargets()))
        hinge.SetCustomDataByKey("thread:hinge_body1", ",".join(str(x) for x in UsdPhysics.Joint(hinge).GetBody1Rel().GetTargets()))
        hinge.SetCustomDataByKey("thread:prismatic_body0", ",".join(str(x) for x in UsdPhysics.Joint(prismatic).GetBody0Rel().GetTargets()))
        hinge.SetCustomDataByKey("thread:prismatic_body1", ",".join(str(x) for x in UsdPhysics.Joint(prismatic).GetBody1Rel().GetTargets()))
        hinge.SetCustomDataByKey("thread:parameter_source", usd.get("parameter_source", "explicit test assumption"))


def assert_thread_coupling_cpu_ready(stage, interactions: list[dict]) -> None:
    from pxr import UsdPhysics

    if not any(spec.get("module") == "thread_coupling" for spec in interactions):
        return
    if not any(prim.HasAPI(UsdPhysics.ArticulationRootAPI) for prim in stage.Traverse()):
        raise RuntimeError("thread coupling requires the source articulation for joint friction")
    scene = next((prim for prim in stage.Traverse() if prim.GetTypeName() == "PhysicsScene"), None)
    if scene and scene.GetAttribute("physxScene:enableGPUDynamics").Get():
        raise RuntimeError("thread coupling requires CPU PhysX; GPU dynamics is enabled")


def _preset_drive(stage, target: dict, drive_type: str, target_position: float, stiffness: float, damping: float, max_force: float) -> None:
    from pxr import UsdPhysics

    prim = _find_prim(stage, target)
    if prim is None:
        raise RuntimeError(f"drive target not found: {target}")
    drive = UsdPhysics.DriveAPI.Apply(prim, drive_type)
    drive.CreateTargetPositionAttr().Set(_to_usd_target(target_position, drive_type))
    drive.CreateStiffnessAttr().Set(float(stiffness))
    drive.CreateDampingAttr().Set(float(damping))
    drive.CreateMaxForceAttr().Set(float(max_force))


def _preset_limits(stage, target: dict, lower: float, upper: float) -> None:
    prim = _find_prim(stage, target)
    if prim is None:
        raise RuntimeError(f"joint target not found: {target}")
    prim.GetAttribute("physics:lowerLimit").Set(float(lower))
    prim.GetAttribute("physics:upperLimit").Set(float(upper))


def _mass_for_names(stage, prim_names: list[str]) -> float:
    from pxr import UsdPhysics

    total = 0.0
    wanted = set(prim_names)
    for prim in stage.Traverse():
        if prim.GetName() not in wanted:
            continue
        mass = UsdPhysics.MassAPI(prim).GetMassAttr().Get()
        if mass:
            total += float(mass)
    return total


def _urdf_mass_for_names(stage, prim_names: list[str]) -> float:
    try:
        usd_path = Path(stage.GetRootLayer().realPath)
    except Exception:
        return 0.0
    urdf_path = usd_path.with_suffix(".urdf")
    if not urdf_path.exists():
        return 0.0

    wanted = set(prim_names)
    total = 0.0
    root = ET.parse(urdf_path).getroot()
    for link in root.findall("link"):
        if link.get("name") not in wanted:
            continue
        mass = link.find("inertial/mass")
        if mass is not None and mass.get("value"):
            total += float(mass.get("value"))
    return total


def _bbox_mass_for_names(stage, prim_names: list[str], usd: dict) -> float:
    if not usd.get("estimate_mass_from_bbox"):
        return 0.0

    from pxr import UsdGeom

    density = float(usd.get("density_kg_per_m3", 1000.0))
    fill_ratio = float(usd.get("fill_ratio", 0.5))
    max_mass = float(usd.get("max_bbox_mass", 0.25))
    cache = UsdGeom.BBoxCache(0.0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=True)
    total = 0.0
    wanted = set(prim_names)
    for prim in stage.Traverse():
        if prim.GetName() not in wanted:
            continue
        box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        size = box.GetSize()
        volume = max(0.0, float(size[0]) * float(size[1]) * float(size[2]))
        total += volume * density * fill_ratio
    return min(total, max_mass)


def _drive_params(stage, usd: dict, physics: dict | None = None) -> tuple[float, float, float]:
    physics = physics or {}
    base_stiffness = float(usd.get("stiffness", physics.get("kp", 0.0)))
    base_damping = float(usd.get("damping", physics.get("kd", 0.0)))
    base_max_force = float(usd.get("max_force", physics.get("max_force", 0.0)))
    if "mass_prim_names" not in usd:
        return base_stiffness, base_damping, base_max_force

    authored_mass = _mass_for_names(stage, usd["mass_prim_names"])
    urdf_mass = _urdf_mass_for_names(stage, usd["mass_prim_names"])
    bbox_mass = _bbox_mass_for_names(stage, usd["mass_prim_names"], usd)
    measured_mass = max(authored_mass, urdf_mass, bbox_mass)
    mass = float(usd.get("base_effective_mass", usd.get("min_effective_mass", 0.02)))
    mass += measured_mass * float(usd.get("mass_scale", 1.0))
    gravity = float(usd.get("gravity", 9.81))
    safety_factor = float(usd.get("safety_factor", 8.0))
    allowed_sag = float(usd.get("allowed_sag", 0.001))
    min_force = float(usd.get("min_max_force", base_max_force))
    min_stiffness = float(usd.get("min_stiffness", base_stiffness))
    damping_ratio = float(usd.get("damping_ratio", 1.0))
    effective_mass = max(mass, float(usd.get("min_effective_mass", 0.02)))
    if "max_effective_mass" in usd:
        effective_mass = min(effective_mass, float(usd["max_effective_mass"]))

    max_force = max(base_max_force, min_force, effective_mass * gravity * safety_factor)
    stiffness = max(min_stiffness, max_force / allowed_sag)
    damping = max(base_damping, float(usd.get("min_damping", base_damping)), 2.0 * damping_ratio * math.sqrt(stiffness * effective_mass))
    return stiffness, damping, max_force


def _support_params(stage, support: dict) -> tuple[float, float, float]:
    return _drive_params(stage, support)


def preset_trigger_drives(stage, interactions: list[dict]) -> None:
    for spec in interactions:
        if spec.get("module") != "trigger":
            continue
        usd = spec.get("usd", {})
        trigger_drive = usd.get("trigger_drive")
        if trigger_drive:
            target = _target_from_spec(trigger_drive) or {"prim_name": spec["target"]["prim_name"]}
            _preset_drive(
                stage,
                target,
                trigger_drive.get("drive_type", "angular"),
                float(trigger_drive.get("target_position", 0.0)),
                float(trigger_drive["stiffness"]),
                float(trigger_drive["damping"]),
                float(trigger_drive["max_force"]),
            )
        latch = usd.get("latch")
        if latch and latch.get("initial_state") == "CLOSED_LOCKED":
            _preset_drive(
                stage,
                _target_from_spec(latch),
                latch.get("drive_type", "angular"),
                float(latch["closed_position"]),
                float(latch["lock_stiffness"]),
                float(latch["lock_damping"]),
                float(latch["lock_max_force"]),
            )
        guard_drive = usd.get("guard_drive")
        if guard_drive:
            _preset_drive(
                stage,
                _target_from_spec(guard_drive),
                guard_drive.get("drive_type", "angular"),
                float(guard_drive.get("target_position", 0.0)),
                float(guard_drive["stiffness"]),
                float(guard_drive["damping"]),
                float(guard_drive["max_force"]),
            )


def preset_spring_drives(stage, interactions: list[dict]) -> None:
    for spec in interactions:
        if spec.get("module") != "spring":
            continue
        target = spec["target"]
        drive_type = spec.get("usd", {}).get("drive_type")
        if not drive_type:
            prim = _find_prim(stage, target)
            if prim is None:
                raise RuntimeError(f"spring target not found: {spec.get('id')}")
            drive_type = _drive_type(prim, spec)
        stiffness, damping, max_force = _drive_params(stage, spec.get("usd", {}), spec.get("physics", {}))
        _preset_drive(
            stage,
            target,
            drive_type,
            float(spec.get("usd", {}).get("target_position", spec.get("physics", {}).get("x_ref", 0.0))),
            stiffness,
            damping,
            max_force,
        )
        usd = spec.get("usd", {})
        if "lower_limit" in usd and "upper_limit" in usd:
            _preset_limits(stage, target, usd["lower_limit"], usd["upper_limit"])
        for support in spec.get("usd", {}).get("support_drives", []):
            stiffness, damping, max_force = _support_params(stage, support)
            target = _target_from_spec(support)
            _preset_drive(
                stage,
                target,
                support.get("drive_type", drive_type),
                float(support.get("target_position", 0.0)),
                stiffness,
                damping,
                max_force,
            )
            if "lower_limit" in support and "upper_limit" in support:
                _preset_limits(stage, target, support["lower_limit"], support["upper_limit"])
        for disabled in spec.get("usd", {}).get("disable_drives", []):
            _preset_drive(
                stage,
                _target_from_spec(disabled),
                disabled.get("drive_type", drive_type),
                float(disabled.get("target_position", 0.0)),
                0.0,
                0.0,
                0.0,
            )


def package_asset(usd_path: Path, interactions: list[dict], asset_id: str | None = None) -> dict:
    from pxr import Usd

    usd_path = Path(usd_path).resolve()
    if not interactions:
        raise ValueError(f"no interactions for USD: {usd_path}")
    resource_dir = usd_path.parent / "resource"
    resource_dir.mkdir(exist_ok=True)
    script_path = resource_dir / "asset_behavior.py"
    script_path.write_text(
        render_asset_behavior(asset_id or interactions[0]["batch"], interactions),
        encoding="utf-8",
    )
    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        raise RuntimeError(f"cannot open USD: {usd_path}")
    clear_owned_joint_behaviors(stage, interactions)
    zero_rigid_body_damping(stage)
    prim_path = attach_behavior_script(stage)
    preset_damping(stage, interactions)
    neutralize_magnetic_drives(stage, interactions)
    preset_spring_drives(stage, interactions)
    preset_trigger_drives(stage, interactions)
    preset_thread_couplings(stage, interactions)
    assert_thread_coupling_cpu_ready(stage, interactions)
    stage.Save()
    return {
        "batch": interactions[0]["batch"],
        "usd": str(usd_path),
        "script": str(script_path),
        "script_asset_path": "./resource/asset_behavior.py",
        "prim_path": prim_path,
        "modules": [item["id"] for item in interactions],
        "status": "ok",
    }


def package_assets(config_path: Path) -> list[dict]:
    ws = workspace_root()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    grouped = defaultdict(list)
    for interaction in data["interactions"]:
        grouped[interaction["target"]["usd"]].append(interaction)

    report = []
    for configured_path, interactions in grouped.items():
        usd_path = Path(configured_path)
        if not usd_path.is_absolute():
            usd_path = ws / usd_path
        report.append(package_asset(usd_path, interactions))
    return report


def package_selected_assets(config_path: Path, interaction_ids: set[str]) -> list[dict]:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    selected = [item for item in data["interactions"] if item["id"] in interaction_ids]
    if len(selected) != len(interaction_ids):
        found = {item["id"] for item in selected}
        raise ValueError(f"unknown interaction ids: {sorted(interaction_ids - found)}")
    grouped = defaultdict(list)
    for interaction in selected:
        grouped[interaction["target"]["usd"]].append(interaction)
    ws = workspace_root()
    return [package_asset(ws / path, specs) for path, specs in grouped.items()]


def main() -> int:
    from isaacsim import SimulationApp

    ws = workspace_root()
    parser = argparse.ArgumentParser(description="Package self-contained Isaac Sim behaviors into copied assets")
    parser.add_argument("--config", type=Path, default=ws / "config" / "interaction_bindings.json")
    parser.add_argument("--only", help="comma-separated interaction IDs")
    args = parser.parse_args()
    app = SimulationApp({"headless": True})
    try:
        report = package_selected_assets(args.config, set(args.only.split(","))) if args.only else package_assets(args.config)
        out = Path(os.environ.get("PIPELINE_REPORT_DIR", ws / ".pipeline_reports")) / "behavior_package_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        for row in report:
            print(f"ok: {row['batch']} -> {row['prim_path']} ({len(row['modules'])} modules)")
        print(f"Wrote {out}")
        return 0
    except Exception as error:
        out = Path(os.environ.get("PIPELINE_REPORT_DIR", ws / ".pipeline_reports")) / "behavior_package_error.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"error": str(error), "traceback": traceback.format_exc()}, indent=2) + "\n", encoding="utf-8")
        raise
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
