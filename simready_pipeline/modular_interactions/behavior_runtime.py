from __future__ import annotations

import math


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def damped_velocity(velocity: float, *, gamma: float, dt: float, mass: float) -> float:
    """RoboPhyScan supplement: v_next = v * max(0, 1 - gamma * dt / m)."""
    if mass <= 0 or dt <= 0 or gamma < 0:
        raise ValueError("mass/dt must be positive and gamma must be non-negative")
    return velocity * max(0.0, 1.0 - gamma * dt / mass)


def spring_effort(
    position: float,
    previous_position: float,
    *,
    dt: float,
    x_ref: float,
    kp: float,
    kd: float,
    max_force: float,
) -> float:
    """RoboPhyScan supplement PD spring, capped to a realistic actuator effort."""
    if dt <= 0 or kp < 0 or kd < 0 or max_force <= 0:
        raise ValueError("invalid spring parameters")
    effort = kp * (x_ref - position) - kd * (position - previous_position) / dt
    return _clamp(effort, -max_force, max_force)


def magnetic_state(distance: float, *, d_on: float, d_off: float, was_active: bool) -> bool:
    if not 0 <= d_on < d_off:
        raise ValueError("magnetic thresholds require 0 <= d_on < d_off")
    if distance < d_on:
        return True
    if distance > d_off:
        return False
    return was_active


def magnetic_force(distance: float, *, d_on: float, k_mag: float, max_force: float) -> float:
    return _clamp(k_mag * (d_on - distance), 0.0, max_force)


def trigger_state(position: float, *, low: float, high: float, previous: int) -> int:
    if high <= low:
        raise ValueError("trigger high threshold must exceed low threshold")
    if position > high:
        return 1
    if position < low:
        return 0
    return int(bool(previous))


def _find_prim(stage, target: dict):
    path = target.get("prim_path")
    if path:
        prim = stage.GetPrimAtPath(path)
        return prim if prim and prim.IsValid() else None
    name = target.get("prim_name")
    for prim in stage.Traverse():
        if prim.GetName() == name:
            return prim
    return None


def _target_from_spec(spec: dict) -> dict:
    target = {}
    if "prim_path" in spec:
        target["prim_path"] = spec["prim_path"]
    if "target_prim" in spec:
        target["prim_path"] = spec["target_prim"]
    if "prim_name" in spec:
        target["prim_name"] = spec["prim_name"]
    if "target_prim_name" in spec:
        target["prim_name"] = spec["target_prim_name"]
    return target


def _drive_type(prim, spec: dict) -> str:
    configured = spec.get("usd", {}).get("drive_type")
    if configured:
        return configured
    return "linear" if prim.GetTypeName() == "PhysicsPrismaticJoint" else "angular"


def _to_usd_target(value: float, drive_type: str) -> float:
    return math.degrees(value) if drive_type == "angular" else value


def thread_lead_and_ratio(
    pitch_m: float, starts: int, *, opening_sign: int = 1, meters_per_unit: float = 1.0
) -> tuple[float, float]:
    """Return lead in metres and native rack-and-pinion ratio in degrees/stage unit."""
    pitch_m = float(pitch_m)
    if pitch_m <= 0:
        raise ValueError("thread pitch must be positive")
    if isinstance(starts, bool) or int(starts) != starts or starts <= 0:
        raise ValueError("thread starts must be a positive integer")
    if opening_sign not in (-1, 1):
        raise ValueError("thread opening_sign must be -1 or 1")
    meters_per_unit = float(meters_per_unit)
    if meters_per_unit <= 0:
        raise ValueError("meters_per_unit must be positive")
    lead_m = pitch_m * int(starts)
    return lead_m, opening_sign * 360.0 * meters_per_unit / lead_m


def thread_motion_slope(pitch_m: float, starts: int) -> float:
    """Return the unsigned paper relation dz/dtheta in metres per radian."""
    lead_m, _ = thread_lead_and_ratio(pitch_m, starts)
    return lead_m / (2.0 * math.pi)


def _configure_drive(prim, spec: dict, *, damping_only: bool = False, disabled: bool = False):
    from pxr import UsdPhysics

    usd = spec.get("usd", {})
    physics = spec.get("physics", {})
    drive_type = _drive_type(prim, spec)
    drive = UsdPhysics.DriveAPI.Apply(prim, drive_type)
    x_ref = float(physics.get("x_ref", physics.get("closed_position", usd.get("target_position", 0.0))))
    drive.CreateTargetPositionAttr().Set(_to_usd_target(x_ref, drive_type))
    stiffness = float(usd.get("stiffness", physics.get("kp", 0.0)))
    damping = float(physics.get("gamma", usd.get("damping", physics.get("kd", 0.0))))
    max_force = usd.get("max_force", physics.get("max_force"))
    if "mass_prim_names" in usd:
        stiffness = max(stiffness, float((prim.GetAttribute(f"drive:{drive_type}:physics:stiffness") or drive.CreateStiffnessAttr()).Get() or 0.0))
        damping = max(damping, float((prim.GetAttribute(f"drive:{drive_type}:physics:damping") or drive.CreateDampingAttr()).Get() or 0.0))
        if max_force is not None:
            max_force = max(float(max_force), float((prim.GetAttribute(f"drive:{drive_type}:physics:maxForce") or drive.CreateMaxForceAttr()).Get() or 0.0))
    drive.CreateStiffnessAttr().Set(0.0 if disabled or damping_only else stiffness)
    drive.CreateDampingAttr().Set(0.0 if disabled else damping)
    if disabled:
        drive.CreateMaxForceAttr().Set(0.0)
    elif max_force is not None:
        drive.CreateMaxForceAttr().Set(float(max_force))
    return drive, drive_type


def _configure_joint_friction(prim, spec: dict):
    from pxr import PhysxSchema, UsdPhysics

    for drive_type in ("linear", "angular"):
        prefix = f"drive:{drive_type}:"
        for attr in list(prim.GetAttributes()):
            if attr.GetName().startswith(prefix):
                attr.Block()
        if f"PhysicsDriveAPI:{drive_type}" in (prim.GetAppliedSchemas() or []):
            prim.RemoveAPI(UsdPhysics.DriveAPI, drive_type)

    for attr in list(prim.GetAttributes()):
        if attr.GetName().startswith("state:linear:"):
            attr.Block()

    friction = float(spec.get("usd", {})["joint_friction"])
    PhysxSchema.PhysxJointAPI.Apply(prim).CreateJointFrictionAttr().Set(friction)


def _configure_limits(prim, usd: dict):
    from pxr import Sdf

    for key, attr_name in (("lower_limit", "physics:lowerLimit"), ("upper_limit", "physics:upperLimit")):
        if key not in usd:
            continue
        attr = prim.GetAttribute(attr_name)
        if not attr:
            attr = prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Float)
        attr.Set(float(usd[key]))


class RuntimeContext:
    """Isaac Sim adapter shared by all modules in one asset behavior."""

    def __init__(self, stage, root_path):
        self.stage = stage
        self.root_path = str(root_path)
        self._dc = None
        self._articulation = None
        self._handles = {}

    def on_play(self):
        from omni.isaac.dynamic_control import _dynamic_control

        self._dc = _dynamic_control.acquire_dynamic_control_interface()
        self._articulation = self._dc.get_articulation(self.root_path)
        self._handles.clear()

    def on_stop(self):
        self._articulation = None
        self._handles.clear()

    def prim(self, spec: dict):
        prim = _find_prim(self.stage, spec.get("target", {}))
        if prim is None:
            raise RuntimeError(f"interaction target not found: {spec.get('id')}")
        return prim

    def _handle(self, spec: dict):
        if self._dc is None or self._articulation is None:
            return None
        name = self.prim(spec).GetName()
        if name not in self._handles:
            self._handles[name] = self._dc.find_articulation_dof(self._articulation, name)
        return self._handles[name]

    def position(self, spec: dict) -> float:
        runtime_value = self.runtime_position(spec)
        if runtime_value is not None:
            return runtime_value
        prim = self.prim(spec)
        drive_type = _drive_type(prim, spec)
        attr = prim.GetAttribute(f"state:{drive_type}:physics:position")
        value = float(attr.Get() or 0.0) if attr else 0.0
        return math.radians(value) if drive_type == "angular" else value

    def runtime_position(self, spec: dict):
        handle = self._handle(spec)
        if handle is not None:
            from omni.isaac.dynamic_control import _dynamic_control

            if handle != _dynamic_control.INVALID_HANDLE:
                return float(self._dc.get_dof_state(handle, _dynamic_control.STATE_ALL).pos)
        return None

    def velocity(self, spec: dict):
        handle = self._handle(spec)
        if handle is None:
            return None
        from omni.isaac.dynamic_control import _dynamic_control

        if handle == _dynamic_control.INVALID_HANDLE:
            return None
        return float(self._dc.get_dof_state(handle, _dynamic_control.STATE_ALL).vel)

    def set_velocity(self, spec: dict, velocity: float) -> bool:
        handle = self._handle(spec)
        if handle is None:
            return False
        from omni.isaac.dynamic_control import _dynamic_control

        if handle == _dynamic_control.INVALID_HANDLE:
            return False
        self._dc.set_dof_velocity(handle, float(velocity))
        return True

    def apply_effort(self, spec: dict, effort: float) -> bool:
        handle = self._handle(spec)
        if handle is None:
            return False
        from omni.isaac.dynamic_control import _dynamic_control

        if handle == _dynamic_control.INVALID_HANDLE:
            return False
        self._dc.set_dof_effort(handle, float(effort))
        return True

    def set_position_target(self, action: dict):
        target = _target_from_spec(action)
        drive_type = action.get("drive_type", "angular")
        value = float(action["target_position"])
        action_spec = {"id": "action", "target": target, "usd": {"drive_type": drive_type}}
        from pxr import UsdPhysics

        drive = UsdPhysics.DriveAPI.Apply(self.prim(action_spec), drive_type)
        drive.CreateStiffnessAttr().Set(float(action["stiffness"]))
        drive.CreateDampingAttr().Set(float(action["damping"]))
        drive.CreateMaxForceAttr().Set(float(action["max_force"]))
        handle = self._handle(action_spec)
        if handle is not None:
            from omni.isaac.dynamic_control import _dynamic_control

            if handle != _dynamic_control.INVALID_HANDLE:
                self._dc.set_dof_position_target(handle, value)
                return
        drive.CreateTargetPositionAttr().Set(_to_usd_target(value, drive_type))


class InteractionModule:
    """Uniform lifecycle contract for every asset-bound interaction module."""

    def __init__(self, context: RuntimeContext, spec: dict):
        self.context = context
        self.spec = spec
        self.prim = None

    def initialize(self):
        self.prim = self.context.prim(self.spec)

    def on_play(self):
        pass

    def update(self, current_time: float, delta_time: float):
        pass

    def on_stop(self):
        pass

    def destroy(self):
        self.prim = None


class DampingModule(InteractionModule):
    def initialize(self):
        super().initialize()
        _configure_joint_friction(self.prim, self.spec)

    def on_play(self):
        pass

    def update(self, current_time: float, delta_time: float):
        pass

    def on_stop(self):
        pass

    def destroy(self):
        super().destroy()


class SpringModule(InteractionModule):
    def initialize(self):
        super().initialize()
        _configure_drive(self.prim, self.spec)
        _configure_limits(self.prim, self.spec.get("usd", {}))
        for support in self.spec.get("usd", {}).get("support_drives", []):
            support_spec = {
                "id": f"{self.spec.get('id', 'spring')}:support",
                "target": _target_from_spec(support),
                "usd": support,
                "physics": {},
            }
            support_prim = self.context.prim(support_spec)
            _configure_drive(support_prim, support_spec)
            _configure_limits(support_prim, support)

    def on_play(self):
        pass

    def update(self, current_time: float, delta_time: float):
        pass

    def on_stop(self):
        pass

    def destroy(self):
        super().destroy()


class MagneticModule(InteractionModule):
    def initialize(self):
        super().initialize()
        self.drive, self.drive_type = _configure_drive(self.prim, self.spec, disabled=True)
        self.active = False

    def on_play(self):
        self.active = False
        self._set_drive(False)

    def _set_drive(self, active: bool, max_force: float = 0.0):
        usd = self.spec.get("usd", {})
        self.drive.CreateStiffnessAttr().Set(float(usd.get("stiffness", 0.0)) if active else 0.0)
        self.drive.CreateDampingAttr().Set(float(usd.get("damping", 0.0)) if active else 0.0)
        self.drive.CreateMaxForceAttr().Set(max_force if active else 0.0)

    def update(self, current_time: float, delta_time: float):
        physics = self.spec["physics"]
        position_reader = getattr(self.context, "runtime_position", None)
        position = position_reader(self.spec) if position_reader else self.context.position(self.spec)
        if position is None:
            self.active = False
            self._set_drive(False)
            return
        closed = float(physics.get("closed_position", 0.0))
        if self.drive_type == "angular":
            radius = float(physics["effective_radius"])
            distance = 2.0 * radius * math.sin(min(abs(position - closed), math.pi) / 2.0)
        else:
            radius = 1.0
            distance = abs(position - closed)
        self.active = magnetic_state(
            distance,
            d_on=float(physics["d_on"]),
            d_off=float(physics["d_off"]),
            was_active=self.active,
        )
        force = magnetic_force(
            distance,
            d_on=float(physics["d_on"]),
            k_mag=float(physics["k_mag"]),
            max_force=float(physics["max_force"]),
        ) if self.active else 0.0
        self._set_drive(self.active, force * radius)

    def on_stop(self):
        self.active = False
        self._set_drive(False)

    def destroy(self):
        self.drive = None
        super().destroy()


class TriggerModule(InteractionModule):
    CLOSED_LOCKED = "CLOSED_LOCKED"
    UNLOCK_KICK = "UNLOCK_KICK"
    OPEN_HOLD = "OPEN_HOLD"

    def initialize(self):
        super().initialize()
        self.state = int(self.spec["physics"].get("state_initial", 0))
        self._initialize_latch_state()
        self._apply_trigger_drive()

    def on_play(self):
        self.state = int(self.spec["physics"].get("state_initial", 0))
        self._initialize_latch_state()
        self._apply_trigger_drive()

    def _latch_spec(self, latch: dict) -> dict:
        return {
            "id": "latch",
            "target": _target_from_spec(latch),
            "usd": {"drive_type": latch.get("drive_type", "angular")},
        }

    def _initialize_latch_state(self):
        latch = self.spec.get("usd", {}).get("latch")
        if not latch:
            self.mode = self.OPEN_HOLD
            self.locked = False
            return
        explicit = latch.get("initial_state")
        if explicit:
            self.mode = explicit
        else:
            position = self.context.position(self._latch_spec(latch))
            closed = abs(position - float(latch["closed_position"])) <= float(latch["closed_tolerance"])
            self.mode = self.CLOSED_LOCKED if closed else self.OPEN_HOLD
        self.locked = self.mode == self.CLOSED_LOCKED
        self.kick_remaining = 0.0
        self.kick_fallback = False
        if self.locked:
            self.context.set_position_target(self._latch_action(latch, opening=False))

    def _disable_latch_drive(self, latch: dict):
        action = self._latch_action(latch, opening=True)
        action.update({"stiffness": 0.0, "damping": 0.0, "max_force": 0.0})
        self.context.set_position_target(action)

    def _apply_trigger_drive(self):
        drive = self.spec.get("usd", {}).get("trigger_drive")
        if not drive:
            return
        action = dict(drive)
        action.setdefault("target_prim_name", self.spec["target"]["prim_name"])
        action.setdefault("target_position", 0.0)
        action.setdefault("drive_type", "angular")
        self.context.set_position_target(action)

    def _latch_action(self, latch: dict, *, opening: bool) -> dict:
        prefix = "open" if opening else "lock"
        return {
            "type": "set_drive_target",
            "target_prim_name": latch["target_prim_name"],
            "drive_type": latch.get("drive_type", "angular"),
            "target_position": float(latch["open_position"] if opening else latch["closed_position"]),
            "stiffness": float(latch[f"{prefix}_stiffness"]),
            "damping": float(latch[f"{prefix}_damping"]),
            "max_force": float(latch[f"{prefix}_max_force"]),
        }

    def _kick_lid(self, latch: dict):
        kick = self.spec.get("usd", {}).get("kick", {})
        if not kick:
            self.context.set_position_target(self._latch_action(latch, opening=True))
            return
        lid_spec = self._latch_spec(latch)
        direction = 1.0 if float(latch["open_position"]) >= float(latch["closed_position"]) else -1.0
        velocity = direction * float(kick["velocity"])
        set_velocity = getattr(self.context, "set_velocity", None)
        self.kick_fallback = not set_velocity or not set_velocity(lid_spec, velocity)
        if self.kick_fallback:
            action = self._latch_action(latch, opening=True)
            action.update(kick.get("fallback_drive", {}))
            self.context.set_position_target(action)
        self.kick_remaining = float(kick.get("max_duration", 0.0))

    def _apply_hold_resistance(self, latch: dict):
        resistance = self.spec.get("usd", {}).get("hold_resistance", {})
        if not resistance.get("enabled", False):
            return
        lid_spec = self._latch_spec(latch)
        position = self.context.position(lid_spec)
        closing_direction = 1.0 if float(latch["closed_position"]) >= float(latch["open_position"]) else -1.0
        distance = abs(position - float(latch["closed_position"]))
        velocity_reader = getattr(self.context, "velocity", None)
        velocity = velocity_reader(lid_spec) if velocity_reader else None
        if distance > float(resistance["zone"]) or velocity is None or velocity * closing_direction <= 0:
            return
        effort = -closing_direction * min(
            float(resistance["max_force"]),
            float(resistance["damping"]) * abs(velocity),
        )
        apply_effort = getattr(self.context, "apply_effort", None)
        if apply_effort:
            apply_effort(lid_spec, effort)

    def _guard_allows_open(self) -> bool:
        guard = self.spec.get("usd", {}).get("guard")
        if not guard:
            return True
        target = _target_from_spec(guard)
        value = self.context.position({"id": "guard", "target": target, "usd": {"drive_type": guard.get("drive_type", "angular")}})
        threshold = float(guard["threshold"])
        return value <= threshold if guard.get("operator") == "<=" else value >= threshold

    def _update_latch(self, pressed: bool) -> bool:
        latch = self.spec.get("usd", {}).get("latch")
        if not latch:
            return False
        lid_spec = self._latch_spec(latch)
        velocity_reader = getattr(self.context, "velocity", None)
        velocity = velocity_reader(lid_spec) if velocity_reader else None
        closing_direction = 1.0 if float(latch["closed_position"]) >= float(latch["open_position"]) else -1.0
        closed = abs(self.context.position(lid_spec) - float(latch["closed_position"])) <= float(latch["closed_tolerance"])
        closing = velocity is None or velocity * closing_direction > 0
        if self.mode == self.CLOSED_LOCKED and pressed and self._guard_allows_open():
            self.mode = self.UNLOCK_KICK
            self.locked = False
            self._disable_latch_drive(latch)
            self._kick_lid(latch)
            self.mode = self.OPEN_HOLD
            return True
        elif self.mode == self.CLOSED_LOCKED:
            self.context.set_position_target(self._latch_action(latch, opening=False))
        elif self.mode == self.OPEN_HOLD:
            self._apply_hold_resistance(latch)
            if closed and closing:
                self.mode = self.CLOSED_LOCKED
                self.locked = True
                self.context.set_position_target(self._latch_action(latch, opening=False))
        return True

    def update(self, current_time: float, delta_time: float):
        physics = self.spec["physics"]
        previous = self.state
        self.state = trigger_state(
            self.context.position(self.spec),
            low=float(physics["epsilon_low"]),
            high=float(physics["epsilon_high"]),
            previous=previous,
        )
        if self.kick_remaining > 0:
            self.kick_remaining = max(0.0, self.kick_remaining - delta_time)
            if self.kick_fallback and self.kick_remaining == 0:
                self._disable_latch_drive(self.spec["usd"]["latch"])
                self.kick_fallback = False
        if self._update_latch(previous == 0 and self.state == 1):
            return
        if previous == 0 and self.state == 1:
            for action in self.spec.get("usd", {}).get("actions", []):
                if action.get("type") == "set_drive_target":
                    self.context.set_position_target(action)

    def on_stop(self):
        self.state = int(self.spec["physics"].get("state_initial", 0))

    def destroy(self):
        super().destroy()


MODULE_TYPES = {
    "damping": DampingModule,
    "spring": SpringModule,
    "magnetic": MagneticModule,
    "trigger": TriggerModule,
}


def create_module(context: RuntimeContext, spec: dict) -> InteractionModule:
    if spec["module"] == "thread_coupling":
        raise ValueError("thread_coupling is packaged into USD and has no runtime controller")
    try:
        return MODULE_TYPES[spec["module"]](context, spec)
    except KeyError as error:
        raise ValueError(f"unknown interaction module: {spec.get('module')}") from error
