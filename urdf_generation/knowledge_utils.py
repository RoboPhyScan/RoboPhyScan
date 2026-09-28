import numpy as np
from scipy.spatial.transform import Rotation as Rot


PROXIMITY_THRES = 0.02
AFFORDACE_PROXIMITY_THRES = 0.02
SAMPLENUM = 10000


class Region_Knowledge_Wrapper():
    def __init__(self, template_instance):
        self.instance = template_instance
    
    def check(self, func, pts, *args, **kwargs):
        res = []
        for pt in pts:
            if (not self.instance.proximation(pt)):
                res.append(False)
            else:
                res.append(func(self.instance, pt, *args, **kwargs))

        return res


def transformation_matrix(position, rotation):
    RT = np.eye(4)
    RT[:3, :3] = Rot.from_euler('xyz', rotation, degrees=False).as_matrix()
    RT[:3, -1] = position
    return RT


def inverse_transformation(pt, position, rotation):
    RT = transformation_matrix(position, rotation)
    _pt = np.array([pt[0], pt[1], pt[2], 1])
    _pt = np.linalg.inv(RT) @ _pt
    return _pt[:3]


def build_transformation_matrix(approach, closing, position):
    """
    Build a 4x4 grasp pose from gripper semantic axes.
    approach: gripper local z axis in object-local coordinates.
    closing: gripper local x axis in object-local coordinates.
    position: grasp origin in object-local coordinates.
    """
    z_axis = approach / (np.linalg.norm(approach) + 1e-12)
    x_axis = closing / (np.linalg.norm(closing) + 1e-12)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-12)
    x_axis = np.cross(y_axis, z_axis)

    T = np.eye(4)
    T[:3, 0] = x_axis
    T[:3, 1] = y_axis
    T[:3, 2] = z_axis
    T[:3, 3] = position
    return T


def convert_articulation_frame(joints, source_format="joint", target_format="world"):
    """Convert Box articulation joints between parent-relative and world frames."""
    if source_format != "joint" or target_format != "world":
        raise ValueError("Only source_format='joint' and target_format='world' are supported")

    by_child = {}
    for joint in joints:
        child = joint["child_concept"]
        if child in by_child:
            raise ValueError(f"Duplicate articulation child: {child}")
        by_child[child] = joint

    world_poses = {"world": np.eye(4)}
    pending = dict(by_child)
    while pending:
        progressed = False
        for child, joint in list(pending.items()):
            parent = joint["parent_concept"]
            if parent not in world_poses:
                continue
            origin = np.asarray(joint["origin"], dtype=float)
            if origin.shape != (4, 4) or not np.isfinite(origin).all():
                raise ValueError(f"Invalid joint origin for {parent} -> {child}")
            world_poses[child] = world_poses[parent] @ origin
            del pending[child]
            progressed = True
        if not progressed:
            unresolved = ", ".join(sorted(pending))
            raise ValueError(f"Unresolvable articulation topology: {unresolved}")

    converted = []
    for joint in joints:
        child = joint["child_concept"]
        pose = world_poses[child]
        result = {
            "parent": joint["parent"],
            "child": joint["child"],
            "parent_concept": joint["parent_concept"],
            "child_concept": child,
            "type": joint["type"],
            "origin": pose[:3, 3].tolist(),
        }
        if joint["type"] != "fixed":
            axis = np.asarray(joint.get("axis"), dtype=float)
            if axis.shape != (3,) or not np.isfinite(axis).all():
                raise ValueError(f"Invalid joint axis for {joint['parent_concept']} -> {child}")
            norm = np.linalg.norm(axis)
            if norm <= 0:
                raise ValueError(f"Zero-length joint axis for {joint['parent_concept']} -> {child}")
            result["axis"] = (pose[:3, :3] @ (axis / norm)).tolist()
            result["limit"] = joint["limit"]
        converted.append(result)
    return converted
