from concept_template import *
from geometry_template import *
from knowledge_utils import *
from scipy.spatial.transform import Rotation as Rot
import copy
import pickle


def cover_affordance(obj, pt):

    def is_affordance(obj, pt):
        if (isinstance(obj, Regular_Cover)):
            _pt = inverse_transformation(pt, obj.position, obj.rotation)

            return (_pt[0] >= -obj.outer_size[0]/2 - AFFORDACE_PROXIMITY_THRES and
                    _pt[0] <= obj.outer_size[0]/2 + AFFORDACE_PROXIMITY_THRES and
                    _pt[2] <= obj.outer_size[2] + AFFORDACE_PROXIMITY_THRES and
                    _pt[2] >= obj.outer_size[2]/2 + obj.inner_size[2]/2 - AFFORDACE_PROXIMITY_THRES and
                    _pt[1] >= -AFFORDACE_PROXIMITY_THRES and
                    _pt[1] <= obj.outer_size[1] + AFFORDACE_PROXIMITY_THRES)
        

        elif (isinstance(obj, Fourfold_Cover)):
            _pt = inverse_transformation(pt, obj.position, obj.rotation)


            if (obj.has_cover[0] == 1):                 # front
                cover_position = np.array([
                    0, 
                    obj.front_behind_size[1] * np.cos(obj.cover_rotation[0]) / 2, 
                    obj.cover_separation[0] / 2 + obj.front_behind_size[1] * np.sin(obj.cover_rotation[0]) / 2
                ])
                cover_rotation = np.array([
                    obj.cover_rotation[0],
                    0,
                    0
                ])
                __pt = inverse_transformation(_pt, cover_position, cover_rotation)

                if (__pt[0] >= -obj.front_behind_size[0]/2 - AFFORDACE_PROXIMITY_THRES and
                    __pt[0] <= obj.front_behind_size[0]/2 + AFFORDACE_PROXIMITY_THRES and 
                    __pt[1] >= obj.front_behind_size[1]/8 - AFFORDACE_PROXIMITY_THRES and
                    __pt[1] <= obj.front_behind_size[1]/2 + AFFORDACE_PROXIMITY_THRES and
                    __pt[2] >= -obj.front_behind_size[2]/2 - AFFORDACE_PROXIMITY_THRES and
                    __pt[2] <= obj.front_behind_size[2]/2 + AFFORDACE_PROXIMITY_THRES):
                    return True
            

            if (obj.has_cover[1] == 1):               # back
                cover_position = np.array([
                    0, 
                    obj.front_behind_size[1] * np.cos(obj.cover_rotation[1]) / 2, 
                    -obj.cover_separation[0] / 2 + obj.front_behind_size[1] * np.sin(obj.cover_rotation[1]) / 2
                ])
                cover_rotation = np.array([
                    obj.cover_rotation[1],
                    0,
                    0
                ])
                __pt = inverse_transformation(_pt, cover_position, cover_rotation)

                if (__pt[0] >= -obj.front_behind_size[0]/2 - AFFORDACE_PROXIMITY_THRES and
                    __pt[0] <= obj.front_behind_size[0]/2 + AFFORDACE_PROXIMITY_THRES and 
                    __pt[1] >= obj.front_behind_size[1]/8 - AFFORDACE_PROXIMITY_THRES and
                    __pt[1] <= obj.front_behind_size[1]/2 + AFFORDACE_PROXIMITY_THRES and
                    __pt[2] >= -obj.front_behind_size[2]/2 - AFFORDACE_PROXIMITY_THRES and
                    __pt[2] <= obj.front_behind_size[2]/2 + AFFORDACE_PROXIMITY_THRES):
                    return True
                

            if (obj.has_cover[2] == 1):               # left
                cover_position = np.array([
                    -obj.cover_separation[1] / 2 - obj.left_right_size[1] * np.sin(obj.cover_rotation[2]) / 2, 
                    obj.left_right_size[1] * np.cos(obj.cover_rotation[2]) / 2, 
                    0
                ])
                cover_rotation = np.array([
                    0, 
                    0,
                    obj.cover_rotation[2],
                ])
                __pt = inverse_transformation(_pt, cover_position, cover_rotation)

                if (__pt[0] >= -obj.left_right_size[0]/2 - AFFORDACE_PROXIMITY_THRES and
                    __pt[0] <= obj.left_right_size[0]/2 + AFFORDACE_PROXIMITY_THRES and 
                    __pt[1] >= obj.left_right_size[1]/8 - AFFORDACE_PROXIMITY_THRES and
                    __pt[1] <= obj.left_right_size[1]/2 + AFFORDACE_PROXIMITY_THRES and
                    __pt[2] >= -obj.left_right_size[2]/2 - AFFORDACE_PROXIMITY_THRES and
                    __pt[2] <= obj.left_right_size[2]/2 + AFFORDACE_PROXIMITY_THRES):
                    return True
                

            if (obj.has_cover[3] == 1):               # right
                cover_position = np.array([
                    obj.cover_separation[1] / 2 - obj.left_right_size[1] * np.sin(obj.cover_rotation[3]) / 2, 
                    obj.left_right_size[1] * np.cos(obj.cover_rotation[3]) / 2, 
                    0
                ])
                cover_rotation = np.array([
                    0, 
                    0,
                    obj.cover_rotation[3],
                ])
                __pt = inverse_transformation(_pt, cover_position, cover_rotation)

                if (__pt[0] >= -obj.left_right_size[0]/2 - AFFORDACE_PROXIMITY_THRES and
                    __pt[0] <= obj.left_right_size[0]/2 + AFFORDACE_PROXIMITY_THRES and 
                    __pt[1] >= obj.left_right_size[1]/8 - AFFORDACE_PROXIMITY_THRES and
                    __pt[1] <= obj.left_right_size[1]/2 + AFFORDACE_PROXIMITY_THRES and
                    __pt[2] >= -obj.left_right_size[2]/2 - AFFORDACE_PROXIMITY_THRES and
                    __pt[2] <= obj.left_right_size[2]/2 + AFFORDACE_PROXIMITY_THRES):
                    return True
                
            return False

        return False
    
    return is_affordance(obj, pt)


def part_pose(obj):
    RT = transformation_matrix(obj.position, obj.rotation)
    return RT


def _pose_from_parameters(parameters):
    position = parameters.get("position", [0, 0, 0])
    rotation = [value / 180 * np.pi for value in parameters.get("rotation", [0, 0, 0])]
    return transformation_matrix(position, rotation)


def _part_semantic_name(concept_name):
    if concept_name == "world":
        return "world"
    if concept_name == "Cuboidal_Body":
        return "body"
    if concept_name in ("Regular_Cover", "Fourfold_Cover") or concept_name.startswith("Fourfold_Cover_id"):
        return "cover"
    if concept_name == "Cuboidal_Leg":
        return "leg"
    return concept_name


def _joint(parent, child, joint_type, origin, **extra):
    result = {
        "parent": _part_semantic_name(parent),
        "child": _part_semantic_name(child),
        "parent_concept": parent,
        "child_concept": child,
        "type": joint_type,
        "origin": origin,
    }
    result.update(extra)
    return result


def _fourfold_joint(parameters, cover_index, body_pose, cover_pose):
    rotations = np.deg2rad(parameters["cover_rotation"])
    separation = parameters["cover_separation"]
    front_behind_size = parameters["front_behind_size"]
    left_right_size = parameters["left_right_size"]

    if cover_index == 0:
        mesh_position = [
            0,
            front_behind_size[1] * np.cos(rotations[0]) / 2,
            separation[0] / 2 + front_behind_size[1] * np.sin(rotations[0]) / 2,
        ]
        mesh_rotation = [rotations[0], 0, 0]
        axis = [1, 0, 0]
        lower, upper = -rotations[0] - np.pi / 2, np.pi - rotations[0]
        cover_length = front_behind_size[1]
    elif cover_index == 1:
        mesh_position = [
            0,
            front_behind_size[1] * np.cos(rotations[1]) / 2,
            -separation[0] / 2 + front_behind_size[1] * np.sin(rotations[1]) / 2,
        ]
        mesh_rotation = [rotations[1], 0, 0]
        axis = [1, 0, 0]
        lower, upper = -rotations[1] - np.pi, np.pi / 2 - rotations[1]
        cover_length = front_behind_size[1]
    elif cover_index == 2:
        mesh_position = [
            -separation[1] / 2 - left_right_size[1] * np.sin(rotations[2]) / 2,
            left_right_size[1] * np.cos(rotations[2]) / 2,
            0,
        ]
        mesh_rotation = [0, 0, rotations[2]]
        axis = [0, 0, 1]
        lower, upper = -rotations[2] - np.pi / 2, np.pi - rotations[2]
        cover_length = left_right_size[1]
    else:
        mesh_position = [
            separation[1] / 2 - left_right_size[1] * np.sin(rotations[3]) / 2,
            left_right_size[1] * np.cos(rotations[3]) / 2,
            0,
        ]
        mesh_rotation = [0, 0, rotations[3]]
        axis = [0, 0, 1]
        lower, upper = -rotations[3] - np.pi, np.pi / 2 - rotations[3]
        cover_length = left_right_size[1]

    pose_adjust = transformation_matrix(mesh_position, mesh_rotation)
    pose_adjust = pose_adjust @ transformation_matrix([0, -cover_length / 2, 0], [0, 0, 0])
    return _joint(
        "Cuboidal_Body", f"Fourfold_Cover_id{cover_index}", "revolute",
        np.linalg.inv(body_pose) @ cover_pose @ pose_adjust,
        axis=axis, limit={"lower": float(lower), "upper": float(upper)},
    )


def _regular_cover_joint(body_parameters, cover_parameters, body_pose, cover_pose):
    hx = body_parameters["top_size"][0] / 2
    hz = body_parameters["top_size"][1] / 2
    hy = body_parameters["height"][0] / 2
    ox, oz = body_parameters["top_bottom_offset"]
    body_edges = [
        (np.array([ox, hy, oz - hz, 1]), [1, 0, 0]),
        (np.array([ox, hy, oz + hz, 1]), [1, 0, 0]),
        (np.array([ox - hx, hy, oz, 1]), [0, 0, 1]),
        (np.array([ox + hx, hy, oz, 1]), [0, 0, 1]),
    ]

    outer_size = cover_parameters["outer_size"]
    cover_edges = [
        ([0, 0, 0], [1, 0, 0]),
        ([0, 0, outer_size[2]], [1, 0, 0]),
        ([-outer_size[0] / 2, 0, outer_size[2] / 2], [0, 0, 1]),
        ([outer_size[0] / 2, 0, outer_size[2] / 2], [0, 0, 1]),
    ]

    minimum = None
    for cover_edge, cover_axis in cover_edges:
        cover_edge_world = (cover_pose @ np.append(cover_edge, 1))[:3]
        for body_edge, _ in body_edges:
            body_edge_world = (body_pose @ body_edge)[:3]
            distance = np.linalg.norm(cover_edge_world - body_edge_world)
            if minimum is None or distance < minimum[0]:
                minimum = distance, cover_axis, body_edge_world

    _, axis, body_edge_world = minimum
    hinge_in_cover = (np.linalg.inv(cover_pose) @ np.append(body_edge_world, 1))[:3]
    pose_adjust = transformation_matrix(hinge_in_cover, [0, 0, 0])
    body_y = body_pose[:3, 1]
    cover_y = cover_pose[:3, 1]
    current_angle = np.arccos(np.clip(np.dot(body_y, cover_y), -1, 1))
    test_rotation = transformation_matrix([0, 0, 0], np.asarray(axis) * 0.1)
    tested_pose = cover_pose @ pose_adjust @ test_rotation @ np.linalg.inv(pose_adjust)
    tested_angle = np.arccos(np.clip(np.dot(body_y, tested_pose[:3, 1]), -1, 1))
    if tested_angle > current_angle:
        lower, upper = -current_angle, np.pi - current_angle
    else:
        lower, upper = -np.pi + current_angle, current_angle

    return _joint(
        "Cuboidal_Body", "Regular_Cover", "revolute",
        np.linalg.inv(body_pose) @ cover_pose @ pose_adjust,
        axis=axis, limit={"lower": float(lower), "upper": float(upper)},
    )


def _articulation_from_asset(asset):
    if not isinstance(asset, dict):
        raise TypeError("Each PKL asset must be a dictionary")
    concepts = copy.deepcopy(asset.get("conceptualization"))
    if not isinstance(concepts, list):
        raise ValueError("PKL asset must contain a conceptualization list")

    bodies = [concept for concept in concepts if concept.get("template") == "Cuboidal_Body"]
    if len(bodies) != 1:
        raise ValueError("Box articulation requires exactly one Cuboidal_Body")
    body_parameters = bodies[0]["parameters"]
    body_pose = _pose_from_parameters(body_parameters)

    joints = [_joint("world", "Cuboidal_Body", "fixed", body_pose)]
    for concept in concepts:
        template = concept.get("template")
        parameters = concept.get("parameters", {})
        if template == "Cuboidal_Body":
            continue
        if template == "Fourfold_Cover":
            parameters["cover_separation"] = [value + 0.0001 for value in parameters["cover_separation"]]
            cover_pose = _pose_from_parameters(parameters)
            if len(parameters["has_cover"]) != 4:
                raise ValueError("Fourfold_Cover.has_cover must contain four values")
            for cover_index, enabled in enumerate(parameters["has_cover"]):
                if enabled == 1:
                    joints.append(_fourfold_joint(parameters, cover_index, body_pose, cover_pose))
        elif template == "Regular_Cover":
            position = list(parameters.get("position", [0, 0, 0]))
            position[1] += 0.0001
            position[2] -= 0.0001
            parameters["position"] = position
            cover_pose = _pose_from_parameters(parameters)
            joints.append(_regular_cover_joint(body_parameters, parameters, body_pose, cover_pose))
        elif template == "Cuboidal_Leg":
            joints.append(_joint(
                "Cuboidal_Body", "Cuboidal_Leg", "fixed",
                np.linalg.inv(body_pose) @ _pose_from_parameters(parameters),
            ))
        else:
            raise ValueError(f"Unsupported Box template: {template}")

    result = {"joints": joints}
    if "id" in asset:
        result = {"id": asset["id"], **result}
    return result


def get_articulation(pkl_path, output_format="world"):
    """Return Box articulation knowledge in world or parent-relative joint format."""
    if output_format not in ("world", "joint"):
        raise ValueError("output_format must be 'world' or 'joint'")
    with open(pkl_path, "rb") as file:
        data = pickle.load(file)

    def format_asset(asset):
        result = _articulation_from_asset(asset)
        if output_format == "joint":
            return result
        converted = convert_articulation_frame(result["joints"])
        world_result = {"joints": converted}
        if "id" in result:
            world_result = {"id": result["id"], **world_result}
        return world_result

    if isinstance(data, dict):
        return format_asset(data)
    if isinstance(data, list):
        return [format_asset(asset) for asset in data]
    raise TypeError("PKL root must be an asset dictionary or a list of asset dictionaries")


def _compose_local_grasp_spec(obj, local_pos, approach_dir, finger_closing_dir, grasp_width, manip_params_size, local_force_direction=None):
    obj_rot_mat = Rot.from_euler('xyz', obj.rotation).as_matrix()
    obj_pos = np.array(obj.position)

    T_obj_world = np.eye(4)
    T_obj_world[:3, :3] = obj_rot_mat
    T_obj_world[:3, 3] = obj_pos

    T_local = build_transformation_matrix(approach_dir, finger_closing_dir, local_pos)
    T_world = T_obj_world @ T_local

    world_pos = T_world[:3, 3]
    world_rot_mat = T_world[:3, :3]
    world_quat = Rot.from_matrix(world_rot_mat).as_quat()

    world_approach = world_rot_mat @ np.array([0.0, 0.0, 1.0])
    world_closing = world_rot_mat @ np.array([1.0, 0.0, 0.0])

    spec = {
        "world_transformation_matrix": T_world,
        "world_position": world_pos,
        "world_rotation": world_quat,  # quaternion [x, y, z, w]
        "world_approach_direction": world_approach,
        "world_finger_closing_direction": world_closing,
        "grasp_width": grasp_width,
        "manip_params_size": manip_params_size,
    }

    if local_force_direction is not None:
        world_force = obj_rot_mat @ np.array(local_force_direction)
        spec["world_force_direction"] = world_force / (np.linalg.norm(world_force) + 1e-12)

    return spec


def _compose_child_grasp_spec(obj, child_pos, child_rot, child_local_pos, child_local_approach, child_local_closing, grasp_width, manip_params_size, child_local_force=None):
    T_child_parent = transformation_matrix(child_pos, child_rot)
    T_grasp_child = build_transformation_matrix(
        child_local_approach,
        child_local_closing,
        child_local_pos,
    )
    T_grasp_parent = T_child_parent @ T_grasp_child

    parent_rot_mat = Rot.from_euler('xyz', obj.rotation).as_matrix()
    T_parent_world = np.eye(4)
    T_parent_world[:3, :3] = parent_rot_mat
    T_parent_world[:3, 3] = np.array(obj.position)

    T_world = T_parent_world @ T_grasp_parent
    world_rot_mat = T_world[:3, :3]
    world_quat = Rot.from_matrix(world_rot_mat).as_quat()

    spec = {
        "world_transformation_matrix": T_world,
        "world_position": T_world[:3, 3],
        "world_rotation": world_quat,  # quaternion [x, y, z, w]
        "world_approach_direction": world_rot_mat @ np.array([0.0, 0.0, 1.0]),
        "world_finger_closing_direction": world_rot_mat @ np.array([1.0, 0.0, 0.0]),
        "grasp_width": grasp_width,
        "manip_params_size": manip_params_size,
    }

    if child_local_force is not None:
        force_parent = T_child_parent[:3, :3] @ np.array(child_local_force)
        world_force = parent_rot_mat @ force_parent
        spec["world_force_direction"] = world_force / (np.linalg.norm(world_force) + 1e-12)

    return spec


def _select_fourfold_cover_index(obj, cover_rot_ratio):
    active_indices = [index for index, value in enumerate(obj.has_cover) if value == 1]
    if not active_indices:
        raise ValueError("Fourfold_Cover has no enabled cover segment")

    cover_selector = np.clip((cover_rot_ratio + 1.0) / 2.0, 0.0, 1.0 - 1e-12)
    return active_indices[int(cover_selector * len(active_indices))]


def _fourfold_cover_pose(obj, selected_index):
    if selected_index == 0:
        return (
            np.array([
                0.0,
                obj.front_behind_size[1] * np.cos(obj.cover_rotation[0]) / 2,
                obj.cover_separation[0] / 2 + obj.front_behind_size[1] * np.sin(obj.cover_rotation[0]) / 2,
            ]),
            np.array([obj.cover_rotation[0], 0.0, 0.0]),
        )

    if selected_index == 1:
        return (
            np.array([
                0.0,
                obj.front_behind_size[1] * np.cos(obj.cover_rotation[1]) / 2,
                -obj.cover_separation[0] / 2 + obj.front_behind_size[1] * np.sin(obj.cover_rotation[1]) / 2,
            ]),
            np.array([obj.cover_rotation[1], 0.0, 0.0]),
        )

    if selected_index == 2:
        return (
            np.array([
                -obj.cover_separation[1] / 2 - obj.left_right_size[1] * np.sin(obj.cover_rotation[2]) / 2,
                obj.left_right_size[1] * np.cos(obj.cover_rotation[2]) / 2,
                0.0,
            ]),
            np.array([0.0, 0.0, obj.cover_rotation[2]]),
        )

    return (
        np.array([
            obj.cover_separation[1] / 2 - obj.left_right_size[1] * np.sin(obj.cover_rotation[3]) / 2,
            obj.left_right_size[1] * np.cos(obj.cover_rotation[3]) / 2,
            0.0,
        ]),
        np.array([0.0, 0.0, obj.cover_rotation[3]]),
    )


def get_grasp_spec(obj, manipulation_params=None):
    if isinstance(obj, Regular_Cover):
        trans_ratio, rot_ratio = manipulation_params if manipulation_params is not None else (0.0, 0.0)
        local_rotation = Rot.from_euler("x", rot_ratio * np.pi / 10, degrees=False).as_matrix() @ np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
            ]
        )
        finger_closing_dir = local_rotation[:, 0]
        approach_dir = local_rotation[:, 2]
        edge_position = np.array(
            [
                trans_ratio * obj.outer_size[0] / 2 * 0.8,
                obj.inner_size[1] / 2,
                obj.outer_size[2],
            ]
        )
        local_position = edge_position - approach_dir * 0.00

        return _compose_local_grasp_spec(
            obj,
            local_position,
            approach_dir,
            finger_closing_dir,
            obj.outer_size[0],
            2,
            local_force_direction=np.array([0.0, 1.0, 0.0]),
        )

    if isinstance(obj, Fourfold_Cover):
        trans_ratio, rot_ratio, cover_rot_ratio = manipulation_params if manipulation_params is not None else (0.0, 0.0, 0.0)
        selected_index = _select_fourfold_cover_index(obj, cover_rot_ratio)
        cover_position, cover_rotation = _fourfold_cover_pose(obj, selected_index)

        if selected_index in (0, 1):
            child_local_position = np.array(
                [
                    trans_ratio * obj.front_behind_size[0] / 2 * 0.8,
                    obj.front_behind_size[1] / 2 + 0.00,
                    0.0,
                ]
            )
            child_local_rotation = Rot.from_euler("x", rot_ratio * np.pi / 10, degrees=False).as_matrix() @ np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0],
                    [0.0, 1.0, 0.0],
                ]
            )
            child_local_closing = child_local_rotation[:, 0]
            child_local_approach = child_local_rotation[:, 2]
            child_local_force = np.array([0.0, 0.0, 1.0 if selected_index == 0 else -1.0])
            grasp_width = obj.front_behind_size[0]
        else:
            child_local_position = np.array(
                [
                    0.0,
                    obj.left_right_size[1] / 2 + 0.00,
                    trans_ratio * obj.left_right_size[2] / 2 * 0.8,
                ]
            )
            child_local_rotation = Rot.from_euler("z", rot_ratio * np.pi / 10, degrees=False).as_matrix() @ np.array(
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, -1.0],
                    [-1.0, 0.0, 0.0],
                ]
            )
            child_local_closing = child_local_rotation[:, 0]
            child_local_approach = child_local_rotation[:, 2]
            child_local_force = np.array([1.0 if selected_index == 2 else -1.0, 0.0, 0.0])
            grasp_width = obj.left_right_size[2]

        return _compose_child_grasp_spec(
            obj,
            cover_position,
            cover_rotation,
            child_local_position,
            child_local_approach,
            child_local_closing,
            grasp_width,
            3,
            child_local_force=child_local_force,
        )

    return None
