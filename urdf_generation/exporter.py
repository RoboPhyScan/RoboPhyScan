from exporter_utils import *
from concept_template import *
from knowledge_utils import *
from knowledge_definitions import *
from urdf_knowledge import URDF_Knowledge_Definitions
import pickle
import numpy as np
import trimesh

template_name = "Bottle"
use_raw_obj = True
# Joint configuration for the lid. 
# Defaults to prismatic as requested. 
# Can be set to ["prismatic", "revolute"] for screw-like motion if supported by the downstream definition.
lid_joint_config = ["prismatic", "revolute"]

class Bottle:
    def __init__(self, file_name, mesh, output_dir, raw_obj_dir=None):
        self.file_name = file_name
        self.raw_obj_dir = raw_obj_dir
        self.output_dir = output_dir
        self.parts = []
        self.mesh_data = mesh
        self.fix_body_pose = mesh["fixed_body_pose"]["pose"] if "fixed_body_pose" in mesh else np.eye(4)
        self.fix_body_pos = mesh["fixed_body_pose"]["pos"] if "fixed_body_pose" in mesh else [0, 0, 0]
        self.fix_body_rot = mesh["fixed_body_pose"]["rot"] if "fixed_body_pose" in mesh else [0, 0, 0]

        self.body_templates = {"Multilevel_Body"}
        self.lid_templates = {"Cylindrical_Lid"}
        self.lid_child_templates = {"Round_U_Handle", "Cylindrical_Button"}
        self.connector_templates = {"Cylindrical_Connector"}
        self.has_lid_accessories = any(
            c["template"] in self.lid_child_templates for c in mesh["conceptualization"]
        )
        template_seen = {}
        
        for c in mesh["conceptualization"]:
            module = eval(c["template"])
            # Instantiate component
            component = module(**c["parameters"])
            component.template_name = c["template"]
            if component.template_name in self.connector_templates:
                component.radius = c["parameters"].get("radius", [0])
                component.thickness = c["parameters"].get("thickness", [0])
            component.template_occurrence_index = template_seen.get(component.template_name, 0)
            template_seen[component.template_name] = component.template_occurrence_index + 1
            component.actual_poses = []
            
            # Load raw meshes if requested
            if raw_obj_dir is not None:
                component.raw_obj_meshes = find_obj_in_folder(self.raw_obj_dir, component.template_name)
                for raw_mesh in component.raw_obj_meshes:
                    pos, rot, order, offset_first = self._inverse_global_transform(component)
                    raw_mesh.metadata["raw_obj_vertex_transform"] = {

                        "position": pos,

                        "rotation": rot,

                        "rotation_order": order,

                        "offset_first": offset_first,

                    }

                    raw_mesh.vertices = apply_transformation(
                        raw_mesh.vertices,
                        pos,
                        rot,
                        rotation_order=order,
                        offset_first=offset_first
                    )
                    raw_mesh = trimesh.Trimesh(raw_mesh.vertices, raw_mesh.faces)

            pos, rot, order, offset_first = self._inverse_global_transform(component)
            component.vertices = apply_transformation(
                component.vertices,
                pos,
                rot,
                rotation_order=order,
                offset_first=offset_first
            )
            component.overall_obj_mesh = trimesh.Trimesh(component.vertices, component.faces)
            self.parts.append(component)
        
        # Enforce order so parent links are emitted before their children.
        self.order = [
            "Multilevel_Body",
            "Cylindrical_Lid",
            "Cylindrical_Connector",
            "Round_U_Handle",
            "Cylindrical_Button",
        ]
        self.base_lid_sort_occurrence_index = self._base_lid_occurrence_index_for_sort()
        self.parts.sort(key=self._part_sort_key)

        self.urdf_knowledge = URDF_Knowledge_Definitions(
            template_name = template_name,
            output_dir = output_dir
        )
        
        # Identify parent-child hierarchy
        self.parent_id = []
        body_idx = -1
        lid_indices = []
        
        # Find the main body index
        for i, part in enumerate(self.parts):
            if part.template_name in self.body_templates:
                body_idx = i
                break
        for i, part in enumerate(self.parts):
            if part.template_name in self.lid_templates:
                lid_indices.append(i)
        self.connector_indices = [
            i for i, part in enumerate(self.parts)
            if part.template_name in self.connector_templates
        ]
        self.body_idx = body_idx
        self.lid_indices = lid_indices
        self.is_thermos_lid = (
            len(lid_indices) >= 2
            and (self.has_lid_accessories or len(self.connector_indices) > 0)
        )
        self.lid_base_idx, self.lid_cover_idx = self._classify_lid_indices(lid_indices)
        self.lid_cover_connector_idx = (
            self._nearest_connector_idx(self.parts[self.lid_cover_idx])
            if self.lid_cover_idx != -1 else -1
        )
        
        all_miny = []
        global_min_y = 0.0

        for i, part in enumerate(self.parts):
            # Establish Hierarchy
            if body_idx != -1:
                if i == body_idx:
                    self.parent_id.append(None) # Root
                elif (
                    self.is_thermos_lid
                    and i == self.lid_cover_idx
                    and self.lid_cover_connector_idx != -1
                ):
                    self.parent_id.append(self.lid_cover_connector_idx)
                elif self.is_thermos_lid and i == self.lid_cover_idx:
                    self.parent_id.append(self.lid_base_idx)
                elif part.template_name in self.connector_templates:
                    self.parent_id.append(self._nearest_below_part_idx(i))
                elif part.template_name in self.lid_child_templates and self.lid_base_idx != -1:
                    self.parent_id.append(self.lid_base_idx)
                else:
                    self.parent_id.append(body_idx) # Child of Body
            else:
                # Fallback: First item is root
                if i == 0:
                    self.parent_id.append(None)
                else:
                    self.parent_id.append(0)

            # Collect vertices for ground offset calculation
            if raw_obj_dir is not None and hasattr(part, 'raw_obj_meshes') and part.raw_obj_meshes:
                verts = np.array([])
                for m in part.raw_obj_meshes:
                    verts = np.vstack((verts, np.asarray(m.vertices))) if verts.size else np.asarray(m.vertices)
            else:
                verts = np.asarray(part.overall_obj_mesh.vertices)
            
            if verts.size > 0:
                order, offset_first = self._global_transform_config(part)
                verts = apply_transformation(
                    verts,
                    part.position,
                    part.rotation,
                    rotation_order=order,
                    offset_first=offset_first
                )
                verts = apply_transformation(
                    verts,
                    self.fix_body_pos,
                    self.fix_body_rot,
                    rotation_order="ZYX",
                    offset_first=True
                )
                all_miny.append(np.min(verts[:, 1]))
        
        if len(all_miny) > 0:
            global_min_y = min(all_miny)
        self.body_offset = global_min_y

        self.link_actual_poses = {}
        self.part_link_names = {}

    def _part_sort_key(self, part):
        base_order = self.order.index(part.template_name) if part.template_name in self.order else 999
        if part.template_name == "Cylindrical_Lid":
            base_occurrence_index = getattr(self, "base_lid_sort_occurrence_index", 0)
            lid_phase = 0 if part.template_occurrence_index == base_occurrence_index else 2
            return (base_order, lid_phase, part.template_occurrence_index)
        if part.template_name in self.connector_templates:
            return (1, 1, part.template_occurrence_index)
        return (base_order, 0, part.template_occurrence_index)

    def _base_lid_occurrence_index_for_sort(self):
        lid_parts = [part for part in self.parts if part.template_name == "Cylindrical_Lid"]
        if len(lid_parts) == 0:
            return 0

        for part in lid_parts:
            if self._is_ring_only_lid(part):
                return part.template_occurrence_index

        return lid_parts[0].template_occurrence_index

    def _is_ring_only_lid(self, part):
        if part.template_name != "Cylindrical_Lid":
            return False
        return abs(float(part.outer_size[2]) - float(part.inner_size[2])) < 1e-6

    def _classify_lid_indices(self, lid_indices):
        if len(lid_indices) == 0:
            return -1, -1

        base_idx = lid_indices[0]
        for idx in lid_indices:
            if self._is_ring_only_lid(self.parts[idx]):
                base_idx = idx
                break

        cover_idx = -1
        for idx in lid_indices:
            if idx != base_idx:
                cover_idx = idx
                break

        return base_idx, cover_idx

    def _global_transform_config(self, part):
        if part.template_name in {"Cylindrical_Button", "Cylindrical_Connector"}:
            return "XYZ", True
        return "XYZ", False

    def _inverse_global_transform(self, part):
        order, offset_first = self._global_transform_config(part)
        pos = [-x for x in part.position.copy()]
        rot = [-x for x in part.rotation.copy()]
        return pos, rot, order[::-1], (not offset_first)

    def _part_pose(self, part):
        order, offset_first = self._global_transform_config(part)
        if order == "XYZ" and not offset_first:
            return part_pose(part)
        return pose_from_apply(
            part.position,
            part.rotation,
            rotation_order=order,
            offset_first=offset_first
        )

    def _mesh_for_part(self, part):
        if self.raw_obj_dir is not None and hasattr(part, 'raw_obj_meshes') and part.raw_obj_meshes:
            if len(part.raw_obj_meshes) > part.template_occurrence_index:
                return part.raw_obj_meshes[part.template_occurrence_index]
            return part.raw_obj_meshes[0]
        return part.overall_obj_mesh

    def _axis_in_part_frame(self, part, world_axis):
        part_pose = self._part_pose(part)
        return self._axis_in_pose_frame(part_pose, world_axis)

    def _axis_in_pose_frame(self, pose, world_axis):
        axis = pose[:3, :3].T @ np.asarray(world_axis, dtype=float)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-8:
            return [0, 1, 0]
        return (axis / axis_norm).tolist()

    def _part_center(self, part):
        return self._part_pose(part)[:3, 3]

    def _nearest_below_part_idx(self, part_idx):
        if part_idx < 0 or part_idx >= len(self.parts):
            return self.lid_base_idx if self.lid_base_idx != -1 else self.body_idx

        part = self.parts[part_idx]
        part_center = self._part_center(part)
        body_axis = self._body_axis_world()
        candidates = []

        for idx, candidate in enumerate(self.parts):
            if idx == part_idx or candidate.template_name in self.connector_templates:
                continue
            if idx >= part_idx:
                continue

            candidate_center = self._part_center(candidate)
            height_delta = float(np.dot(candidate_center - part_center, body_axis))
            if height_delta > 1e-6:
                continue
            candidates.append((np.linalg.norm(candidate_center - part_center), idx))

        if len(candidates) == 0:
            if self.lid_base_idx != -1 and self.lid_base_idx < part_idx:
                return self.lid_base_idx
            if self.body_idx != -1 and self.body_idx < part_idx:
                return self.body_idx
            return 0 if part_idx != 0 else None

        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def _nearest_connector_idx(self, part):
        if len(self.connector_indices) == 0:
            return -1

        part_center = self._part_center(part)
        best_idx = -1
        best_distance = np.inf
        for idx in self.connector_indices:
            distance = np.linalg.norm(self._part_center(self.parts[idx]) - part_center)
            if distance < best_distance:
                best_idx = idx
                best_distance = distance
        return best_idx

    def _body_axis_world(self):
        if self.body_idx == -1:
            return np.array([0.0, 1.0, 0.0])

        body_pose = self._part_pose(self.parts[self.body_idx])
        axis = body_pose[:3, :3] @ np.array([0.0, 1.0, 0.0])
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-8:
            return np.array([0.0, 1.0, 0.0])
        return axis / axis_norm

    def _base_lid_axis(self, base_part):
        return self._axis_in_part_frame(base_part, self._body_axis_world())

    def _away_from_body_axis_world(self, part):
        body_axis = self._body_axis_world()
        if self.body_idx == -1:
            return body_axis

        body_center = self._part_pose(self.parts[self.body_idx])[:3, 3]
        part_center = self._part_pose(part)[:3, 3]
        if np.dot(part_center - body_center, body_axis) < 0.0:
            return -body_axis
        return body_axis

    def _base_lift_axis_limit(self, part, axis_pose, lift):
        body_axis = self._body_axis_world()
        away_axis = self._away_from_body_axis_world(part)
        joint_axis = self._axis_in_pose_frame(axis_pose, body_axis)

        if np.dot(away_axis, body_axis) >= 0.0:
            limit = {"lower": 0.0, "upper": lift, "effort": 10, "velocity": 2}
        else:
            limit = {"lower": -lift, "upper": 0.0, "effort": 10, "velocity": 2}
        return joint_axis, limit

    def _pose_on_body_axis(self, part):
        pose = self._part_pose(part)
        if self.body_idx == -1:
            return pose

        body_pose = self._part_pose(self.parts[self.body_idx])
        body_center = body_pose[:3, 3]
        body_axis = self._body_axis_world()
        part_center = pose[:3, 3]
        projected_center = body_center + body_axis * np.dot(part_center - body_center, body_axis)

        axis_pose = pose.copy()
        axis_pose[:3, 3] = projected_center
        return axis_pose

    def _fallback_cover_offset(self, part):
        hinge_y = 0.0
        hinge_z = -max(float(part.outer_size[0]), float(part.outer_size[1]))
        return transformation_matrix([0, hinge_y, hinge_z], [0, 0, 0])

    def _accessory_front_direction_in_base(self, base_part, accessory_parts):
        if len(accessory_parts) == 0:
            return None

        base_pose = self._part_pose(base_part)
        base_pose_inv = np.linalg.inv(base_pose)
        front_vectors = []

        for accessory_part in accessory_parts:
            accessory_pose = self._part_pose(accessory_part)
            accessory_center = np.append(accessory_pose[:3, 3], 1.0)
            accessory_center_base = base_pose_inv @ accessory_center
            front_vector = np.array(
                [accessory_center_base[0], 0.0, accessory_center_base[2]],
                dtype=float
            )
            if np.linalg.norm(front_vector) > 1e-8:
                front_vectors.append(front_vector)

        if len(front_vectors) == 0:
            return None

        front_direction = np.mean(front_vectors, axis=0)
        direction_norm = np.linalg.norm(front_direction)
        if direction_norm < 1e-8:
            return None
        return front_direction / direction_norm

    def _cover_hinge_direction_in_base(self, cover_part, base_part):
        base_pose = self._part_pose(base_part)
        cover_pose = self._part_pose(cover_part)
        cover_center_base = np.linalg.inv(base_pose) @ np.append(cover_pose[:3, 3], 1.0)
        hinge_direction = np.array(
            [cover_center_base[0], 0.0, cover_center_base[2]],
            dtype=float
        )
        direction_norm = np.linalg.norm(hinge_direction)
        if direction_norm < 1e-8:
            return None
        return hinge_direction / direction_norm

    def _hinge_transform_from_base(self, cover_part, base_part, accessory_parts):
        hinge_direction = self._cover_hinge_direction_in_base(cover_part, base_part)
        if hinge_direction is None:
            front_direction = self._accessory_front_direction_in_base(base_part, accessory_parts)
            if front_direction is None:
                return None
            hinge_direction = -front_direction

        hinge_radius = 0.98 * max(
            float(base_part.outer_size[0]),
            float(base_part.outer_size[1]),
            float(cover_part.outer_size[0]),
            float(cover_part.outer_size[1])
        )
        hinge_position_base = hinge_direction * hinge_radius
        base_pose = self._part_pose(base_part)
        cover_pose = self._part_pose(cover_part)
        hinge_position_world = (base_pose @ np.append(hinge_position_base, 1.0))[:3]

        ring_edge_y = -float(cover_part.outer_size[2]) / 2.0
        ring_edge_world = (cover_pose @ np.array([0.0, ring_edge_y, 0.0, 1.0]))[:3]
        cover_y_axis = cover_pose[:3, 1]
        hinge_position_world = (
            hinge_position_world
            + cover_y_axis * np.dot(ring_edge_world - hinge_position_world, cover_y_axis)
        )

        hinge_pose = np.eye(4)
        hinge_pose[:3, :3] = cover_pose[:3, :3]
        hinge_pose[:3, 3] = hinge_position_world
        return hinge_pose

    def _axis_rotation_matrix(self, axis, angle):
        axis = np.asarray(axis, dtype=float)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-8:
            return np.eye(3)
        axis = axis / axis_norm
        x, y, z = axis
        skew = np.array([
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0]
        ])
        return (
            np.eye(3) * np.cos(angle)
            + (1.0 - np.cos(angle)) * np.outer(axis, axis)
            + np.sin(angle) * skew
        )

    def _closed_cover_center(self, cover_part, base_part):
        base_pose = self._part_pose(base_part)
        y_offset = 0.5 * (
            float(base_part.outer_size[2]) - float(cover_part.outer_size[2])
        )
        return (base_pose @ np.array([0.0, y_offset, 0.0, 1.0]))[:3]

    def _cover_open_angle(self, cover_part, base_part):
        if base_part is None:
            return np.pi

        cover_pose = self._part_pose(cover_part)
        base_pose = self._part_pose(base_part)
        relative_rotation = cover_pose[:3, :3].T @ base_pose[:3, :3]
        x_angle = np.arctan2(relative_rotation[2, 1], relative_rotation[1, 1])
        if abs(float(x_angle)) < np.deg2rad(1.0):
            return np.pi
        return float(min(abs(x_angle), np.pi))

    def _cover_close_angle(self, cover_part, base_part, hinge_pose=None):
        open_angle = self._cover_open_angle(cover_part, base_part)
        if base_part is None or hinge_pose is None:
            return -open_angle

        cover_pose = self._part_pose(cover_part)
        hinge_axis = hinge_pose[:3, :3] @ np.array([1.0, 0.0, 0.0])
        hinge_axis_norm = np.linalg.norm(hinge_axis)
        if hinge_axis_norm < 1e-8:
            return -open_angle
        hinge_axis = hinge_axis / hinge_axis_norm

        cover_center = cover_pose[:3, 3]
        hinge_center = hinge_pose[:3, 3]
        target_center = self._closed_cover_center(cover_part, base_part)
        base_pose = self._part_pose(base_part)
        orientation_weight = max(
            float(base_part.outer_size[0]),
            float(base_part.outer_size[1]),
            float(cover_part.outer_size[0]),
            float(cover_part.outer_size[1])
        )

        candidates = [-open_angle, open_angle]
        best_angle = candidates[0]
        best_error = np.inf
        for angle in candidates:
            rotation = self._axis_rotation_matrix(hinge_axis, angle)
            closed_center = hinge_center + rotation @ (cover_center - hinge_center)
            closed_rotation = rotation @ cover_pose[:3, :3]
            center_error = np.linalg.norm(closed_center - target_center)
            orientation_error = (
                np.linalg.norm(closed_rotation[:, 1] - base_pose[:3, 1])
                + np.linalg.norm(closed_rotation[:, 2] - base_pose[:3, 2])
            )
            error = center_error + orientation_weight * orientation_error
            if error < best_error:
                best_angle = angle
                best_error = error
        return best_angle

    def _cover_limit(self, cover_part, base_part, hinge_pose=None):
        close_angle = self._cover_close_angle(cover_part, base_part, hinge_pose)
        lower, upper = sorted((0.0, close_angle))
        return {"lower": lower, "upper": upper, "effort": 10, "velocity": 2}

    def _button_lid_attachment_score(self, button_part, lid_part):
        lid_pose = self._part_pose(lid_part)
        button_center = np.append(self._part_pose(button_part)[:3, 3], 1.0)
        button_center_lid = np.linalg.inv(lid_pose) @ button_center

        radius = max(float(lid_part.outer_size[0]), float(lid_part.outer_size[1]))
        radial_dist = np.linalg.norm([button_center_lid[0], button_center_lid[2]])
        radial_excess = max(radial_dist - radius, 0.0)

        half_height = float(lid_part.outer_size[2]) / 2.0
        cap_distance = min(
            abs(float(button_center_lid[1]) - half_height),
            abs(float(button_center_lid[1]) + half_height)
        )

        return radial_excess + cap_distance

    def _lid_attachment_score(self, accessory_part, lid_part):
        if accessory_part.template_name == "Cylindrical_Button":
            return self._button_lid_attachment_score(accessory_part, lid_part)

        lid_pose = self._part_pose(lid_part)
        accessory_center = np.append(self._part_pose(accessory_part)[:3, 3], 1.0)
        accessory_center_lid = np.linalg.inv(lid_pose) @ accessory_center

        radius = max(float(lid_part.outer_size[0]), float(lid_part.outer_size[1]))
        radial_dist = np.linalg.norm([accessory_center_lid[0], accessory_center_lid[2]])
        radial_error = abs(radial_dist - radius)

        half_height = float(lid_part.outer_size[2]) / 2.0
        y_excess = max(abs(float(accessory_center_lid[1])) - half_height, 0.0)
        return radial_error + y_excess

    def _lid_accessory_parent_idx(self, accessory_part):
        candidates = []
        if self.lid_base_idx != -1:
            candidates.append(self.lid_base_idx)
        if self.lid_cover_idx != -1:
            candidates.append(self.lid_cover_idx)
        if len(candidates) == 0:
            return self.lid_base_idx

        best_idx = candidates[0]
        best_score = np.inf
        for idx in candidates:
            score = self._lid_attachment_score(accessory_part, self.parts[idx])
            if score < best_score:
                best_idx = idx
                best_score = score
        return best_idx

    def get_cylindrical_lid_cover_pose(self, cover_part, base_part=None, accessory_parts=None):
        hinge_pose = None
        if base_part is not None and accessory_parts is not None:
            hinge_pose = self._hinge_transform_from_base(cover_part, base_part, accessory_parts)
            if hinge_pose is not None:
                cover_pose = self._part_pose(cover_part)
                cover_offset = np.linalg.inv(cover_pose) @ hinge_pose
            else:
                cover_offset = self._fallback_cover_offset(cover_part)
        else:
            cover_offset = self._fallback_cover_offset(cover_part)

        joint_info = {
            "axis": [1, 0, 0],
            "limit": self._cover_limit(cover_part, base_part, hinge_pose)
        }
        return cover_offset, joint_info

    def get_button_pose(self, part):
        thickness = float(part.thickness[0])
        button_offset = transformation_matrix([0, 0, thickness / 2], [0, 0, 0])
        joint_info = {
            "axis": [1, 0, 0],
            "limit": {
                "lower": -np.deg2rad(10.0),
                "upper": np.deg2rad(10.0),
                "effort": 10,
                "velocity": 2,
            }
        }
        return button_offset, joint_info

    def get_connector_pose(self, part):
        thickness = float(part.thickness[0]) if hasattr(part, "thickness") else 0.0
        return transformation_matrix([0, 0, thickness / 2], [0, 0, 0])

    def _handle_initial_x_angle(self, part, parent_part=None):
        if parent_part is None:
            return float(part.rotation[0]) if hasattr(part, "rotation") else 0.0

        handle_pose = self._part_pose(part)
        parent_pose = self._part_pose(parent_part)
        relative_rotation = parent_pose[:3, :3].T @ handle_pose[:3, :3]
        return float(np.arctan2(relative_rotation[2, 1], relative_rotation[1, 1]))

    def _handle_outside_score(self, part, parent_part, sweep_angle):
        if parent_part is None or not hasattr(part, "vertical_length"):
            return -np.inf

        parent_pose = self._part_pose(parent_part)
        handle_pose = self._part_pose(part)
        handle_pose_parent = np.linalg.inv(parent_pose) @ handle_pose

        radial_direction = np.array(
            [handle_pose_parent[0, 3], 0.0, handle_pose_parent[2, 3]],
            dtype=float
        )
        radial_norm = np.linalg.norm(radial_direction)
        if radial_norm < 1e-8:
            return -np.inf
        radial_direction /= radial_norm

        reference_point = np.array([0.0, float(part.vertical_length[0]), 0.0, 1.0])
        min_outside_score = np.inf
        for fraction in (0.25, 0.5, 0.75):
            rotation = np.eye(4)
            rotation[:3, :3] = self._axis_rotation_matrix([1, 0, 0], sweep_angle * fraction)
            point_parent = (handle_pose_parent @ rotation @ reference_point)[:3]
            outside_score = float(np.dot(
                np.array([point_parent[0], 0.0, point_parent[2]]),
                radial_direction
            ))
            min_outside_score = min(min_outside_score, outside_score)

        return min_outside_score

    def _handle_outward_sweep_angle(self, part, parent_part, candidates):
        if len(candidates) == 1:
            return candidates[0]

        scored_candidates = [
            (self._handle_outside_score(part, parent_part, angle), angle)
            for angle in candidates
        ]
        scored_candidates.sort(key=lambda item: item[0], reverse=True)
        return scored_candidates[0][1]

    def _handle_limit(self, part, parent_part=None):
        initial_x_angle = self._handle_initial_x_angle(part, parent_part)
        initial_x_angle = float(np.arctan2(np.sin(initial_x_angle), np.cos(initial_x_angle)))
        if abs(initial_x_angle) < np.deg2rad(1.0):
            sweep_angle = self._handle_outward_sweep_angle(part, parent_part, [-np.pi, np.pi])
            lower, upper = sorted((0.0, sweep_angle))
            return {"lower": lower, "upper": upper, "effort": 10, "velocity": 2}

        sweep_angle = -initial_x_angle
        if abs(abs(sweep_angle) - np.pi) < np.deg2rad(1.0):
            sweep_angle = self._handle_outward_sweep_angle(part, parent_part, [-np.pi, np.pi])

        lower, upper = sorted((0.0, sweep_angle))
        return {
            "lower": lower,
            "upper": upper,
            "effort": 10,
            "velocity": 2,
        }

    def get_round_u_handle_pose(self, part, parent_part=None):
        joint_info = {
            "axis": [1, 0, 0],
            "limit": self._handle_limit(part, parent_part)
        }
        return np.eye(4), joint_info

    def _unique_name(self, base_name, used_names):
        if base_name not in used_names:
            used_names.add(base_name)
            return base_name

        idx = 0
        while f"{base_name}_id{idx}" in used_names:
            idx += 1
        name = f"{base_name}_id{idx}"
        used_names.add(name)
        return name
        
    def export_urdf(self):
        links, joints = [], []
        used_names = set()
        
        for i, part in enumerate(self.parts):
            # Determine Parent Name and Pose
            if self.parent_id[i] is None:
                parent_name = "world"
                parent_pose = transformation_matrix([0, 0, self.body_offset], [0, 0, 0])
            else:
                parent_part = self.parts[self.parent_id[i]]
                parent_name = self.part_link_names.get(self.parent_id[i], parent_part.template_name)
                parent_pose = parent_part.actual_poses[0] if len(parent_part.actual_poses) > 0 else self._part_pose(parent_part)

            # Select Mesh
            mesh = self._mesh_for_part(part)

            # Case 1: Body (Multilevel_Body)
            if part.template_name == "Multilevel_Body":
                link_name = self._unique_name(part.template_name, used_names)
                pose_adjust = np.eye(4)
                pose = self._part_pose(part)
                link, joint = self.urdf_knowledge.add_part(link_name,
                                        mesh,
                                        pose,
                                        pose_adjust=pose_adjust,
                                        fix_body_pose=self.fix_body_pose,
                                        parent_name=parent_name,
                                        parent_pose=parent_pose,
                                        joint_type="fixed",
                                        material='plastic')
                links.append(link)
                joints.append(joint)
                part.actual_poses.append(pose)
                self.part_link_names[i] = link_name
                self.link_actual_poses[link_name] = pose

            # Case 2: Lid (Cylindrical_Lid)
            elif part.template_name == "Cylindrical_Lid":
                if not self.is_thermos_lid:
                    enable_prismatic = "prismatic" in lid_joint_config
                    enable_revolute = "revolute" in lid_joint_config

                    if self.raw_obj_dir is None:
                        lid_mesh = mesh.copy()
                        lid_mesh.vertices *= 1.001
                        lid_mesh.apply_translation([0, 0.0001, 0])
                    else:
                        lid_mesh = mesh

                    parent_for_final = parent_name
                    parent_pose_for_final = parent_pose
                    part_base_pose = self._part_pose(part)

                    if enable_prismatic and enable_revolute:
                        inter_link_name = self._unique_name(f"{part.template_name}_virtual_prismatic", used_names)
                        limit_pris = {"lower": 0.0, "upper": 0.1, "effort": 10, "velocity": 2}
                        link_v, joint_v = self.urdf_knowledge.add_part(
                            link_name=inter_link_name,
                            mesh=None,
                            pose=part_base_pose,
                            pose_adjust=np.eye(4),
                            parent_name=parent_name,
                            parent_pose=parent_pose,
                            fix_body_pose=self.fix_body_pose,
                            joint_type="prismatic",
                            joint_axis=[0, 1, 0],
                            joint_limit=limit_pris,
                            material='plastic'
                        )
                        links.append(link_v)
                        joints.append(joint_v)
                        parent_for_final = inter_link_name
                        parent_pose_for_final = part_base_pose
                        self.link_actual_poses[inter_link_name] = part_base_pose

                        j_type = "revolute"
                        limit = {"lower": -np.pi, "upper": np.pi, "effort": 10, "velocity": 2}
                        axis = self._base_lid_axis(part)
                    elif enable_revolute:
                        j_type = "revolute"
                        limit = {"lower": -np.pi, "upper": np.pi, "effort": 10, "velocity": 2}
                        axis = self._base_lid_axis(part)
                    else:
                        j_type = "prismatic"
                        limit = {"lower": 0.0, "upper": 0.1, "effort": 10, "velocity": 2}
                        axis = [0, 1, 0]

                    link_name = self._unique_name(part.template_name, used_names)
                    pose_adjust = np.eye(4)
                    pose = part_base_pose @ pose_adjust
                    link, joint = self.urdf_knowledge.add_part(
                                            link_name,
                                            lid_mesh,
                                            pose,
                                            pose_adjust=pose_adjust,
                                            fix_body_pose=self.fix_body_pose,
                                            parent_name=parent_for_final,
                                            parent_pose=parent_pose_for_final,
                                            joint_type=j_type,
                                            joint_axis=axis,
                                            joint_limit=limit,
                                            material='plastic')
                    links.append(link)
                    joints.append(joint)
                    part.actual_poses.append(pose)
                    self.part_link_names[i] = link_name
                    self.link_actual_poses[link_name] = pose
                    continue

                part_base_pose = self._part_pose(part)

                lid_mesh = mesh
                if self.raw_obj_dir is None:
                    lid_mesh = mesh.copy()
                    lid_mesh.vertices *= 1.001
                    lid_mesh.apply_translation([0, 0.0001, 0])

                if i == self.lid_base_idx:
                    axis_pose = self._pose_on_body_axis(part)
                    axis_pose_adjust = np.linalg.inv(part_base_pose) @ axis_pose

                    inter_link_name = self._unique_name(f"{part.template_name}_virtual_prismatic", used_names)
                    lift = max(0.02, float(part.outer_size[2]) + 0.02)
                    lift_axis, limit_pris = self._base_lift_axis_limit(part, axis_pose, lift)
                    link_v, joint_v = self.urdf_knowledge.add_part(
                        link_name=inter_link_name,
                        mesh=None,
                        pose=axis_pose,
                        pose_adjust=np.eye(4),
                        parent_name=parent_name,
                        parent_pose=parent_pose,
                        fix_body_pose=self.fix_body_pose,
                        joint_type="prismatic",
                        joint_axis=lift_axis,
                        joint_limit=limit_pris,
                        material='plastic'
                    )
                    links.append(link_v)
                    joints.append(joint_v)

                    link_name = self._unique_name(f"{part.template_name}_base", used_names)
                    pose_adjust = axis_pose_adjust
                    pose = axis_pose
                    link, joint = self.urdf_knowledge.add_part(
                                        link_name,
                                        lid_mesh,
                                        pose,
                                        pose_adjust=pose_adjust,
                                        fix_body_pose=self.fix_body_pose,
                                        parent_name=inter_link_name,
                                        parent_pose=axis_pose,
                                        joint_type="revolute",
                                        joint_axis=self._base_lid_axis(part),
                                        joint_limit={"lower": -np.pi, "upper": np.pi, "effort": 10, "velocity": 2},
                                        material='plastic')
                    links.append(link)
                    joints.append(joint)
                    part.actual_poses.append(pose)
                    self.part_link_names[i] = link_name
                    self.link_actual_poses[inter_link_name] = axis_pose
                    self.link_actual_poses[link_name] = pose

                elif i == self.lid_cover_idx:
                    base_part = self.parts[self.lid_base_idx] if self.lid_base_idx != -1 else None
                    accessory_parts = [
                        p for p in self.parts
                        if p.template_name in self.lid_child_templates
                    ]
                    if self.lid_cover_connector_idx != -1:
                        connector_name = self.part_link_names.get(self.lid_cover_connector_idx)
                        if connector_name in self.link_actual_poses:
                            parent_name = connector_name
                            parent_pose = self.link_actual_poses[parent_name]
                        pose_adjust = np.eye(4)
                    else:
                        pose_adjust, cover_joint_info = self.get_cylindrical_lid_cover_pose(
                            part,
                            base_part=base_part,
                            accessory_parts=accessory_parts
                        )
                    pose = part_base_pose @ pose_adjust
                    link_name = self._unique_name(f"{part.template_name}_cover", used_names)
                    link, joint = self.urdf_knowledge.add_part(
                                        link_name,
                                        lid_mesh,
                                        pose,
                                        pose_adjust=pose_adjust,
                                        fix_body_pose=self.fix_body_pose,
                                        parent_name=parent_name,
                                        parent_pose=parent_pose,
                                        joint_type="fixed" if self.lid_cover_connector_idx != -1 else "revolute",
                                        joint_axis=(
                                            [0, 0, 1]
                                            if self.lid_cover_connector_idx != -1
                                            else cover_joint_info["axis"]
                                        ),
                                        joint_limit=(
                                            None
                                            if self.lid_cover_connector_idx != -1
                                            else cover_joint_info["limit"]
                                        ),
                                        material='plastic')
                    links.append(link)
                    joints.append(joint)
                    part.actual_poses.append(pose)
                    self.part_link_names[i] = link_name
                    self.link_actual_poses[link_name] = pose

                else:
                    link_name = self._unique_name(part.template_name, used_names)
                    pose_adjust = np.eye(4)
                    pose = part_base_pose @ pose_adjust
                    link, joint = self.urdf_knowledge.add_part(
                                        link_name,
                                        lid_mesh,
                                        pose,
                                        pose_adjust=pose_adjust,
                                        fix_body_pose=self.fix_body_pose,
                                        parent_name=parent_name,
                                        parent_pose=parent_pose,
                                        joint_type="fixed",
                                        material='plastic')
                    links.append(link)
                    joints.append(joint)
                    part.actual_poses.append(pose)
                    self.part_link_names[i] = link_name
                    self.link_actual_poses[link_name] = pose

            elif part.template_name == "Round_U_Handle":
                link_name = self._unique_name(part.template_name, used_names)

                lid_parent_idx = self._lid_accessory_parent_idx(part)
                lid_parent_part = (
                    self.parts[lid_parent_idx]
                    if lid_parent_idx is not None and lid_parent_idx >= 0
                    else None
                )
                pose_adjust, joint_info = self.get_round_u_handle_pose(part, lid_parent_part)
                pose = self._part_pose(part) @ pose_adjust

                lid_parent_name = self.part_link_names.get(lid_parent_idx)
                if lid_parent_name in self.link_actual_poses:
                    parent_name = lid_parent_name
                    parent_pose = self.link_actual_poses[parent_name]

                link, joint = self.urdf_knowledge.add_part(
                                        link_name,
                                        mesh,
                                        pose,
                                        pose_adjust=pose_adjust,
                                        fix_body_pose=self.fix_body_pose,
                                        parent_name=parent_name,
                                        parent_pose=parent_pose,
                                        joint_type="revolute",
                                        joint_axis=joint_info["axis"],
                                        joint_limit=joint_info["limit"],
                                        material='plastic')
                links.append(link)
                joints.append(joint)
                part.actual_poses.append(pose)
                self.part_link_names[i] = link_name
                self.link_actual_poses[link_name] = pose

            elif part.template_name == "Cylindrical_Connector":
                link_name = self._unique_name(part.template_name, used_names)
                pose_adjust = self.get_connector_pose(part)
                pose = self._part_pose(part) @ pose_adjust

                cover_part = self.parts[self.lid_cover_idx] if self.lid_cover_idx != -1 else None
                base_part = self.parts[self.lid_base_idx] if self.lid_base_idx != -1 else None
                if cover_part is not None and self.lid_cover_connector_idx == i:
                    accessory_parts = [
                        p for p in self.parts
                        if p.template_name in self.lid_child_templates
                    ]
                    _, joint_info = self.get_cylindrical_lid_cover_pose(
                        cover_part,
                        base_part=base_part,
                        accessory_parts=accessory_parts
                    )
                    joint_type = "revolute"
                    joint_axis = [0, 0, 1]
                    joint_limit = joint_info["limit"]
                else:
                    joint_type = "fixed"
                    joint_axis = [0, 0, 1]
                    joint_limit = None

                link, joint = self.urdf_knowledge.add_part(
                                        link_name,
                                        mesh,
                                        pose,
                                        pose_adjust=pose_adjust,
                                        fix_body_pose=self.fix_body_pose,
                                        parent_name=parent_name,
                                        parent_pose=parent_pose,
                                        joint_type=joint_type,
                                        joint_axis=joint_axis,
                                        joint_limit=joint_limit,
                                        material='plastic')
                links.append(link)
                joints.append(joint)
                part.actual_poses.append(pose)
                self.part_link_names[i] = link_name
                self.link_actual_poses[link_name] = pose

            elif part.template_name == "Cylindrical_Button":
                link_name = self._unique_name(part.template_name, used_names)
                pose_adjust, joint_info = self.get_button_pose(part)
                pose = self._part_pose(part) @ pose_adjust

                lid_parent_idx = self._lid_accessory_parent_idx(part)
                lid_parent_name = self.part_link_names.get(lid_parent_idx)
                if lid_parent_name in self.link_actual_poses:
                    parent_name = lid_parent_name
                    parent_pose = self.link_actual_poses[parent_name]

                link, joint = self.urdf_knowledge.add_part(
                                        link_name,
                                        mesh,
                                        pose,
                                        pose_adjust=pose_adjust,
                                        fix_body_pose=self.fix_body_pose,
                                        parent_name=parent_name,
                                        parent_pose=parent_pose,
                                        joint_type="revolute",
                                        joint_axis=joint_info["axis"],
                                        joint_limit=joint_info["limit"],
                                        material='plastic')
                links.append(link)
                joints.append(joint)
                part.actual_poses.append(pose)
                self.part_link_names[i] = link_name
                self.link_actual_poses[link_name] = pose

            else:
                link_name = self._unique_name(part.template_name, used_names)
                pose_adjust = np.eye(4)
                pose = self._part_pose(part)
                link, joint = self.urdf_knowledge.add_part(
                                        link_name,
                                        mesh,
                                        pose,
                                        pose_adjust=pose_adjust,
                                        fix_body_pose=self.fix_body_pose,
                                        parent_name=parent_name,
                                        parent_pose=parent_pose,
                                        joint_type="fixed",
                                        material='plastic')
                links.append(link)
                joints.append(joint)
                part.actual_poses.append(pose)
                self.part_link_names[i] = link_name
                self.link_actual_poses[link_name] = pose
        
        # Write URDF
        exporter = Export(links, joints, template_name, self.output_dir)
        exporter.write_urdf(self.file_name)
        print(f"URDF for {template_name} has been exported to {self.file_name}")

def adjust_pkl(data):
    # Adjust parameters to center the main reference part at (0,0,0)
    ref_pose = None
 
    for c in data['conceptualization']:
        if c['template'] == "Multilevel_Body": # Assuming the main body is the reference
            pos = np.array(c['parameters'].get('position', [0, 0, 0]))
            rot_deg = np.array(c['parameters'].get('rotation', [0, 0, 0]))
            ref_pose = transformation_matrix(pos, rot_deg * np.pi / 180.0)
            break
            
    if ref_pose is None:
        return data
 
    ref_inv = np.linalg.inv(ref_pose)
    pos_trans, rot_trans = pos.copy(), rot_deg.copy()
    pos_trans = [-x for x in pos]
    rot_trans = [-x * np.pi / 180 for x in rot_deg]
    data["fixed_body_pose"] = {"pose": ref_inv, "pos": pos_trans, "rot": rot_trans}
    
    return data

if __name__ == "__main__":
    
    with open("conceptualization.pkl", "rb") as f:
        data_list = pickle.load(f).copy()
        
    for i in range(len(data_list)):
        print(f"Exporting {i}-th {template_name} urdf, id: {data_list[i]['id']}")
        data_id = data_list[i]['id']
        output_path = f"{template_name}_urdf/" + data_id
        if use_raw_obj:
            output_path = f"{template_name}_urdf_raw_obj/" + data_id
        data = data_list[i].copy()
        data = adjust_pkl(data)
        if use_raw_obj:
            raw_obj_dir = f"{template_name}_raw_obj/" + data_id + "/"
            if not os.path.exists(raw_obj_dir):
                print(f"Raw obj directory {raw_obj_dir} does not exist. Please check.")
                continue
            bottle = Bottle(f"{template_name}.urdf", data, output_path, raw_obj_dir=raw_obj_dir)
        else:
            bottle = Bottle(f"{template_name}.urdf", data, output_path)
        bottle.export_urdf()
