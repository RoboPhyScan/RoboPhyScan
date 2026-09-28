import os
import xml.etree.ElementTree as ET
import trimesh
import numpy as np

class Export:
    def __init__(self, links, joints, template_name, output_dir):
        self.template_name = template_name
        self.output_dir = output_dir
        self.links = links
        self.joints = joints

    def write_urdf(self, filename="robot.urdf"):
        robot = ET.Element('robot', attrib={'name': self.template_name})
        world_el = ET.SubElement(robot, 'link', attrib={'name': 'world'})
        
        # links
        for l in self.links:
            link_el = ET.SubElement(robot, 'link', attrib={'name': l['name']})

            inertial = ET.SubElement(link_el, 'inertial')
            mass_el = ET.SubElement(inertial, 'mass', attrib={'value': '{:g}'.format(l['mass'])})
            inertia_el = ET.SubElement(inertial, 'inertia', attrib={
                'ixx': '{:g}'.format(max(l['inertia'][0][0], 1e-3)),
                'iyy': '{:g}'.format(max(l['inertia'][1][1], 1e-3)),
                'izz': '{:g}'.format(max(l['inertia'][2][2], 1e-3)),
                'ixy': '{:g}'.format(l['inertia'][0][1]),
                'ixz': '{:g}'.format(l['inertia'][0][2]),
                'iyz': '{:g}'.format(l['inertia'][1][2])
            })

            if l.get('mesh') is not None:
                visual = ET.SubElement(link_el, 'visual')
                origin_v = ET.SubElement(visual, 'origin', attrib={'xyz': self._vec_to_str(l['origin_xyz']), 'rpy': self._vec_to_str(l['origin_rpy'])})
                geometry_v = ET.SubElement(visual, 'geometry')
                mesh_v = ET.SubElement(geometry_v, 'mesh', attrib={'filename': l['mesh']})

                collision = ET.SubElement(link_el, 'collision')
                origin_c = ET.SubElement(collision, 'origin', attrib={'xyz': self._vec_to_str(l['origin_xyz']), 'rpy': self._vec_to_str(l['origin_rpy'])})
                geometry_c = ET.SubElement(collision, 'geometry')
                mesh_c = ET.SubElement(geometry_c, 'mesh', attrib={'filename': l['mesh']})

        # joints 
        for j in self.joints:
            joint_el = ET.SubElement(robot, 'joint', attrib={'name': f"{j['parent']}_to_{j['child']}", 'type': j['type']})
            parent_el = ET.SubElement(joint_el, 'parent', attrib={'link': j['parent']})
            child_el = ET.SubElement(joint_el, 'child', attrib={'link': j['child']})
            origin_el = ET.SubElement(joint_el, 'origin', attrib={'xyz': self._vec_to_str(j['origin_xyz']), 'rpy': self._vec_to_str(j['origin_rpy'])})
            axis_el = ET.SubElement(joint_el, 'axis', attrib={'xyz': '{:.6f} {:.6f} {:.6f}'.format(float(j['axis'][0]), float(j['axis'][1]), float(j['axis'][2]))})
            if j.get('limit') is not None:
                limit = j['limit']
                ET.SubElement(joint_el, 'limit', attrib={
                    'lower': str(limit.get('lower', 0.0)),
                    'upper': str(limit.get('upper', 0.0)),
                    'effort': str(limit.get('effort', 0.0)),
                    'velocity': str(limit.get('velocity', 0.0))
                })

        # adjust indentation for pretty printing
        self._indent(robot)
        tree = ET.ElementTree(robot)
        out_path = os.path.join(self.output_dir, filename)
        tree.write(out_path, encoding='utf-8', xml_declaration=True)
    
    def _vec_to_str(self, v):
        return "{:.6f} {:.6f} {:.6f}".format(float(v[0]), float(v[1]), float(v[2]))

    def _indent(self, elem, level=0):
        i = "\n" + level * "    "
        if len(elem):
            if level != 0 and (not elem.tail or not elem.tail.strip()):
                elem.tail = i
            if not elem.text or not elem.text.strip():
                elem.text = i + "    "
            for e in elem:
                self._indent(e, level + 1)
            if not e.tail or not e.tail.strip():
                e.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

def get_transformation_matrix(position, rotation, rotation_order="XYZ", offset_first=False):
    from utils import get_rodrigues_matrix
    rot_mat = {"X": get_rodrigues_matrix([1, 0, 0], rotation[0]),
               "Y": get_rodrigues_matrix([0, 1, 0], rotation[1]),
               "Z": get_rodrigues_matrix([0, 0, 1], rotation[2])}
    R = np.eye(3)
    for s in rotation_order:
        R = np.matmul(R, rot_mat[s].T)
        
    T = np.eye(4)
    T[:3, :3] = R
    if offset_first:
        T[:3, 3] = R.dot(position)
    else:
        T[:3, 3] = position
    return T

def find_obj_in_folder(folder_path, target_name):
    meshes = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.obj') and filename.startswith(target_name):
            file_path = os.path.join(folder_path, filename)
            mesh = trimesh.load(file_path, force='mesh')
            mesh.metadata['file_path'] = file_path
            meshes.append(mesh)
    return meshes

def get_rotation_matrix(order='xyz', rotation=[0, 0, 0]):
    matrix = {}
    matrix['x'] = np.array([
        [1, 0, 0],
        [0, np.cos(rotation[0]), -np.sin(rotation[0])],
        [0, np.sin(rotation[0]), np.cos(rotation[0])]
    ])
    matrix['y'] = np.array([
        [np.cos(rotation[1]), 0, np.sin(rotation[1])],
        [0, 1, 0],
        [-np.sin(rotation[1]), 0, np.cos(rotation[1])]
    ])
    matrix['z'] = np.array([
        [np.cos(rotation[2]), -np.sin(rotation[2]), 0],
        [np.sin(rotation[2]), np.cos(rotation[2]), 0],
        [0, 0, 1]
    ])
    rot_mat = np.eye(3)
    for s in order:
        rot_mat = rot_mat @ matrix[s]
    return rot_mat

def get_part_pose(position, rotation, order='yxz', offset_first=False):
    if offset_first:
        RT = np.eye(4)
        RT[:3, :3] = get_rotation_matrix(order, rotation)
        RT[:3, -1] = position
    else:
        RT = np.eye(4)
        rot = get_rotation_matrix(order, rotation)
        pos = rot @ np.array(position)
        RT[:3, :3] = rot
        RT[:3, -1] = pos
    return RT

def matrix_to_pose(T):
    xyz = T[:3, 3].copy()
    rpy = list(trimesh.transformations.euler_from_matrix(T, axes='sxyz'))
    return xyz, rpy

def pose_from_apply(position, rotation, rotation_order="YXZ", offset_first=False):
    """Build 4x4 pose with the exact same convention as apply_transformation."""
    from utils import apply_transformation
    basis = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    transformed = apply_transformation(
        basis.copy(),
        position,
        rotation,
        rotation_order=rotation_order,
        offset_first=offset_first,
    )

    origin = transformed[0]
    x_axis = transformed[1] - origin
    y_axis = transformed[2] - origin
    z_axis = transformed[3] - origin

    RT = np.eye(4)
    RT[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    RT[:3, -1] = origin
    return RT
