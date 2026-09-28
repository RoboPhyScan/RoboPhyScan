import os
import numpy as np
import trimesh
from knowledge_definitions import *
import trimesh.transformations as tra
from exporter_utils import matrix_to_pose
from utils import apply_transformation

class URDF_Knowledge_Definitions:
    def __init__(self, template_name, output_dir):
        self.template_name = template_name
        self.output_dir = output_dir
        self.mesh_dir = os.path.join(self.output_dir, "meshes")
        os.makedirs(self.mesh_dir, exist_ok=True)
        self.default_density = {
            'aluminum': 2700.0,      # Aluminium
            'steel': 7850.0,         # Steel
            'copper': 8960.0,        # Copper
            'wood': 500.0,           # Wood
            'plastic': 950.0,        # Plastic
            'water': 1000.0,         # Water
            'concrete': 2400.0,      # Concrete
            'rubber': 1100.0,        # Rubber
            'glass': 2500.0,         # Glass
            'air': 1.2,              # Air
        }

    def _write_export_asset(self, out_path, content):
        if isinstance(content, bytes):
            with open(out_path, 'wb') as f:
                f.write(content)
            return
        if isinstance(content, str):
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return
        if hasattr(content, 'save'):
            content.save(out_path)
            return
        raise TypeError(f"Unsupported exported asset type for {out_path}: {type(content)}")

    def _shift_obj_index(self, token, offsets):
        parts = token.split('/')
        out = []
        for i, value in enumerate(parts):
            if value == '':
                out.append(value)
                continue
            idx = int(value)
            if idx > 0:
                idx += offsets[i]
            out.append(str(idx))
        return '/'.join(out)

    def _copy_texture_assets(self, src_dir, mesh_dir):
        import os
        import shutil

        for file in os.listdir(src_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                shutil.copy2(os.path.join(src_dir, file), os.path.join(mesh_dir, file))

    def _copy_first_mtl(self, src_dir, mesh_dir, mtl_files, new_mtl_name):
        import os
        import shutil

        for mtl_file in mtl_files:
            mtl_path = os.path.join(src_dir, mtl_file)
            if os.path.exists(mtl_path):
                shutil.copy2(mtl_path, os.path.join(mesh_dir, new_mtl_name))
                return True

        fallback_mtl = next((f for f in os.listdir(src_dir) if f.endswith('.mtl')), None)
        if fallback_mtl:
            shutil.copy2(os.path.join(src_dir, fallback_mtl), os.path.join(mesh_dir, new_mtl_name))
            return True
        return False

    def _transform_raw_obj_vertex(self, values, transform):
        vertex = np.array([[float(values[0]), float(values[1]), float(values[2])]], dtype=float)
        if transform is None:
            return vertex[0]
        return apply_transformation(
            vertex,
            transform["position"],
            transform["rotation"],
            rotation_order=transform["rotation_order"],
            offset_first=transform["offset_first"],
        )[0]

    def _transform_export_vertex(self, vertex, transform):
        if transform is None:
            return vertex
        vertex_h = np.append(np.asarray(vertex, dtype=float), 1.0)
        return (np.asarray(transform, dtype=float) @ vertex_h)[:3]

    def _transform_raw_obj_normal(self, values, transform):
        normal = np.array([[float(values[0]), float(values[1]), float(values[2])]], dtype=float)
        if transform is not None:
            normal = apply_transformation(
                normal,
                [0, 0, 0],
                transform["rotation"],
                rotation_order=transform["rotation_order"],
                offset_first=False,
            )
        norm = np.linalg.norm(normal[0])
        if norm > 1e-12:
            normal[0] = normal[0] / norm
        return normal[0]

    def _transform_export_normal(self, normal, transform):
        if transform is None:
            return normal
        normal = np.asarray(transform[:3, :3], dtype=float) @ np.asarray(normal, dtype=float)
        norm = np.linalg.norm(normal)
        if norm > 1e-12:
            normal = normal / norm
        return normal

    def _export_obj_from_sources(self, source_meshes, mesh_fname, link_name):
        import os

        mesh_dir = os.path.dirname(mesh_fname)
        new_mtl_name = f"{link_name}.mtl"
        out_lines = ["# OBJ exported from source mesh with original UVs", f"mtllib {new_mtl_name}"]
        offsets = [0, 0, 0]
        copied_mtl = False

        for source_mesh in source_meshes:
            if not hasattr(source_mesh, 'metadata') or 'file_path' not in source_mesh.metadata:
                return False

            src_path = source_mesh.metadata['file_path']
            src_dir = os.path.dirname(src_path)
            raw_transform = source_mesh.metadata.get("raw_obj_vertex_transform")
            use_transformed_vertices = source_mesh.metadata.get("use_transformed_vertices_for_export", False)
            transformed_vertices = iter(source_mesh.vertices) if raw_transform is None and not use_transformed_vertices else None
            mtl_files = []

            with open(src_path, 'r', encoding='utf-8') as f:
                source_lines = f.readlines()

            for line in source_lines:
                if line.startswith('mtllib '):
                    mtl_files.extend(line.strip().split()[1:])

            if not copied_mtl:
                copied_mtl = self._copy_first_mtl(src_dir, mesh_dir, mtl_files, new_mtl_name)
            self._copy_texture_assets(src_dir, mesh_dir)

            local_counts = [0, 0, 0]
            for line in source_lines:
                stripped = line.strip()
                if line.startswith('mtllib ') or stripped == '':
                    continue
                if line.startswith('v '):
                    try:
                        if raw_transform is None and not use_transformed_vertices:
                            v = next(transformed_vertices)
                        else:
                            v = self._transform_raw_obj_vertex(stripped.split()[1:4], raw_transform)
                            if use_transformed_vertices:
                                v = self._transform_export_vertex(v, source_mesh.metadata.get("export_vertex_transform"))
                    except (IndexError, ValueError, KeyError, StopIteration):
                        return False
                    out_lines.append(f"v {float(v[0]):.8f} {float(v[1]):.8f} {float(v[2]):.8f}")
                    local_counts[0] += 1
                elif line.startswith('vt '):
                    out_lines.append(stripped)
                    local_counts[1] += 1
                elif line.startswith('vn '):
                    try:
                        vn = self._transform_raw_obj_normal(stripped.split()[1:4], raw_transform)
                        if use_transformed_vertices:
                            vn = self._transform_export_normal(vn, source_mesh.metadata.get("export_normal_transform"))
                    except (IndexError, ValueError, KeyError):
                        return False
                    out_lines.append(f"vn {float(vn[0]):.8f} {float(vn[1]):.8f} {float(vn[2]):.8f}")
                    local_counts[2] += 1
                elif line.startswith('f '):
                    tokens = stripped.split()
                    shifted = [self._shift_obj_index(t, offsets) for t in tokens[1:]]
                    out_lines.append('f ' + ' '.join(shifted))
                elif line.startswith('#'):
                    continue
                else:
                    out_lines.append(stripped)

            offsets = [offsets[i] + local_counts[i] for i in range(3)]

        with open(mesh_fname, 'w', encoding='utf-8') as f:
            f.write('\n'.join(out_lines) + '\n')
        return True

    def _export_obj_with_textures(self, mesh_transformed, mesh_fname, link_name):
        """Export OBJ and sidecar MTL/texture files when textured visual data exists."""
        exported = False
        try:
            source_meshes = []
            if hasattr(mesh_transformed, 'metadata') and 'source_meshes' in mesh_transformed.metadata:
                source_meshes = mesh_transformed.metadata['source_meshes']
            elif hasattr(mesh_transformed, 'metadata') and 'file_path' in mesh_transformed.metadata:
                source_meshes = [mesh_transformed]

            if source_meshes and self._export_obj_from_sources(source_meshes, mesh_fname, link_name):
                exported = True
                print(f"Exported source-UV-preserved mesh for link '{link_name}' to {mesh_fname}")
        except Exception as e:
            print(f"Source OBJ UV preservation failed: {e}")

        try:
            if not exported and hasattr(mesh_transformed, 'metadata') and 'file_path' in mesh_transformed.metadata:
                src_path = mesh_transformed.metadata['file_path']
                import open3d as o3d
                import trimesh
                import numpy as np
                import shutil
                import os
                import re

                def replace_or_insert_obj_line(obj_text, pattern, replacement):
                    if re.search(pattern, obj_text, flags=re.MULTILINE):
                        return re.sub(pattern, replacement, obj_text, count=1, flags=re.MULTILINE)

                    lines = obj_text.splitlines()
                    insert_at = 0
                    while insert_at < len(lines) and lines[insert_at].startswith('#'):
                        insert_at += 1
                    lines.insert(insert_at, replacement)
                    return '\n'.join(lines) + ('\n' if obj_text.endswith('\n') else '')
                
                try:
                    orig_mesh = trimesh.load(src_path, process=True)
                    A = orig_mesh.vertices
                    B = mesh_transformed.vertices
                    
                    if A.shape == B.shape:
                        A_homog = np.hstack((A, np.ones((A.shape[0], 1))))
                        T_T, _, _, _ = np.linalg.lstsq(A_homog, B, rcond=None)
                        T = np.eye(4)
                        T[:3, :] = T_T.T
                        
                        mesh_o3d = o3d.io.read_triangle_mesh(src_path, enable_post_processing=True)
                        mesh_o3d.transform(T)
                        
                        mesh_o3d.textures = []
                        o3d.io.write_triangle_mesh(mesh_fname, mesh_o3d, write_triangle_uvs=True)
                        
                        src_dir = os.path.dirname(src_path)
                        mesh_dir = os.path.dirname(mesh_fname)
                        new_mtl_name = f"{link_name}.mtl"
                        
                        mtl_files = []
                        orig_usemtl = 'usemtl material_0\n'
                        with open(src_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                if line.startswith('mtllib '):
                                    mtl_files.extend(line.strip().split()[1:])
                                elif line.startswith('usemtl '):
                                    orig_usemtl = line
                        if len(mtl_files) == 0:
                            mtl_files = [f for f in os.listdir(src_dir) if f.endswith('.mtl')]
                            
                        with open(mesh_fname, 'r', encoding='utf-8') as f:
                            obj_data = f.read()
                        
                        obj_data = replace_or_insert_obj_line(obj_data, r'^mtllib .*$', f'mtllib {new_mtl_name}')
                        obj_data = replace_or_insert_obj_line(obj_data, r'^usemtl .*$', orig_usemtl.strip())
                        
                        with open(mesh_fname, 'w', encoding='utf-8') as f:
                            f.write(obj_data)
                            
                        copied_mtl = False
                        for mtl_file in mtl_files:
                            mtl_path = os.path.join(src_dir, mtl_file)
                            if os.path.exists(mtl_path):
                                shutil.copy2(mtl_path, os.path.join(mesh_dir, new_mtl_name))
                                copied_mtl = True
                                break
                        if not copied_mtl:
                            fallback_mtl = next((f for f in os.listdir(src_dir) if f.endswith('.mtl')), None)
                            if fallback_mtl:
                                shutil.copy2(os.path.join(src_dir, fallback_mtl), os.path.join(mesh_dir, new_mtl_name))
                        for file in os.listdir(src_dir):
                            if file.endswith(('.jpg', '.jpeg', '.png')):
                                shutil.copy2(os.path.join(src_dir, file), os.path.join(mesh_dir, file))
                                
                        exported = True
                        print(f"Exported Open3D UV-preserved mesh for link '{link_name}' to {mesh_fname}")
                except Exception as e:
                    print(f"Open3D UV preservation failed: {e}")
        except Exception as e:
            print(f"Failed to check metadata: {e}")
            
        if not exported:
            try:
                obj_text, asset_map = trimesh.exchange.obj.export_obj(
                    mesh_transformed,
                    include_normals=True,
                    include_texture=True,
                    return_texture=True,
                    mtl_name=f"{link_name}.mtl"
                )
                with open(mesh_fname, 'w', encoding='utf-8') as f:
                    f.write(obj_text)

                if isinstance(asset_map, dict):
                    mesh_dir = os.path.dirname(mesh_fname)
                    for asset_name, asset_content in asset_map.items():
                        out_name = os.path.basename(str(asset_name))
                        out_path = os.path.join(mesh_dir, out_name)
                        self._write_export_asset(out_path, asset_content)
                return
            except Exception:
                # Fallback for meshes without texture visual data or unsupported exporters.
                mesh_transformed.export(mesh_fname, include_normals=True)

    def _is_valid_inertia(self, inertia_mat):
        if inertia_mat is None:
            return False
        I = np.asarray(inertia_mat, dtype=float)
        if I.shape != (3, 3):
            return False
        if not np.all(np.isfinite(I)):
            return False
        # symmetry
        I = 0.5 * (I + I.T)
        # principal moments must be positive
        evals = np.linalg.eigvalsh(I)
        if np.any(evals <= 1e-12):
            return False
        # URDF common physical constraints for principal moments
        d = np.diag(I)
        if np.any(d <= 1e-12):
            return False
        if d[0] > d[1] + d[2] or d[1] > d[0] + d[2] or d[2] > d[0] + d[1]:
            return False
        return True

    def _fallback_box_inertial(self, mesh_transformed, density):
        # Conservative, always valid inertial parameters from axis-aligned bounding box
        extents = np.asarray(mesh_transformed.bounding_box.extents, dtype=float)
        if extents.shape != (3,) or not np.all(np.isfinite(extents)):
            extents = np.array([1e-3, 1e-3, 1e-3], dtype=float)
        extents = np.maximum(extents, 1e-6)

        volume = float(np.prod(extents))
        mass = max(float(density) * volume, 1e-9)

        x, y, z = extents
        ixx = mass * (y * y + z * z) / 12.0
        iyy = mass * (x * x + z * z) / 12.0
        izz = mass * (x * x + y * y) / 12.0
        inertia = np.array([
            [ixx, 0.0, 0.0],
            [0.0, iyy, 0.0],
            [0.0, 0.0, izz]
        ], dtype=float)
        return mass, inertia

    def add_part(self, link_name, mesh, pose, parent_name, parent_pose, pose_adjust=np.eye(4), fix_body_pose=np.eye(4), joint_type="fixed", joint_axis=[0, 0, 1], joint_limit=None, material='steel', joint_origin_xyz=None, joint_origin_rpy=None):
        mesh_transformed = None
        mesh_relpath = None
        if mesh is not None:
            mesh_transformed = mesh.copy()
            mesh_transformed.apply_transform(np.linalg.inv(pose_adjust))
            if not np.allclose(pose_adjust, np.eye(4)):
                mesh_transformed.metadata["use_transformed_vertices_for_export"] = True
                mesh_transformed.metadata["export_vertex_transform"] = np.linalg.inv(pose_adjust)
                mesh_transformed.metadata["export_normal_transform"] = np.linalg.inv(pose_adjust)

            # export mesh as OBJ
            mesh_fname = os.path.join(self.mesh_dir, f"{link_name}.obj")
            self._export_obj_with_textures(mesh_transformed, mesh_fname, link_name)
            print(f"Exported mesh for link '{link_name}' to {mesh_fname}")

        # compute mass and inertia from local mesh
        density = self.default_density.get(material, 1000.0)  # default to 1000 kg/m^3 if material not found
        mass = None
        inertia_mat = None

        if mesh_transformed is not None:
            # Try trimesh inertial first
            try:
                mesh_transformed.density = density
                mass_try = float(mesh_transformed.mass)
                inertia_try = np.asarray(mesh_transformed.moment_inertia, dtype=float)
                if np.isfinite(mass_try) and mass_try > 1e-12 and self._is_valid_inertia(inertia_try):
                    mass = mass_try
                    inertia_mat = 0.5 * (inertia_try + inertia_try.T)
            except Exception:
                pass

            # Fallback for non-watertight / inconsistent meshes
            if mass is None or inertia_mat is None:
                mass, inertia_mat = self._fallback_box_inertial(mesh_transformed, density)
            mesh_relpath = os.path.relpath(mesh_fname, self.output_dir).replace('\\', '/')
        else:
            # virtual link: no visual/collision mesh
            mass = 1e-6
            inertia_mat = np.eye(3) * 1e-6

        link_origin_xyz, link_origin_rpy = [0, 0, 0], [0, 0, 0]

        # construct link dictionary
        link = {
            'name': link_name,
            'mesh': mesh_relpath,
            'origin_xyz': link_origin_xyz,
            'origin_rpy': link_origin_rpy,
            'mass': mass,
            'inertia': inertia_mat
        }

        # compute joint origin in parent link frame 
        if joint_origin_xyz is not None and joint_origin_rpy is not None:
            joint_origin_xyz_used = joint_origin_xyz
            joint_origin_rpy_used = joint_origin_rpy
        else:
            pose_trans = fix_body_pose @ pose
            # fix rotation for URDF joint axis convention
            if parent_name == "world":
                fix_rotation = tra.euler_matrix(np.pi/2, 0, 0, axes='sxyz')  
                parent_pose_trans = parent_pose
            else:
                fix_rotation = tra.euler_matrix(0, 0, 0, axes='sxyz')
                parent_pose_trans = fix_body_pose @ parent_pose
            T_joint = np.dot(np.linalg.inv(parent_pose_trans), pose_trans)
            T_joint = np.dot(T_joint, fix_rotation)
            joint_origin_xyz_used, joint_origin_rpy_used = matrix_to_pose(T_joint)
        
        # construct joint dictionary
        joint = {
            'type': joint_type,
            'axis': joint_axis,
            'origin_xyz': joint_origin_xyz_used,
            'origin_rpy': joint_origin_rpy_used,
            'parent': parent_name,
            'child': link_name,
            'limit': joint_limit
        }

        return link, joint
