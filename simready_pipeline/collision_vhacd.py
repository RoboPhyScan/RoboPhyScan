"""Prepare visual-reused collision meshes and bake Isaac Sim VHACD hulls."""

import json
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from omni.convexdecomposition.bindings import _convexdecomposition as vhacd


ROOT = Path(__file__).resolve().parent
# Set MESH_CLEAN_PYTHON to the Python interpreter that has pymeshlab or open3d.
# Without it, the current interpreter is used.
DEFAULT_CLEAN_PYTHON = Path(sys.executable)
ERROR_PERCENTAGES = (10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5)
VHACD_TIMEOUT_SECONDS = float(os.environ.get("VHACD_TIMEOUT_SECONDS", "900"))


def _parse_obj(path):
    vertices = []
    faces = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append(tuple(float(x) for x in parts[1:4]))
            elif parts[0] == "f" and len(parts) >= 4:
                face = []
                for token in parts[1:]:
                    index = int(token.split("/", 1)[0])
                    face.append(index - 1 if index > 0 else len(vertices) + index)
                for i in range(1, len(face) - 1):
                    faces.append((face[0], face[i], face[i + 1]))
    return vertices, faces


def _write_obj(path, vertices, faces):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for vertex in vertices:
            handle.write("v %.9g %.9g %.9g\n" % vertex)
        for face in faces:
            handle.write("f %d %d %d\n" % tuple(index + 1 for index in face))


def _clean_collision_mesh(input_path, output_path, target_face_num):
    input_path = Path(input_path)
    output_path = Path(output_path)
    cleaner = Path(os.environ.get("MESH_CLEAN_PYTHON", str(DEFAULT_CLEAN_PYTHON)))
    if not cleaner.exists():
        cleaner = Path(sys.executable)
    def run_pymeshlab():
        code = (
            "import pymeshlab; "
            "ms=pymeshlab.MeshSet(); "
            f"ms.load_new_mesh(r'{input_path}'); "
            "ms.meshing_remove_duplicate_vertices(); "
            "ms.meshing_remove_duplicate_faces(); "
            "ms.meshing_remove_null_faces(); "
            "ms.meshing_remove_unreferenced_vertices(); "
            "ms.meshing_decimation_quadric_edge_collapse("
            f"targetfacenum={target_face_num}, preserveboundary=True, preservetopology=True, "
            "preservenormal=True, optimalplacement=True); "
            "ms.meshing_remove_unreferenced_vertices(); "
            "ms.compute_normal_per_face(); "
            "ms.compute_normal_per_vertex(); "
            f"ms.save_current_mesh(r'{output_path}')"
        )
        return subprocess.run([str(cleaner), "-c", code], check=False)

    result = run_pymeshlab()
    if result.returncode != 0 or not output_path.exists():
        output_path.unlink(missing_ok=True)
        code = (
            "import open3d as o3d,sys; "
            "mesh=o3d.io.read_triangle_mesh(sys.argv[1]); "
            "mesh.remove_duplicated_vertices(); mesh.remove_duplicated_triangles(); "
            "mesh.remove_degenerate_triangles(); mesh.remove_unreferenced_vertices(); "
            "mesh=mesh.simplify_quadric_decimation(int(sys.argv[3])); "
            "mesh.remove_unreferenced_vertices(); "
            "assert o3d.io.write_triangle_mesh(sys.argv[2], mesh)"
        )
        result = subprocess.run(
            [str(cleaner), "-c", code, str(input_path), str(output_path), str(target_face_num)], check=False
        )
    if result.returncode != 0 or not output_path.exists():
        raise RuntimeError(f"collision mesh clean failed for {input_path} (exit={result.returncode})")


def _vhacd_hulls(obj_path, minimum_count):
    vertices, faces = _parse_obj(obj_path)
    if not vertices or not faces:
        raise RuntimeError(f"empty OBJ: {obj_path}")
    simple_mesh = vhacd.SimpleMesh()
    simple_mesh.vertices = [value for vertex in vertices for value in vertex]
    simple_mesh.indices = [value for face in faces for value in face]
    interface = vhacd.acquire_convexdecomposition_interface()
    try:
        selected = None
        deadline = time.monotonic() + VHACD_TIMEOUT_SECONDS if VHACD_TIMEOUT_SECONDS > 0 else None
        for error_percentage in ERROR_PERCENTAGES:
            handle = interface.create_vhacd()
            params = vhacd.Parameters()
            params.error_percentage = error_percentage
            params.max_convex_hull_count = int(os.environ.get("VHACD_MAX_HULLS", "256"))
            params.max_hull_vertices = 64
            interface.begin_vhacd(handle, simple_mesh, params)
            while not interface.is_complete(handle):
                if deadline is not None and time.monotonic() > deadline:
                    interface.cancel_vhacd(handle)
                    interface.release_vhacd(handle)
                    raise TimeoutError(
                        f"VHACD timeout after {VHACD_TIMEOUT_SECONDS:.0f}s for {obj_path} "
                        f"(error={error_percentage})"
                    )
                time.sleep(0.01)
            count = interface.get_convex_hull_count(handle)
            if count >= minimum_count:
                selected = (handle, error_percentage, count)
                break
            interface.release_vhacd(handle)
        if selected is None:
            raise RuntimeError(f"VHACD produced fewer than {minimum_count} hulls for {obj_path}")
        handle, error_percentage, count = selected
        hulls = []
        for index in range(count):
            hull = vhacd.SimpleMesh()
            if not interface.get_convex_hull(handle, index, hull):
                raise RuntimeError(f"cannot read VHACD hull {index} for {obj_path}")
            hulls.append((list(hull.vertices), list(hull.indices)))
        interface.release_vhacd(handle)
        return hulls, error_percentage
    finally:
        vhacd.release_convexdecomposition_interface(interface)


def _template_hull_minimum(conceptualization_path):
    if not conceptualization_path.exists():
        return {}
    with conceptualization_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        data = data[0] if data else {}
    result = {}
    for item in data.get("conceptualization", []):
        template = item.get("template", "")
        params = item.get("parameters", {})
        component = item.get("component")
        if template == "Multilevel_Body":
            count = sum(int(x) for x in params.get("num_levels", [1]))
        elif "Half_Ring" in template or "Half_Torus" in template:
            count = 2
        elif "Ring" in template or "Torus" in template or template == "Cylindrical_Lid":
            count = 4 if template != "Cylindrical_Lid" else 5
        else:
            count = 1
        if component:
            result[component] = result.get(component, 0) + count
        result[template] = result.get(template, 0) + count
    return result


def _resolve_mesh(urdf_path, filename):
    return (urdf_path.parent / filename).resolve()


def prepare_vhacd_urdf(urdf_path, output_dir, conceptualization_path):
    """Create output URDF whose collision entries reference baked convex hull OBJs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    collision_dir = output_dir / "meshes" / "collision_vhacd"
    collision_dir.mkdir(parents=True, exist_ok=True)
    # 清除本流程上一次生成的文件，避免较大 hull 编号残留在输出目录。
    for pattern in ("*_clean.obj", "*_clean.obj.mtl", "*_hull_*.obj"):
        for stale_path in collision_dir.glob(pattern):
            stale_path.unlink()
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    minimums = _template_hull_minimum(conceptualization_path)
    manifest = {"source_urdf": urdf_path.name, "meshes": []}

    for link in root.findall("link"):
        visual_mesh = None
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is not None and mesh.get("filename"):
                visual_mesh = mesh.get("filename")
                mesh.set("filename", os.path.join("..", "sim_ready", visual_mesh).replace("\\", "/"))
        if visual_mesh is None:
            continue
        collisions = list(link.findall("collision"))
        if not collisions:
            collision = ET.Element("collision", {"name": f"{link.get('name')}_visual_vhacd"})
            geometry = ET.SubElement(collision, "geometry")
            ET.SubElement(geometry, "mesh", {"filename": visual_mesh})
            collisions = [collision]
        for collision in collisions:
            mesh = collision.find("geometry/mesh")
            if mesh is None or not mesh.get("filename"):
                continue
            collision_mesh = mesh.get("filename")
            visual_path = _resolve_mesh(urdf_path, visual_mesh)
            collision_path = _resolve_mesh(urdf_path, collision_mesh)
            if visual_path != collision_path:
                mesh.set("filename", os.path.join("..", "sim_ready", collision_mesh).replace("\\", "/"))
                continue
            cleaned_path = collision_dir / f"{Path(collision_mesh).stem}_clean.obj"
            source_for_vhacd = collision_path
            source_faces = len(_parse_obj(collision_path)[1])
            target_face_num = max(1, source_faces // 10)
            if source_faces > target_face_num:
                _clean_collision_mesh(collision_path, cleaned_path, target_face_num)
                source_for_vhacd = cleaned_path
            minimum = max(1, minimums.get(link.get("name"), minimums.get(link.get("name", ""), 1)))
            hulls, error_percentage = _vhacd_hulls(source_for_vhacd, minimum)
            for old in list(link.findall("collision")):
                if old is collision:
                    link.remove(old)
            for index, (vertices, indices) in enumerate(hulls):
                hull_path = collision_dir / f"{Path(collision_mesh).stem}_hull_{index:03d}.obj"
                faces = [tuple(indices[i:i + 3]) for i in range(0, len(indices), 3)]
                _write_obj(hull_path, list(zip(*[iter(vertices)] * 3)), faces)
                new_collision = ET.Element("collision", {"name": f"{link.get('name')}_vhacd_{index:03d}"})
                origin = collision.find("origin")
                if origin is not None:
                    new_collision.append(ET.fromstring(ET.tostring(origin, encoding="unicode")))
                geometry = ET.SubElement(new_collision, "geometry")
                ET.SubElement(geometry, "mesh", {"filename": os.path.relpath(hull_path, output_dir).replace("\\", "/")})
                link.append(new_collision)
            manifest["meshes"].append({"link": link.get("name"), "source_faces": source_faces, "hulls": len(hulls), "error_percentage": error_percentage, "source": os.path.relpath(source_for_vhacd, output_dir).replace("\\", "/"), "minimum_hulls": minimum})

    output_urdf = output_dir / urdf_path.name
    tree.write(output_urdf, encoding="unicode")
    (output_dir / "vhacd_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return str(output_urdf), manifest
