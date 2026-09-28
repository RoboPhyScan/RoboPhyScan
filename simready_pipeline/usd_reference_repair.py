"""Remove importer-authored references to geometry-less virtual links."""

from pathlib import Path
import xml.etree.ElementTree as ET

from pxr import Sdf, Usd


def _reference_target(reference, source_layer):
    asset = reference.assetPath
    target_layer = source_layer
    if asset:
        target_layer = Sdf.Layer.FindOrOpen(str((Path(source_layer.realPath).parent / asset).resolve()))
    return target_layer, reference.primPath


def repair_usd_references(base_path, physics_path=None, prepared_urdf=None, remove=True):
    """Repair only missing references on /visuals prims with no mesh geometry.

    Returns a structured audit record. References on geometry-bearing prims are
    reported as errors and are never redirected or removed.
    """
    base_layer = Sdf.Layer.FindOrOpen(str(Path(base_path).resolve()))
    if base_layer is None:
        return {"status": "error", "error": f"cannot open {base_path}", "removed": [], "errors": []}
    physics_stage = Usd.Stage.Open(str(physics_path)) if physics_path and Path(physics_path).exists() else None
    geometry_links = _links_with_mesh_geometry(prepared_urdf)
    removed, unresolved, errors = [], [], []
    for prim_spec in _walk_specs(base_layer.rootPrims):
        refs = prim_spec.referenceList.GetAddedOrExplicitItems()
        if not refs:
            continue
        source_path = str(prim_spec.path)
        for ref in refs:
            target_layer, target_path = _reference_target(ref, base_layer)
            target_exists = bool(target_layer and target_layer.GetPrimAtPath(target_path))
            if target_exists:
                continue
            entry = {"source_layer": base_layer.identifier, "source_prim": source_path,
                     "target_layer": target_layer.identifier if target_layer else str(ref.assetPath),
                     "target_prim": str(target_path), "list_op": "added"}
            unresolved.append(entry)
            if not source_path.endswith("/visuals"):
                errors.append({**entry, "error": "missing reference outside visual scope"})
                continue
            # The importer creates these references only for virtual links. A
            # real geometry reference must be fixed at the source, not guessed.
            source_name = source_path.rsplit("/", 2)[-2]
            if source_name in geometry_links:
                errors.append({**entry, "error": "missing reference on geometry-bearing link"})
                continue
            if "virtual" not in source_name.lower() and source_name not in {
                "Regular_lever", "Double_Cuboidal_Shaft"
            }:
                errors.append({**entry, "error": "missing reference on non-virtual visual prim"})
                continue
            if physics_stage and physics_stage.GetPrimAtPath(target_path):
                errors.append({**entry, "error": "target missing from composed layer but source target exists"})
                continue
            if remove:
                prim_spec.referenceList.Remove(ref)
                removed.append({**entry, "status": "removed_virtual_geometry_reference"})
            break
    if remove and removed:
        base_layer.Save()
    return {"status": "error" if errors else "ok", "removed": removed,
            "unresolved": unresolved, "errors": errors}


def _walk_specs(specs):
    for spec in specs:
        yield spec
        yield from _walk_specs(spec.nameChildren.values())


def _links_with_mesh_geometry(urdf_path):
    if not urdf_path or not Path(urdf_path).exists():
        return set()
    try:
        root = ET.parse(urdf_path).getroot()
    except ET.ParseError:
        return set()
    return {
        link.get("name") for link in root.findall("link")
        if link.find("visual/geometry/mesh") is not None or link.find("collision/geometry/mesh") is not None
    }
