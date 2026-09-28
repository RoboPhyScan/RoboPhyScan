import argparse
import json
import pymeshlab
import os
import threading
import time
from pathlib import Path


def _safe_close_holes(ms: pymeshlab.MeshSet, maxholesize: int = 100) -> None:
    """
    meshing_close_holes 需要边流形。先尝试修复再补洞，仍失败则跳过以避免整流程中断。
    """
    try:
        ms.meshing_close_holes(maxholesize=maxholesize)
        return
    except Exception as e:
        print(f"[WARN] 首次补洞失败，准备修复后重试: {e}")

    try:
        ms.meshing_repair_non_manifold_edges()
        ms.meshing_repair_non_manifold_vertices()
        ms.meshing_remove_unreferenced_vertices()
        ms.meshing_close_holes(maxholesize=maxholesize)
        return
    except Exception as e:
        print(f"[WARN] 二次补洞仍失败，跳过补洞并继续流程: {e}")


def _run_with_heartbeat(label: str, fn, interval_sec: int = 30):
    start = time.time()
    done = threading.Event()

    def _heartbeat():
        while not done.wait(interval_sec):
            elapsed = time.time() - start
            print(f"[{label}] still running... {elapsed:.1f}s")

    t = threading.Thread(target=_heartbeat, daemon=True)
    t.start()
    try:
        return fn()
    finally:
        done.set()
        elapsed = time.time() - start
        print(f"[{label}] done in {elapsed:.1f}s")


def _compute_scale_factors(
    vertices: list[tuple[float, float, float]],
    shapenet_normalize: bool,
    shapenet_radius: float,
    post_normalize_scale: float,
) -> tuple[float, float]:
    if shapenet_normalize:
        max_norm = 0.0
        for x, y, z in vertices:
            norm = (x * x + y * y + z * z) ** 0.5
            if norm > max_norm:
                max_norm = norm

        if max_norm <= 1e-12:
            shapenet_scale = 1.0
        else:
            shapenet_scale = float(shapenet_radius) / max_norm
    else:
        shapenet_scale = 1.0

    return shapenet_scale, shapenet_scale * post_normalize_scale


def _apply_obj_scale_inplace(
    obj_path: str,
    shapenet_normalize: bool,
    shapenet_radius: float,
    post_normalize_scale: float,
    quantize_decimals: int,
) -> dict:
    source_path = Path(obj_path)
    temp_path = source_path.with_suffix(source_path.suffix + ".scale_tmp")

    vertices: list[tuple[float, float, float]] = []
    with source_path.open("r", encoding="utf-8", errors="ignore") as src:
        for line in src:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))

    shapenet_scale, final_scale = _compute_scale_factors(
        vertices,
        shapenet_normalize=shapenet_normalize,
        shapenet_radius=shapenet_radius,
        post_normalize_scale=post_normalize_scale,
    )

    number_format = f"{{:.{quantize_decimals}f}}" if quantize_decimals >= 0 else None
    with source_path.open("r", encoding="utf-8", errors="ignore") as src, temp_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as dst:
        for line in src:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    # Only scale coordinates. Do not apply any translation or rotation.
                    coords = [
                        float(parts[1]) * final_scale,
                        float(parts[2]) * final_scale,
                        float(parts[3]) * final_scale,
                    ]
                    if number_format is None:
                        payload = [str(value) for value in coords]
                    else:
                        payload = [number_format.format(value) for value in coords]
                    dst.write("v " + " ".join(payload) + "\n")
                    continue
            dst.write(line)

    temp_path.replace(source_path)

    return {
        "shapenet_normalize": shapenet_normalize,
        "shapenet_radius": shapenet_radius,
        "post_normalize_scale": post_normalize_scale,
        "quantize_decimals": quantize_decimals,
        "shapenet_scale": shapenet_scale,
        "final_scale": final_scale,
    }


def _write_scale_meta(meta_path: Path, payload: dict) -> None:
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _postprocess_after_poisson(ms: pymeshlab.MeshSet) -> None:
    print("[9/10] poisson重建后全量清理")
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_null_faces()
    ms.meshing_remove_unreferenced_vertices()

    ms.compute_selection_by_self_intersections_per_face()
    ms.meshing_remove_selected_faces()

    ms.compute_selection_bad_faces()
    ms.meshing_remove_selected_faces()

    ms.compute_selection_by_small_disconnected_components_per_face()
    ms.meshing_remove_selected_faces()
    ms.meshing_remove_unreferenced_vertices()

    ms.meshing_repair_non_manifold_edges()
    ms.meshing_repair_non_manifold_vertices()
    ms.meshing_re_orient_faces_coherently()
    ms.meshing_re_orient_faces_by_geometry()

    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_null_faces()
    ms.meshing_remove_unreferenced_vertices()
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_repair_non_manifold_vertices()

    print("[10/10] 重建后重算法向")
    ms.compute_normal_per_face()
    ms.compute_normal_per_vertex()

def clean_and_decimate_mesh(
    input_path: str,
    output_path: str,
    target_face_num: int = 300000,
    enable_post_scale: bool = False,
    shapenet_normalize: bool = True,
    shapenet_radius: float = 1.0,
    post_normalize_scale: float = 1.0,
    quantize_decimals: int = 4,
):
    """
    对带纹理的原始 Mesh 进行拓扑清洗与减面，并安全导出保留纹理的 obj 文件
    默认目标面片数设为 100w，以平衡仿真器性能与几何细节。
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    ms = pymeshlab.MeshSet()
    print(f"[1/10] 加载模型: {input_path}")
    ms.load_new_mesh(input_path)

    print("[2/10] 删除冗余顶点与重叠面")
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_null_faces()

    print("[3/10] 去噪")
    ms.meshing_remove_connected_component_by_diameter(
        mincomponentdiag=pymeshlab.PercentageValue(5.0)
    )

    print("[4/10] 处理非流形边与非流形顶点")
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_repair_non_manifold_vertices()

    print("[5/10] 处理自相交面")
    ms.compute_selection_by_self_intersections_per_face()
    ms.meshing_remove_selected_faces()

    print("[6/10] 处理 T-型接头 (合并极近顶点)")
    ms.meshing_merge_close_vertices(
        threshold=pymeshlab.PercentageValue(0.1)
    )    

    print("[7/10] 二次修复非流形，填补破洞，封闭平滑表面")
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_repair_non_manifold_vertices()
    ms.meshing_remove_unreferenced_vertices()
    _safe_close_holes(ms, maxholesize=100)

    print(f"[8/10] 减面 (目标面数: {target_face_num})")
    #voxelsize 越小，保留的细节越多。0.1% - 0.3% 是精细重构的黄金值。
    print("poisson重构")
    _run_with_heartbeat(
        "poisson",
        lambda: ms.generate_surface_reconstruction_screened_poisson(
            depth=11,               # 建议 10-12
            fulldepth=2,           # 保持根节点的完整性
            samplespernode=1.5,     # 较低的值（1.0-2.0）能更好地拟合原始点细节
            pointweight=4.0,        # 增加点权重，让生成的表面更贴合原始点云
            preclean=True           # 重建前自动清理一些微小干扰
        ),
        interval_sec=30,
    )

    print("执行高保真减面")
    _run_with_heartbeat(
        "decimation",
        lambda: ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=target_face_num,
            preservenormal=True,
            planarweight=0.1,          # 调高平面权重，降低大平面的细节，增加连接处细节
            boundaryweight=0.5,
        ),
        interval_sec=30,
    )

    # 拉普拉斯平滑（去除减面后的锯齿感）
    ms.apply_coord_laplacian_smoothing()

    _postprocess_after_poisson(ms)

    print(f"导出模型: {output_path}")
    ms.save_current_mesh(
        output_path,
        save_wedge_texcoord=True, 
        save_wedge_normal=True,      
        save_textures=True        
    )
    if enable_post_scale:
        print("[11/11] 导出后应用纯缩放（无平移、无旋转）")
        scale_info = _apply_obj_scale_inplace(
            output_path,
            shapenet_normalize=shapenet_normalize,
            shapenet_radius=shapenet_radius,
            post_normalize_scale=post_normalize_scale,
            quantize_decimals=quantize_decimals,
        )
        meta_path = Path(output_path).with_suffix(Path(output_path).suffix + ".scale_meta.json")
        _write_scale_meta(
            meta_path,
            {
                "output_obj": output_path,
                "scale_info": scale_info,
            },
        )
        print(f"缩放完成: final_scale={scale_info['final_scale']:.8f}")
        print(f"缩放元数据: {meta_path}")

    print("Down!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean, decimate mesh, and optionally apply post-scale.")
    parser.add_argument(
        "--input",
        required=True,
        help="输入 OBJ 网格路径；请填写待清洗的扫描或视觉 mesh。",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="输出 OBJ 网格路径；请填写清洗结果的保存位置。",
    )
    parser.add_argument("--target-face-num", type=int, default=3000, help="Target face number.")

    # Keep defaults aligned with mesh_clean_pro but turn scaling off by default.
    parser.add_argument("--enable-post-scale", action="store_true", help="Enable post-scale on exported OBJ.")
    parser.add_argument("--shapenet-normalize", action="store_true", default=True, help="Enable ShapeNet-style scale normalization.")
    parser.add_argument("--no-shapenet-normalize", dest="shapenet_normalize", action="store_false", help="Disable ShapeNet-style scale normalization.")
    parser.add_argument("--shapenet-radius", type=float, default=1.0, help="Target radius used by ShapeNet-style scaling.")
    parser.add_argument("--post-normalize-scale", type=float, default=1.0, help="Extra global scale multiplied after normalization.")
    parser.add_argument("--quantize-decimals", type=int, default=4, help="Decimal quantization for scaled vertex output. -1 keeps full precision.")
    args = parser.parse_args()

    start_time = time.time()
    clean_and_decimate_mesh(
        args.input,
        args.output,
        target_face_num=args.target_face_num,
        enable_post_scale=args.enable_post_scale,
        shapenet_normalize=args.shapenet_normalize,
        shapenet_radius=args.shapenet_radius,
        post_normalize_scale=args.post_normalize_scale,
        quantize_decimals=args.quantize_decimals,
    )
    end_time = time.time()
    print(f"Time: {end_time - start_time:.2f} seconds")
