import os
import time

import pymeshlab


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
    input_path: str, output_path: str, target_face_num: int = 1000000
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
    ms.meshing_merge_close_vertices(threshold=pymeshlab.PercentageValue(0.1))

    print("[7/10] 填补破洞，封闭平滑表面")
    ms.meshing_close_holes(maxholesize=100)

    print(f"[8/10] 减面 (目标面数: {target_face_num})")
    # voxelsize 越小，保留的细节越多。0.1% - 0.3% 是精细重构的黄金值。
    print("poisson重构")
    ms.generate_surface_reconstruction_screened_poisson(
        depth=11,  # 建议 10-12
        fulldepth=2,  # 保持根节点的完整性
        samplespernode=1.5,  # 较低的值（1.0-2.0）能更好地拟合原始点细节
        pointweight=4.0,  # 增加点权重，让生成的表面更贴合原始点云
        preclean=True,  # 重建前自动清理一些微小干扰
    )

    print("执行高保真减面")
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=target_face_num,
        preservenormal=True,
        planarweight=0.1,  # 调高平面权重，降低大平面的细节，增加连接处细节
        boundaryweight=0.5,
    )

    # 拉普拉斯平滑（去除减面后的锯齿感）
    ms.apply_coord_laplacian_smoothing()

    _postprocess_after_poisson(ms)

    print(f"导出模型: {output_path}")
    ms.save_current_mesh(
        output_path,
        save_wedge_texcoord=True,
        save_wedge_normal=True,
        save_textures=True,
    )
    print("Down!")


if __name__ == "__main__":
    start_time = time.time()
    input_obj_file = "input.obj"
    output_obj_file = "output_processed.obj"

    # input_obj_file = "E:/SJTU/study/cs/Standard Concept Template Library/Mesh_Segmentation/high/high/xiangji1.obj"
    # output_obj_file = "E:/SJTU/study/cs/Standard Concept Template Library/Mesh_Segmentation/high/high/xiangji1_processed.obj"

    clean_and_decimate_mesh(input_obj_file, output_obj_file)
    end_time = time.time()
    print(f"Time: {end_time - start_time:.2f} seconds")
