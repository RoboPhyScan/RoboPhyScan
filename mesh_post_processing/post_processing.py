import pymeshlab


def outlier_removal(
    ms,
    propthreshold=0.990000,
    knearest=16,
    maxholesize=30,
    refineholeedgelen_percent=3.000000,
):
    """
    用于过滤点云中的离群点，保留大多数点的邻域内的点。
    注意，操作后易于导致模型非水密，故后加入补洞。
    :param propthreshold: 离群点的比例阈值，默认为0.98
    :param knearest: 用于计算邻域的最近邻数量，默认为16
    :param maxholesize: 补洞时允许的最大洞穴大小，默认为30
    :param refineholeedgelen_percent: 补洞时细化边长的百分比，默认为3.0
    """
    ms.compute_selection_point_cloud_outliers(
        propthreshold=propthreshold, knearest=knearest
    )
    ms.meshing_remove_selected_vertices()
    ms.meshing_close_holes(
        maxholesize=maxholesize,
        refineholeedgelen=pymeshlab.PercentageValue(refineholeedgelen_percent),
    )


def surface_reconstruction(
    ms, depth=8, samplespernode=1.200000, pointweight=4.000000, threads=15
):
    """
    用于从点云重建表面网格。注意，对模型质量影响较大，建议根据实际情况调整参数或使用。
    :param depth: 八叉树的深度，默认为8
    :param samplespernode: 每个八叉树节点的采样点数量，默认为1.2
    :param pointweight: 点权重，默认为4.0
    :param threads: 使用的线程数量，默认为15
    """
    ms.generate_surface_reconstruction_screened_poisson(
        depth=depth,
        samplespernode=samplespernode,
        pointweight=pointweight,
        preclean=True,
        threads=threads,
    )


def self_intersection_removal(ms):
    """
    用于移除网格中的自交部分。
    """
    ms.compute_selection_by_self_intersections_per_face()
    ms.meshing_remove_selected_faces()


def repair_non_manifold_edges(
    ms, method=0, maxholesize=30, refineholeedgelen_percent=3.000000
):
    """
    用于修复网格中的非流形边。
    注意，操作后易于导致模型非水密，故后加入补洞。
    :param method: 修复方法，默认为0
    0: 通过删除面来移除非流形边（对于每一条非流形边，算法会反复迭代删除面积最小的面，直到它变成正常的 2-流形结构），
    1: 或者通过分离顶点来移除（这样每一条非流形边链都会变成一个开放边界）
    :param maxholesize: 补洞时允许的最大洞穴大小，默认为30
    :param refineholeedgelen_percent: 补洞时细化边长的百分比，默认为3.0
    """
    ms.meshing_repair_non_manifold_edges(method=method)
    ms.meshing_close_holes(
        maxholesize=maxholesize,
        refineholeedgelen=pymeshlab.PercentageValue(refineholeedgelen_percent),
    )


def simplify_mesh_without_texture(
    ms,
    targetfacenum=300000,
    targetperc=0.000000,
    qualitythr=0.300000,
    boundaryweight=1.000000,
    planarweight=0.002000,
):
    """
    适用于白模，用于简化网格以减少面数。
    :param targetfacenum: 目标面数，默认为300000
    :param targetperc: 目标面数的百分比，默认为0.0
    :param qualitythr: 面质量阈值，默认为0.3
    :param boundaryweight: 边界权重，默认为1.0
    :param planarweight: 平面权重，默认为0.002000
    """
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=targetfacenum,
        targetperc=targetperc,
        qualitythr=qualitythr,
        boundaryweight=boundaryweight,
        planarweight=planarweight,
    )


def simplify_mesh_with_texture(
    ms,
    targetfacenum=300000,
    targetperc=0.000000,
    qualitythr=0.300000,
    boundaryweight=1.000000,
):
    """
    适用于带有纹理的网格，用于简化网格以减少面数。
    注意，同文件夹下需保存mtl和贴图的jpg或png文件。
    :param targetfacenum: 目标面数，默认为300000
    :param targetperc: 目标面数的百分比，默认为0.0
    :param qualitythr: 面质量阈值，默认为0.3
    :param boundaryweight: 边界权重，默认为1.0
    """
    ms.meshing_decimation_quadric_edge_collapse_with_texture(
        targetfacenum=targetfacenum,
        targetperc=targetperc,
        qualitythr=qualitythr,
        boundaryweight=boundaryweight,
    )


def ms_load_and_save(input_mesh_path, output_mesh_path, function, *args, **kwargs):
    """
    加载网格，执行指定的处理函数，并保存结果。
    :param input_mesh_path: 输入网格文件路径
    :param output_mesh_path: 输出网格文件路径
    :param function: 要执行的处理函数
    :param args: 传递给处理函数的位置参数
    :param kwargs: 传递给处理函数的关键词参数
    """
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(input_mesh_path)
    function(ms, *args, **kwargs)
    ms.save_current_mesh(output_mesh_path)
    ms.clear()


def pipline_default_without_texture(
    input_mesh_path,
    output_mesh_path,
    bool_outlier_removal=True,
    bool_surface_reconstruction=False,
    bool_self_intersection_removal=True,
    bool_repair_non_manifold_edges=True,
    bool_simplify_mesh_without_texture=True,
    targetfacenum=300000,
):
    """
    默认的处理流程，适用于白模。
    :param input_mesh_path: 输入网格文件路径
    :param output_mesh_path: 输出网格文件路径
    :param bool_outlier_removal: 是否执行离群点移除，默认为True
    :param bool_surface_reconstruction: 是否执行表面重建，默认为False
    :param bool_self_intersection_removal: 是否执行自交移除，默认为True
    :param bool_repair_non_manifold_edges: 是否执行非流形边修复，默认为True
    :param bool_simplify_mesh_without_texture: 是否执行网格简化，默认为True
    :param targetfacenum: 网格简化的目标面数，默认为300000
    """
    if bool_simplify_mesh_without_texture:
        ms_load_and_save(
            input_mesh_path,
            output_mesh_path,
            simplify_mesh_without_texture,
            targetfacenum=targetfacenum,
        )
    if bool_outlier_removal:
        ms_load_and_save(output_mesh_path, output_mesh_path, outlier_removal)
    if bool_surface_reconstruction:
        ms_load_and_save(output_mesh_path, output_mesh_path, surface_reconstruction)
    if bool_self_intersection_removal:
        ms_load_and_save(output_mesh_path, output_mesh_path, self_intersection_removal)
    if bool_repair_non_manifold_edges:
        ms_load_and_save(output_mesh_path, output_mesh_path, repair_non_manifold_edges)


def pipline_default_with_texture(
    input_mesh_path, output_mesh_path, targetfacenum=300000
):
    """
    默认的处理流程，适用于带有纹理的网格。
    注意，同文件夹下需保存mtl和贴图的jpg或png文件。
    :param input_mesh_path: 输入网格文件路径
    :param output_mesh_path: 输出网格文件路径
    :param targetfacenum: 网格简化的目标面数，默认为300000
    """
    ms_load_and_save(
        input_mesh_path,
        output_mesh_path,
        simplify_mesh_with_texture,
        targetfacenum=targetfacenum,
    )


if __name__ == "__main__":
    input_mesh_path = "input.obj"
    output_mesh_path = "output_mesh.obj"
    pipline_default_without_texture(input_mesh_path, output_mesh_path)
