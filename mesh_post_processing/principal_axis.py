import pymeshlab


def align_to_principal_axis(ms):
    """
    基于 MeshLab 的 “Transform: Align to Principal Axis” 过滤器生成对齐到主惯性轴的变换矩阵。
    :param pointsflag: 是否仅使用顶点参与主轴计算；对于点云/非水密网格建议为 True
    :param freeze: 是否将变换矩阵冻结到顶点坐标（True 时会真正改变顶点坐标）
    :param alllayers: 是否对所有可见图层同时应用
    """
    ms.compute_matrix_by_principal_axis()


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


def align_axis(input_mesh_path, output_mesh_path):
    """
    对齐网格到主惯性轴。
    """
    ms_load_and_save(
        input_mesh_path,
        output_mesh_path,
        function=align_to_principal_axis,
    )


if __name__ == "__main__":
    align_axis(
        input_mesh_path="input.obj",
        output_mesh_path="output_principal_axis.obj",
    )
