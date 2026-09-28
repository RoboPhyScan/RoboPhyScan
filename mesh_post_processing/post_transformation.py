"""
用于“矩阵求逆 + 网格逆变换”的工具脚本。

本脚本主要解决两类需求：

1) 对文本形式保存的矩阵（如 4x4 仿射变换矩阵）求逆，并以同样的文本格式写回。
2) 读取 4x4 变换矩阵，将其应用到指定的 mesh（OBJ 等），并导出变换后的模型。

矩阵文本格式要求：
- 每行一行数值，使用空格或逗号分隔；
- 支持空行；
- 对 mesh 变换场景，要求矩阵为 4x4。
"""

from pathlib import Path

import numpy as np


def _default_output_path(input_path: Path) -> Path:
    """
    为矩阵逆文件生成默认输出路径。

    例如：
    - input.txt -> input.inv.txt
    - input.mat -> input.inv.mat
    """
    if input_path.suffix:
        return input_path.with_name(f"{input_path.stem}.inv{input_path.suffix}")
    return input_path.with_name(f"{input_path.name}.inv")


def _default_mesh_output_path(input_mesh_path: Path) -> Path:
    """
    为 mesh 逆变换结果生成默认输出路径。

    例如：
    - input.obj -> input.inv.obj
    """
    if input_mesh_path.suffix:
        return input_mesh_path.with_name(
            f"{input_mesh_path.stem}.inv{input_mesh_path.suffix}"
        )
    return input_mesh_path.with_name(f"{input_mesh_path.name}.inv")


def read_matrix_txt(path: Path) -> np.ndarray:
    """
    读取文本矩阵文件（空格或逗号分隔）。

    :param path: 文本矩阵路径
    :return: numpy 矩阵（二维）
    """
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            rows.append([float(p) for p in parts])
    matrix = np.array(rows, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"输入矩阵维度不正确: {matrix.ndim}D")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"输入矩阵不是方阵: {matrix.shape}")
    return matrix


def write_matrix_txt(path: Path, matrix: np.ndarray, precision: int) -> None:
    """
    将矩阵以文本形式写回文件。

    :param path: 输出路径
    :param matrix: 二维矩阵
    :param precision: 小数位数
    """
    if matrix.ndim != 2:
        raise ValueError(f"输出矩阵维度不正确: {matrix.ndim}D")
    fmt = f"{{:.{precision}f}}"
    with path.open("w", encoding="utf-8") as f:
        for row in matrix:
            f.write(" ".join(fmt.format(float(v)) for v in row) + "\n")


def post_transformation(
    input_path: str | Path,
    output_path: str | Path | None = None,
    precision: int = 12,
) -> Path:
    """
    对输入矩阵求逆，并写出逆矩阵文件。

    :param input_path: 输入矩阵 txt 路径
    :param output_path: 输出矩阵 txt 路径（不传则生成 *.inv.*）
    :param precision: 输出小数位数
    :return: 逆矩阵文件路径
    """
    input_path = Path(input_path)
    resolved_output_path = (
        Path(output_path)
        if output_path is not None
        else _default_output_path(input_path)
    )

    matrix = read_matrix_txt(input_path)
    inv_matrix = np.linalg.inv(matrix)
    write_matrix_txt(resolved_output_path, inv_matrix, precision=precision)
    return resolved_output_path


def inverse_transform_mesh_by_matrix_txt(
    input_mesh_path: str | Path,
    matrix_txt_path: str | Path,
    output_mesh_path: str | Path | None = None,
    freeze: bool = True,
    compose: bool = False,
    save_textures: bool = True,
    save_wedge_texcoord: bool = True,
    save_wedge_normal: bool = True,
) -> Path:
    """
    将 4x4 变换矩阵应用到 mesh，并导出结果。

    :param input_mesh_path: 输入 mesh 路径（如 .obj）
    :param matrix_txt_path: 4x4 矩阵 txt 路径
    :param output_mesh_path: 输出 mesh 路径（不传则生成 *.inv.*）
    :param freeze: 是否将变换“冻结”到顶点坐标（True 时顶点坐标会被真正修改）
    :param compose: 是否与当前矩阵进行复合（compose=True 为叠加；False 为直接设定）
    :param save_textures: 保存纹理引用（对带纹理 OBJ 建议为 True）
    :param save_wedge_texcoord: 保存 wedge 纹理坐标
    :param save_wedge_normal: 保存 wedge 法向
    :return: 输出 mesh 路径
    """
    try:
        import pymeshlab
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "未检测到 pymeshlab。请先安装：pip install pymeshlab"
        ) from e

    input_mesh_path = Path(input_mesh_path)
    matrix_txt_path = Path(matrix_txt_path)
    resolved_output_path = (
        Path(output_mesh_path)
        if output_mesh_path is not None
        else _default_mesh_output_path(input_mesh_path)
    )
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    matrix = read_matrix_txt(matrix_txt_path)
    if matrix.shape != (4, 4):
        raise ValueError(f"期望 4x4 矩阵，实际为: {matrix.shape}")

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(input_mesh_path))

    applied = False
    if hasattr(ms, "apply_filter"):
        try:
            ms.apply_filter(
                "set_matrix",
                transformmatrix=matrix,
                compose=compose,
                freeze=freeze,
            )
            applied = True
        except Exception:
            applied = False

    if not applied and hasattr(ms, "set_matrix"):
        ms.set_matrix(
            transformmatrix=matrix,
            compose=compose,
            freeze=freeze,
        )
        applied = True

    if not applied:
        raise RuntimeError("无法应用矩阵：未找到可用的 set_matrix 接口/滤镜。")

    ms.save_current_mesh(
        str(resolved_output_path),
        save_textures=save_textures,
        save_wedge_texcoord=save_wedge_texcoord,
        save_wedge_normal=save_wedge_normal,
    )
    ms.clear()
    return resolved_output_path


def inverse_transform_pipeline(
    input_mat: str | Path,
    input_obj: str | Path,
    output_inv_mat: str | Path | None = None,
    output_inv_obj: str | Path | None = None,
    precision: int = 12,
    freeze: bool = True,
    compose: bool = False,
    save_textures: bool = True,
    save_wedge_texcoord: bool = True,
    save_wedge_normal: bool = True,
) -> tuple[Path, Path]:
    """
    综合 pipeline：对输入矩阵求逆，然后将“逆矩阵”应用到输入 OBJ。

    输入要求：
    - input_mat: 文本矩阵文件（通常为 4x4）
    - input_obj: 需要执行逆变换的模型（OBJ 等）

    输出：
    - 输出逆矩阵文件路径
    - 输出逆变换 mesh 路径
    """
    inv_mat_path = post_transformation(
        input_path=input_mat,
        output_path=output_inv_mat,
        precision=precision,
    )
    inv_obj_path = inverse_transform_mesh_by_matrix_txt(
        input_mesh_path=input_obj,
        matrix_txt_path=inv_mat_path,
        output_mesh_path=output_inv_obj,
        freeze=freeze,
        compose=compose,
        save_textures=save_textures,
        save_wedge_texcoord=save_wedge_texcoord,
        save_wedge_normal=save_wedge_normal,
    )
    return inv_mat_path, inv_obj_path


if __name__ == "__main__":
    pass
