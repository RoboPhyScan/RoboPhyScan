# 3D Mesh Post-Processing Scripts

This repository contains a set of Python scripts based on `pymeshlab` for handling common 3D mesh and point-cloud issues, including:

- mesh cleanup
- self-intersection and non-manifold repair
- surface reconstruction
- mesh decimation for untextured and textured models
- principal inertia axis alignment

The project currently includes four main scripts for general-purpose processing, a full cleanup pipeline, a principal-axis alignment example, and matrix-based inverse transformation utilities.

## Repository Structure

```text
mesh_post_processing/
|- post_processing.py
|- mesh_clean_process.py
|- principal_axis.py
|- post_transformation.py
`- README.md
```

## Requirements

Python 3 is recommended. Install `pymeshlab` and `numpy` first:

```bash
pip install pymeshlab numpy
```

## File Overview

### `post_processing.py`

A general-purpose mesh post-processing script that provides reusable utility functions and two default pipelines.

Core functions:

- `outlier_removal`: removes outliers from a point cloud and fills holes
- `surface_reconstruction`: performs surface reconstruction based on Screened Poisson
- `self_intersection_removal`: deletes self-intersecting faces
- `repair_non_manifold_edges`: repairs non-manifold edges and fills holes
- `simplify_mesh_without_texture`: decimates untextured meshes
- `simplify_mesh_with_texture`: decimates meshes while preserving textures
- `ms_load_and_save`: shared helper for loading, processing, and saving meshes

Default pipelines:

- `pipline_default_without_texture`: for untextured meshes; can conditionally run decimation, outlier filtering, surface reconstruction, self-intersection removal, and non-manifold repair
- `pipline_default_with_texture`: for textured meshes; performs texture-preserving decimation

Typical use cases:

- quickly calling a single processing step
- chaining multiple basic steps into a simplified workflow
- reusing the functions in other scripts or projects

### `mesh_clean_process.py`

A complete cleanup and decimation pipeline for raw textured meshes. Its core function is `clean_and_decimate_mesh`.

Default processing order:

1. Remove duplicated vertices, duplicated faces, and null faces
2. Remove small noisy connected components
3. Repair non-manifold edges and non-manifold vertices
4. Delete self-intersecting faces
5. Merge extremely close vertices
6. Fill holes
7. Run Screened Poisson surface reconstruction
8. Run high-fidelity decimation
9. Apply Laplacian smoothing
10. Clean again after reconstruction and recompute normals

Function signature:

```python
clean_and_decimate_mesh(
    input_path: str,
    output_path: str,
    target_face_num: int = 1000000,
)
```

Parameters:

- `input_path`: input mesh path, typically a textured `.obj`
- `output_path`: output mesh path; the script automatically creates the target directory
- `target_face_num`: target face count, default is `1000000`

The export step explicitly preserves:

- texture coordinates
- wedge normals
- texture file references

Typical use cases:

- cleaning raw textured meshes generated from scans
- unified post-processing after Poisson reconstruction
- exporting large meshes as usable OBJ files after decimation

### `principal_axis.py`

Demonstrates how to use `pymeshlab` to align a model to its principal inertia axes.

Included functions:

- `align_to_principal_axis`: computes the alignment matrix for principal-axis alignment
- `align_axis`: loads a mesh, aligns it, and saves the result

Typical use cases:

- normalizing model orientation before post-processing
- aligning meshes to principal axes for later analysis or batch processing

### `post_transformation.py`

A utility script for matrix inversion and inverse mesh transformation. Its core features include:

- `post_transformation`: reads a matrix txt file, computes the inverse matrix, and writes it out
- `inverse_transform_mesh_by_matrix_txt`: reads a `4x4` matrix txt file, applies it to a mesh, and exports the result
- `inverse_transform_pipeline`: an integrated pipeline that takes `input_mat` and `input_obj`, then automatically performs matrix inversion and inverse mesh transformation

Matrix text format:

- one row of numeric values per line, separated by spaces or commas
- blank lines are allowed
- the matrix must be `4x4` when used for mesh transformation

Notes for textured models:

- the inverse transformation itself does not damage UVs or texture coordinates
- if the input is a textured OBJ, make sure the `.mtl` and texture file paths are valid, and enable `save_textures/save_wedge_texcoord` when saving

## Quick Start

### 1. Default pipeline for untextured meshes

```python
from post_processing import pipline_default_without_texture

pipline_default_without_texture(
    input_mesh_path="input.obj",
    output_mesh_path="output.obj",
    bool_outlier_removal=True,
    bool_surface_reconstruction=False,
    bool_self_intersection_removal=True,
    bool_repair_non_manifold_edges=True,
    bool_simplify_mesh_without_texture=True,
    targetfacenum=300000,
)
```

### 2. Decimation for textured meshes

```python
from post_processing import pipline_default_with_texture

pipline_default_with_texture(
    input_mesh_path="textured_input.obj",
    output_mesh_path="textured_output.obj",
    targetfacenum=300000,
)
```

### 3. Full cleanup and decimation

```python
from mesh_clean_process import clean_and_decimate_mesh

clean_and_decimate_mesh(
    input_path="textured_input.obj",
    output_path="cleaned_output.obj",
    target_face_num=1000000,
)
```

### 4. Principal-axis alignment

```python
from principal_axis import align_axis

align_axis(
    input_mesh_path="input.obj",
    output_mesh_path="aligned_output.obj",
)
```

### 5. Matrix inversion and inverse mesh transformation (integrated pipeline)

```python
from post_transformation import inverse_transform_pipeline

inv_mat_path, inv_obj_path = inverse_transform_pipeline(
    input_mat="transform.txt",
    input_obj="aligned.obj",
    output_inv_mat="transform.inv.txt",
    output_inv_obj="aligned.inv.obj",
    freeze=True,
    compose=False,
)
print(inv_mat_path)
print(inv_obj_path)
```

## Run the Scripts Directly

All scripts keep `__main__` examples and can be run directly:

```bash
python post_processing.py
python mesh_clean_process.py
python principal_axis.py
python post_transformation.py
```

Before running them, it is recommended to replace the example input and output paths at the bottom of each script with your own paths.

## Usage Tips

- for untextured meshes, start with `post_processing.py` so you can combine steps flexibly
- for raw scanned textured meshes, prefer `mesh_clean_process.py`
- if the input is a textured OBJ, make sure the `.mtl` and texture files are accessible
- Screened Poisson reconstruction changes the mesh surface structure, so output quality depends on input quality
- very large models may require substantial memory and time during reconstruction and decimation

## Notes

- this repository is currently more of a script collection than a packaged CLI tool or Python package
- for batch processing, it is recommended to build your own loop and logging logic on top of the existing functions
