# Mesh Generation and Voxelization Toolkit

## Overview

This toolkit generates and voxelizes 3D geometries from either parametric `.geo` templates (via Gmsh) or prebuilt STL surfaces. It targets simulation workflows (e.g., LBM), where consistent voxel grids, label semantics, and reproducible shapes are essential. Under the hood it uses `gmsh` for templated surface meshing and `trimesh` for voxelization and geometry ops, while keeping both pipelines numerically aligned.

### Key Features

- **Geometry Generation**: Generate 3D geometries based on `.geo` templates.
- **Voxelization**: Convert complex 3D geometries into voxelized representations.
- **Parallel Processing**: Split along the leading axis and voxelize segments (optionally in parallel) while stitching back to the same grid as a single-pass run.
- **Visualization**: Visualize the voxelized mesh using Mayavi.
- **Flexible Storage**: Save voxelized meshes as binary `.npy` files or as human-readable `.txt` files.

## Pipelines

- `.geo → STL → voxels` (templated route)
  - Parametrize templates in `meshgen/geo_templates/*.geo` using placeholder replacement, generate a surface with Gmsh, export STL, then voxelize the STL.
- `STL → voxels` (direct route)
  - Load a closed (watertight) STL and voxelize/fill directly. Absolute and relative STL paths are accepted.

Equivalence: Both routes produce the same simulation-ready output format and label semantics. Choose `.geo` for parametric design; choose STL for existing surfaces (e.g., scans or CAD exports). When `split` is provided, the output remains identical to the no‑split path by design.

Important details:
- Characteristic length: If a `.geo` template contains `DEFINE_H` and you do not pass `h`, it is inferred from `resolution` using `h ≈ max(1e-4, 1e-3 / resolution)`.
- Splitting: The mesh is segmented along the leading axis, each segment is voxelized (optionally in parallel via `ProcessPoolExecutor`), and the stitched occupancy is re-closed with a single-voxel binary dilation before the global fill and normalization. This seals sub-voxel cracks introduced by segmentation, preserves exact equivalence with the no‑split path, and keeps workloads scalable.
- Watertightness: The STL route expects a closed surface; a light repair is attempted (normals/winding and hole filling via trimesh) but you should supply closed inputs for reliable filling.
- Template resolution awareness: The `.geo` route forwards `resolution` into templates as `DEFINE_RESOLUTION`. For example, the junction template applies a sub-voxel axial pad derived from `resolution` so the east outlet caps remain filled identically in both pipelines.

## Requirements

The package requires the following Python libraries:

- `gmsh` (only needed for the `.geo` templated route; STL-only workflows can omit it)
- `trimesh`
- `numpy`
- `scipy`
- `mayavi`
- `tqdm`

## Installation

There are two options: via Conda (recommended for Mayavi/VTK) or plain pip.

### Option A: Conda environment (recommended)

1) Create the environment from `environment.yml` at the repo root:

```bash
conda env create -f environment.yml
# or, if you have mamba:
# mamba env create -f environment.yml
```

2) Activate it:

```bash
conda activate meshgen
```

The environment installs all runtime deps (numpy, scipy, trimesh, gmsh, mayavi, vtk, pyqt, tqdm) and
installs this repo in editable mode for development.

Notes:
- If you are on a headless server and do not need visualization, you can remove `mayavi`, `vtk`, and `pyqt`
  from `environment.yml` before creating the environment.
- Gmsh is provided from conda-forge with the Python API. Ensure OpenGL support is available for Mayavi.

### Option B: Pip-only install

If you already have a working Python environment and do not need Mayavi/VTK from Conda:

```bash
pip install -e .
```

You will also need to install the runtime dependencies yourself:

```bash
pip install numpy scipy trimesh gmsh mayavi tqdm
```

If you only rely on the STL route you may skip installing `gmsh`.

## Usage

Here is a quick example of how to use the `Geometry` class to generate, voxelize, and visualize a 3D geometry.

```python
from meshgen.geometry import Geometry

# Initialize a Geometry instance with parameters
geom = Geometry(name="tcpc_classic", resolution=5)

# Generate the voxel mesh based on the .geo template
geom.generate_voxel_mesh()

# Save the voxel mesh as a binary .npy file
geom.save_voxel_mesh("tcpc_voxel_mesh.npy")

# Save the voxel mesh as a human-readable text file
geom.save_voxel_mesh_to_text("tcpc_voxel_mesh.txt")

# Visualize the voxel mesh
geom.visualize()

```

### Quick example: junction_2d visualization (Mayavi, single process)

This runs the `meshgen/geo_templates/junction_2d_template.geo` template, voxelizes with a modest resolution, and visualizes with Mayavi. It does not split or parallelize the voxelization.

Run the provided example script:

```bash
conda activate meshgen
python examples/junction_2d_visualize.py
```

Notes:
- The junction template exposes these controls via placeholders: `LOWER_ANGLE`, `UPPER_ANGLE`, `UPPER_FLARE`, `LOWER_FLARE`.
- `OFFSET` is fixed to 0.0 in the template as provided; you can edit the template if you want to parameterize it.

### STL route: voxelize a closed STL directly

You can now bypass `.geo` templates and voxelize a closed STL directly via the `Geometry` class by providing `stl_path`:

```python
from meshgen.geometry import Geometry

geom = Geometry(stl_path="examples/glenn_capped.stl", resolution=4, split=None)
geom.generate_voxel_mesh()
geom.save_voxel_mesh_to_text("glenn_voxel_mesh.txt")  # writes geom_/dim_/val_ files
```

This route produces the same output format as the `.geo` route and supports segment-based splitting with the same semantics.

## Geometry API (Python)

- `Geometry(name=None, resolution=1, split=None, num_processes=1, output_dir="output", expected_in_outs=None, stl_path=None, **kwargs)`
  - `.geo` route: set `name="<template>"` and pass template placeholders in `kwargs` (e.g., `lower_angle`, `upper_flare`, …). If the template expects `DEFINE_H` and you don’t pass `h`, it is inferred from `resolution`.
  - STL route: set `stl_path="path/to/closed.stl"` and leave `name=None`.
  - `resolution`: longest axis ~ 128*resolution voxels (adjusted to match Trimesh’s voxel grid behavior).
  - `split`: optional integer to segment along the leading axis and stitch a globally filled lattice; `num_processes` controls parallel workers.
  - `expected_in_outs`: iterable or dict of boundary tags to apply — any of `{N,E,S,W,B,F}` enables the corresponding domain wall labels (see Labels below). Dict inputs use truthy values to toggle faces.
- Methods:
  - `generate_voxel_mesh()` → computes `voxelized_mesh` as a boolean array.
  - `get_voxel_mesh()` → returns voxel array.
  - `save_voxel_mesh(path="voxel_mesh.npy")` → saves `.npy`.
  - `save_voxel_mesh_to_text(path="voxel_mesh.txt")` → writes triplet text output (geom_/dim_/val_ files).
  - `generate_lbm_mesh()` + `save_lbm_mesh_to_text("lbm_mesh.txt")` → LBM-ready labels in the same triplet format.

## Output Format

Text export writes a triplet alongside your chosen filename prefix in the target output folder:
- `geom_<name>.txt` — rows `x y z value` for every voxel
- `dim_<name>.txt` — a single line `Nx Ny Nz`
- `val_<name>.txt` — reserved for downstream solver values (currently empty)

Binary export uses NumPy `.npy` with dtype `bool` for occupancy. The text triplet is the stable format expected by downstream simulation and must not change in naming or ordering.

## Labels and Boundaries

Default labeled volume semantics used by downstream solvers:
- 0 — empty/outside
- 1 — fluid/interior (occupancy)
- 2 — wall/solid (one-voxel shell immediately outside the fluid region)
- 3/4/5 — near-wall bands: first, second, and third bands into the fluid for optional modeling

Optional domain wall tags can be applied on the fluid layer if you pass `expected_in_outs` containing any of `{N, E, S, W, B, F}`. The mapping is applied on the corresponding domain faces:
- N (z max) → 11, E (x max) → 12, S (z min) → 13, W (x min) → 14, B (y max) → 15, F (y min) → 16

When exporting text, missing domain faces are padded by a single wall layer to ensure presence and consistency for downstream codes.

## Splitting & Parallelization

- `split=<int>` divides the mesh along the leading axis, voxelizes each segment, and stitches the results; `num_processes` controls the number of workers used for segment voxelization.
- Segments overlap at their boundaries to avoid gaps; after stitching we clamp indices to the global lattice, run a one-voxel binary dilation to seal seams, and then apply a single global fill so the result matches the no‑split grid exactly.
- Shapes are normalized to the expected target derived from bounds and pitch, ensuring split and no‑split runs are identical.
- For large models, start with small `resolution` (1–2) before scaling up; increase `split`/`num_processes` as needed to stay within memory limits.

Pitch and shape: The voxel pitch is chosen so the longest axis approximates `128 * resolution` cells. Trimesh’s VoxelGrid yields `N ≈ ceil(extent / pitch) + 1` per axis; we set pitch from the longest extent to achieve the target and then normalize results to match the expected dims.

LBM shape normalization: For LBM grid completion, the leading dimension must be a multiple of 128; the remaining dimensions are rounded up to the nearest multiple of 32.

Watertightness note: Splitting assumes robust slicing of a closed surface. For STL inputs, ensure watertightness for best results; light repairs (normals/winding, small hole filling) are attempted but not guaranteed.

Performance: Avoids per-voxel Python loops and relies on NumPy/Scipy vectorization; segment voxelization can use a `ProcessPoolExecutor` when `num_processes > 1`.

## Troubleshooting

- Module import: If running from a repo checkout without installing, use `pip install -e .` or add the repo root to `PYTHONPATH`.
- Mayavi/VTK on headless: Remove `mayavi`, `vtk`, `pyqt` from the environment if not visualizing; or configure an X server.
- Gmsh issues: Ensure `gmsh` from conda-forge is installed; templates rely on OpenCASCADE features. If Gmsh crashes, verify `.geo` placeholders match the parameters you provide.

## Testing Guidance

- Quick shape sanity: with `split=None`, the longest axis should equal `≈ 128 * resolution`.
- Split equivalence: split and no‑split runs should match dimensions after normalization.
- Use small `resolution` (1–2) for fast checks; verify both routes (.geo and STL) produce the same dims.
- Validate label distributions: expect values in `{0,1,2,3,4,5}` and optionally `{11..16}` when domain tags are requested.
- Confirm `dim_*` matches the in-memory array shape and that `geom_*` encodes `x y z value` in that shape.

## Documentation

Detailed module guides with inline examples live under `documentation/`:
- `documentation/geometry.md` — high-level API and workflows
- `documentation/mesher.md` — Gmsh templating and STL generation
- `documentation/voxels.md` — voxelization, splitting, labeling, and export
- `documentation/utilities.md` — visualization and text export helpers

If you change code or behavior, update both this README and the relevant module guides.

## Contributing

- Keep both pipelines equivalent (.geo→STL→voxels and STL→voxels) in output and label semantics.
- Preserve existing APIs; add new ones without breaking current callers.
- Be surgical in changes; match the existing concise Python style.
- Document changes: update README, examples if affected, and `documentation/*.md` accordingly.
