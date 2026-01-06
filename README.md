# Mesh Generation and Voxelization Toolkit

## Overview

This toolkit generates and voxelizes 3D geometries from either parametric `.geo` templates (via Gmsh) or prebuilt STL surfaces. It targets simulation workflows (e.g., LBM), where consistent voxel grids, label semantics, and reproducible shapes are essential. Under the hood it uses `gmsh` for templated surface meshing and `trimesh` for voxelization and geometry ops, while keeping both pipelines numerically aligned.

### Key Features

- **Geometry Generation**: Generate 3D geometries based on `.geo` templates.
- **Voxelization**: Convert complex 3D geometries into voxelized representations.
- **Parallel Processing**: Split along the leading axis and voxelize segments (optionally in parallel) while stitching back to the same grid as a single-pass run.
- **Visualization**: Visualize the voxelized mesh using Mayavi (optional extra).
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

Core requirements:

- `numpy`
- `scipy`
- `trimesh`
- `gmsh` (only needed for the `.geo` templated route; STL-only workflows can omit it)
- `tqdm`

Optional visualization:

- `mayavi` (install with `.[vis]` or use the conda environment; not available on Python 3.13)

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

If you already have a working Python environment:

```bash
pip install -e .
```

Optional visualization stack (Mayavi):

```bash
pip install -e .[vis]
```

If you only rely on the STL route you may skip installing `gmsh` by using `--no-deps` and adding only what you need.

Notes:
- Mayavi does not ship wheels for Python 3.13; use the conda environment (Python 3.10) or a Python 3.10/3.11 venv for visualization.
- The project now ships a `pyproject.toml`; ensure you are using `pip >= 21` so editable installs work with the PEP 517 backend.
- Add `.[dev]` to pick up optional development dependencies (e.g., `pip install -e .[dev]` for Black).

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
Visualization requires Mayavi; install `meshgen[vis]` or use the conda environment.

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

### Example: glenn_extended STL (north inlet, dual outlets)

The repository ships `glenn_extended.stl` (N inlet, E/W outlets). The STL is now fully capped on the N/E/W openings and sanitized to drop the historical 8-triangle speck that produced a trailing fluid voxel during discretization. Run the helper script to voxelize it at resolution 4, apply domain wall tags for `{N, E, W}`, write the geom_/dim_/val_ triplet into `examples/output/`, and open a Mayavi window:

```bash
conda activate meshgen
python examples/glenn_extended_visualize.py
```

The script uses the STL → voxels route, so you can swap in another watertight STL by pointing `Geometry(stl_path=...)` to a different file.

Need to regenerate the cleaned STL? Use the helper in `scripts/repair_glenn_stl.py` to drop stray fragments and copy the watertight, capped surface wherever it is needed:

```bash
python scripts/repair_glenn_stl.py --input examples/glenn_capped.stl --output glenn_extended.stl
```

### Minimal STL voxelization + Mayavi view

For a bare-bones STL → voxels → Mayavi pass (no text output), voxelize the root-level `master_combined_capped.stl` at resolution 4. The script also prints the STL bounds and physical size (same units as the STL) before voxelization so you can confirm the scale:

```bash
conda activate meshgen
python examples/master_combined_visualize.py
```

### Fantom pipeline: taper + target voxel count + per-end labels

The root-level `fantom.py` script wraps the `tcpc_taper_ends.py` end-taper workflow, then voxelizes using a target number of voxels on the longest axis (`LONGEST_AXIS_VOXELS`) instead of a resolution factor. It assigns distinct inlet/outlet labels on the Y extremes: one end on `y_max` and three separate ends on `y_min` (default labels 21, 22, 23, 24). Standard voxel labels (0/1/2/3/4/5) are preserved.

Run:

```bash
conda activate meshgen
python fantom.py
```

Edit the `USER CONFIG` block in `fantom.py` to set the STL path, taper parameters, target voxel count, and end label values. You can also add a 1-voxel padding on faces that are not expected outlets via `PAD_EMPTY_FACES`, `PAD_THICKNESS`, `EXPECTED_OUT_FACES`, and `PAD_VALUE` (default pads with label 2). Mayavi hides the padded layer when `HIDE_PAD_IN_MAYAVI=True`. The script prints STL bounds and voxel spacing after voxelization, exports the `geom_/dim_/val_` triplet, and can open a hollow Mayavi view with per-label colors and a simple legend. The taper step uses Trimesh `update_faces(unique_faces())` and `update_faces(nondegenerate_faces())` for cleanup, so ensure your Trimesh build exposes those helpers.
## Geometry API (Python)

- `Geometry(name=None, resolution=1, split=None, num_processes=1, output_dir="output", expected_in_outs=None, stl_path=None, leading_multiple=128, **kwargs)`
  - `.geo` route: set `name="<template>"` and pass template placeholders in `kwargs` (e.g., `lower_angle`, `upper_flare`, …). If the template expects `DEFINE_H` and you don’t pass `h`, it is inferred from `resolution`.
  - STL route: set `stl_path="path/to/closed.stl"` and leave `name=None`.
  - `resolution`: longest axis ~ `leading_multiple * resolution` voxels (default 128; adjusted to match Trimesh’s voxel grid behavior).
  - `split`: optional integer to segment along the leading axis and stitch a globally filled lattice; `num_processes` controls parallel workers.
  - `leading_multiple`: target multiple for the leading axis voxel count and LBM padding. Use the default 128 for existing behavior or set, e.g., `100` to retarget both pitch and LBM divisibility.
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

Optional domain wall tags can be applied on the fluid layer if you pass `expected_in_outs` containing any of `{N, E, S, W, B, F}`. The mapping is applied on the fluid boundary planes (first/last indices where fluid exists):
- N (z max among fluid) → 11, E (x max among fluid) → 12, S (z min among fluid) → 13, W (x min among fluid) → 14, B (y max among fluid) → 15, F (y min among fluid) → 16

Face-tagging details:
- Tags use the first/last indices along each axis where fluid exists; this aligns with inlets/outlets of the fluid domain.
- In the export pipeline, near-wall bands (3/4/5) are assigned first, then face tags (11..16) are reapplied so they override near-wall labels on the tagged planes.

When exporting text, missing domain faces are padded by a single wall layer to ensure presence and consistency for downstream codes.

## Splitting & Parallelization

- `split=<int>` divides the mesh along the leading axis, voxelizes each segment, and stitches the results; `num_processes` controls the number of workers used for segment voxelization.
- Segments overlap at their boundaries to avoid gaps; after stitching we clamp indices to the global lattice, run a one-voxel binary dilation to seal seams, and then apply a single global fill so the result matches the no‑split grid exactly.
- Shapes are normalized to the expected target derived from bounds and pitch, ensuring split and no‑split runs are identical.
- For large models, start with small `resolution` (1–2) before scaling up; increase `split`/`num_processes` as needed to stay within memory limits.

Pitch and shape: The voxel pitch is chosen so the longest axis approximates `leading_multiple * resolution` cells (default 128). Trimesh’s VoxelGrid yields `N ≈ floor(extent / pitch) + 1` per axis; we set pitch from the longest extent to achieve the target and then normalize results to match the expected dims without trailing empty planes.

LBM shape normalization: For LBM grid completion, the leading dimension must be a multiple of `leading_multiple` (default 128); the remaining dimensions are rounded up to the nearest multiple of 32.

Watertightness note: Splitting assumes robust slicing of a closed surface. For STL inputs, ensure watertightness for best results; light repairs (normals/winding, small hole filling) are attempted but not guaranteed.

Performance: Avoids per-voxel Python loops and relies on NumPy/Scipy vectorization; segment voxelization can use a `ProcessPoolExecutor` when `num_processes > 1`.

## Troubleshooting

- Module import: If running from a repo checkout without installing, use `pip install -e .` or add the repo root to `PYTHONPATH`.
- Mayavi/VTK on headless: Remove `mayavi`, `vtk`, `pyqt` from the environment if not visualizing; or configure an X server.
- Gmsh issues: Ensure `gmsh` from conda-forge is installed; templates rely on OpenCASCADE features. If Gmsh crashes, verify `.geo` placeholders match the parameters you provide.

## Testing Guidance

- Quick shape sanity: with `split=None`, the longest axis should equal `≈ leading_multiple * resolution` (128 by default).
- Split equivalence: split and no‑split runs should match dimensions after normalization.
- Use small `resolution` (1–2) for fast checks; verify both routes (.geo and STL) produce the same dims.
- Validate label distributions: expect values in `{0,1,2,3,4,5}` and optionally `{11..16}` when domain tags are requested.
- Confirm `dim_*` matches the in-memory array shape and that `geom_*` encodes `x y z value` in that shape.

## Documentation

Detailed module guides with inline examples live under `documentation/`:
- `documentation/geometry.md` — high-level API and workflows
- `documentation/mesher.md` — Gmsh templating and STL generation
- `documentation/voxels.md` — voxelization, splitting, labeling, and export

If you change code or behavior, update both this README and the relevant module guides.

## Contributing

- Keep both pipelines equivalent (.geo→STL→voxels and STL→voxels) in output and label semantics.
- Preserve existing APIs; add new ones without breaking current callers.
- Be surgical in changes; match the existing concise Python style.
- Document changes: update README, examples if affected, and `documentation/*.md` accordingly.
