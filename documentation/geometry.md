# Module Guide: meshgen.geometry

The `Geometry` class is the high-level API for running either pipeline:

- `.geo → STL → voxels` for parametric shapes defined by templates.
- `STL → voxels` for prebuilt, closed (watertight) surfaces.

It orchestrates voxelization, labeling, exporting to the text triplet, and optional visualization.

## Key Concepts

- Resolution: the voxel pitch is chosen so the longest axis has approximately `128 * resolution` cells.
- Splitting: large models can be split along the leading axis and processed with multiple workers; results are stitched and filled once globally to match the no‑split behavior.
- Labels: occupancy is converted to solver-ready labels (0 outside, 1 fluid, 2 wall, 3/4/5 near‑wall bands). Optional domain face tags `{N,E,S,W,B,F}` map to 11..16 on the fluid layer.

## API

```
Geometry(
    name: str | None = None,
    resolution: int = 1,
    split: int | None = None,
    num_processes: int = 1,
    output_dir: str = "output",
    expected_in_outs: set[str] | None = None,
    stl_path: str | None = None,
    **kwargs
)
```

- `.geo` route: set `name` to your template base name (e.g., `"junction_2d"`) and pass template placeholders via `kwargs` (e.g., `lower_angle=10`). If the template uses `DEFINE_H` and you don’t specify `h`, it is inferred from `resolution`.
- STL route: set `stl_path` to an absolute or relative STL path and leave `name=None`.

Methods:
- `generate_voxel_mesh()` → compute voxel occupancy (dtype `bool`).
- `get_voxel_mesh()` → return the occupancy array.
- `save_voxel_mesh(path="voxel_mesh.npy")` → save `.npy` occupancy.
- `save_voxel_mesh_to_text(path="voxel_mesh.txt")` → write `geom_/dim_/val_` text triplet.
- `generate_lbm_mesh()` → complete to solver grid and compute labels.
- `save_lbm_mesh_to_text(path="lbm_mesh.txt")` → write labeled volume in triplet format.
- `visualize()` → render the voxel surface with Mayavi.

## Examples

### 1) .geo route (templated geometry)

```python
from meshgen.geometry import Geometry

geom = Geometry(
    name="junction_2d",
    resolution=2,
    split=None,
    num_processes=1,
    output_dir="output",
    lower_angle=10,
    upper_angle=-5,
    upper_flare=0.001,
    lower_flare=0.001,
    # h is optional; inferred when template uses DEFINE_H
)

geom.generate_voxel_mesh()
print("voxels:", geom.get_voxel_mesh().shape)

# Text triplet for simulation
geom.save_voxel_mesh_to_text("junction_2d.txt")

# Optional labeled mesh (LBM-complete)
geom.generate_lbm_mesh()
geom.save_lbm_mesh_to_text("junction_2d_lbm.txt")
```

### 2) STL route (direct surface voxelization)

```python
from meshgen.geometry import Geometry

geom = Geometry(
    stl_path="examples/glenn_capped.stl",
    resolution=2,
    split=None,            # or an integer for splitting
    num_processes=1,
    output_dir="output",
    expected_in_outs={"W", "E", "N"},  # optional domain face tags
)

geom.generate_voxel_mesh()
geom.save_voxel_mesh_to_text("glenn_capped.txt")
```

### 3) Parallel splitting for large models

```python
geom = Geometry(
    stl_path="big_model.stl",
    resolution=2,
    split=6,            # segments along the leading axis
    num_processes=6,    # parallel workers
)
geom.generate_voxel_mesh()
```

Notes:
- Both routes yield the same text triplet and label semantics.
- For best STL results, ensure watertightness. Minor repairs are attempted automatically.

