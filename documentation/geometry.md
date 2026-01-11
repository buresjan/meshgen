# Module Guide: meshgen.geometry

The `Geometry` class is the high-level API for running either pipeline:

- `.geo → STL → voxels` for parametric shapes defined by templates.
- `STL → voxels` for prebuilt, closed (watertight) surfaces.

It orchestrates voxelization, labeling, exporting to the text triplet, and optional visualization.

## Key Concepts

- Resolution & stride: the voxel pitch is chosen so the longest axis has approximately `leading_multiple * resolution` cells (default `leading_multiple=128`).
- Template alignment: templated geometries receive `DEFINE_RESOLUTION`, allowing them to add sub-voxel pads where needed (e.g., `junction_2d` extends the axial outlet slightly) so `.geo` and STL routes agree on the final filled slice.
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
    expected_in_outs: Collection[str] | Mapping[str, bool] | None = None,
    stl_path: str | None = None,
    leading_multiple: int = 128,
    **kwargs
)
```

- `.geo` route: set `name` to your template base name (e.g., `"junction_2d"`) and pass template placeholders via `kwargs` (e.g., `lower_angle=10`). If the template uses `DEFINE_H` and you don’t specify `h`, it is inferred from `resolution`.
- STL route: set `stl_path` to an absolute or relative STL path and leave `name=None`.
- `leading_multiple`: target multiple for the leading-axis voxel count and the LBM padding requirement (default 128). Use the same value for both STL and `.geo` routes to keep outputs aligned.
- `expected_in_outs`: supply any iterable of domain faces (e.g., `{"W","E","N"}`) or a mapping of face → truthy/falsey toggle. Enabled faces receive the 11..16 labels; others are padded with a wall layer during export.

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
    expected_in_outs={"W", "E", "N"},  # iterable or dict of domain faces to label
)

geom.generate_voxel_mesh()
geom.save_voxel_mesh_to_text("glenn_capped.txt")
```

To run this flow end-to-end on the repository’s `glenn_extended.stl` (north inlet with east/west outlets and `expected_in_outs={"N","E","W"}`), execute:

```bash
conda activate meshgen
python examples/glenn_extended_visualize.py
```

The STL now ships capped on the N/E/W ends and cleaned with `scripts/repair_glenn_stl.py`, so voxelization no longer leaves an isolated trailing fluid voxel. Re-run the helper if you ever need to regenerate the sanitized STL from the `examples/glenn_capped.stl` source.

The visualization script voxelizes the STL at resolution 4, writes the `geom_*/dim_*/val_*` triplet to `examples/output/`, and visualizes the result via Mayavi for a quick sanity-check.

### 3) Split parameter (API compatibility)

```python
geom = Geometry(
    stl_path="big_model.stl",
    resolution=2,
    split=6,            # accepted for API compatibility
    num_processes=6,    # accepted for API compatibility
)
geom.generate_voxel_mesh()  # output matches the no‑split path exactly
```

Notes:
- Both routes yield the same text triplet and label semantics.
- The current implementation ensures equivalence by computing a single global voxelization; `split` does not change the output or parallelize the voxelization itself.
- For best STL results, ensure watertightness. Minor repairs are attempted automatically.
- Installation uses `pyproject.toml`; use `pip install -e .` (pip ≥ 21) to get an editable checkout.
- If you plan to use the `vascular_encoding_framework` submodule, install it in the same environment (`pip install -e vascular_encoding_framework` or use the root `environment.yml`).

### 4) Minimal STL voxelization with Mayavi only

If you only need a quick voxelization + viewer (no text export), run the lean script that targets the repository’s root-level `master_combined_capped.stl` at resolution 4. It prints the STL bounds and physical size (in STL units) before voxelization so you can sanity-check scale:

```bash
conda activate meshgen
python examples/master_combined_visualize.py
```
