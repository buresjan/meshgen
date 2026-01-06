# Module Guide: meshgen.voxels

The `voxels` module performs voxelization (with or without splitting), filling, stitching, shape normalization, and conversion to labeled volumes suitable for downstream simulation.

## Overview

- Pitch selection: pitch is chosen so the longest axis yields approximately `leading_multiple * resolution` cells (default 128), aligning with Trimesh VoxelGrid behavior (`N ≈ ceil(extent/pitch) + 1`).
- Splitting: the mesh is partitioned along the leading axis, segments are voxelized independently (optionally in parallel), stitched into a single grid, and then sealed with a one-voxel binary dilation before the global fill. This closes seams introduced by segmentation so the final lattice is identical to the no‑split path.
- Labeling: boolean occupancy is transformed into solver labels: 0 outside, 1 fluid, 2 wall, and near‑wall bands 3/4/5; optionally 11..16 for domain face tags.

## Key Functions

### `voxelize_mesh`

```
voxelize_mesh(name: str, res: int = 1,
              split: int | None = None,
              num_processes: int = 1,
              leading_multiple: int = 128,
              **kwargs) -> np.ndarray
```

Pipeline: `.geo → STL → voxels`. Generates an STL via `mesher.gmsh_surface(name, **kwargs)` then voxelizes it. Returns a boolean occupancy array.
Requires the Gmsh Python module; a descriptive `RuntimeError` is raised if it is unavailable.

Example:

```python
from meshgen.voxels import voxelize_mesh

occ = voxelize_mesh(
    name="junction_2d",
    res=2,
    split=None,
    lower_angle=10,
    upper_angle=-5,
    upper_flare=0.001,
    lower_flare=0.001,
)
print(occ.shape, occ.dtype)
```

### `voxelize_stl`

```
voxelize_stl(path: str, res: int = 1,
             split: int | None = None,
             num_processes: int = 1,
             leading_multiple: int = 128) -> np.ndarray
```

Pipeline: `STL → voxels`. Accepts absolute or relative STL path. Ensures a closed surface if possible (light repairs on normals/winding/holes); then voxelizes. When `split` is provided, faces are partitioned along the leading axis, segments are voxelized (optionally in parallel), stitched, and filled once globally.
This route does not depend on Gmsh, so it works even in environments without the templating tool.

Example:

```python
from meshgen.voxels import voxelize_stl

occ = voxelize_stl("examples/glenn_capped.stl", res=2)
```

> The shipped `examples/glenn_capped.stl` is already capped on N/E/W and cleaned via `scripts/repair_glenn_stl.py`, so the STL → voxels path no longer produces trailing single-voxel fragments.

### Splitting Helpers

- `split_mesh(mesh, voxel_size, n_segments)` — partitions faces along the leading axis with overlap.
- `voxelize_with_splitting(mesh, voxel_size, split, num_processes, target_bounds)` — segment, voxelize (in parallel if desired), clamp indices to the global lattice, dilate once to seal seams, fill, and normalize shape.

### Filling and Completion

- `fill_mesh_inside_surface(mesh)` — fill internal voids via `scipy.ndimage.binary_fill_holes`.
- `complete_mesh(original_mesh, num_type='bool', expected_in_outs=None, leading_multiple=128)` — enforce LBM-friendly shape; embeds the mesh into a grid where the leading dimension is a multiple of `leading_multiple` (default 128) and the others are rounded up to a multiple of 32. If `num_type='int'`, returns integer labels.

### Labeling and Export Prep

- `label_elements(mesh, expected_in_outs=None, num_type='bool')` — convert occupancy to labels:
  - 0 empty/outside; 1 fluid; 2 wall; optionally 11..16 for `{N,E,S,W,B,F}` on the fluid layer.
- `assign_near_walls`, `assign_near_near_walls`, `assign_near_near_near_walls` — set labels 3, 4, and 5 on fluid voxels adjacent to 2, 3, and 4, respectively.
- `prepare_voxel_mesh_txt(mesh, expected_in_outs=None, num_type='int')` — produce the labeled volume for export; pads missing domain faces with a one‑voxel wall layer; adds near‑wall bands; reapplies face tags on fluid boundary planes last so they override near‑wall. Accepts any iterable of faces or a mapping of `face -> enabled` flags.

### Custom End Tags (fantom.py)

The standalone `fantom.py` script applies additional inlet/outlet tags after running `prepare_voxel_mesh_txt`. It identifies connected components on the `y_min` and `y_max` fluid boundary planes and overwrites those plane voxels with distinct labels. Defaults:

- `y_max` opening → 21
- three `y_min` openings → 22, 23, 24 (sorted by (x, z) centroid order)

These labels are script-specific and do not alter the core module semantics (0/1/2/3/4/5 and optional 11..16 remain unchanged elsewhere).

The script can also pad non-expected faces with a 1-voxel layer using `PAD_EMPTY_FACES`, `PAD_THICKNESS`, `EXPECTED_OUT_FACES`, and `PAD_VALUE` (default pads with label 2). This expands the exported array only on faces that are not part of the expected outlets; Mayavi hides the padded layer when `HIDE_PAD_IN_MAYAVI=True`.
It prints the STL bounds and voxel spacing after voxelization for quick scale checks.

## End-to-End Examples

### 1) Templated route

```python
from meshgen.voxels import voxelize_mesh

occ = voxelize_mesh(
    name="junction_2d",
    res=2,
    split=4,
    lower_angle=10,
    upper_angle=-5,
    upper_flare=0.001,
    lower_flare=0.001,
)
```

### 2) STL route with splitting

```python
from meshgen.voxels import voxelize_stl

occ = voxelize_stl("examples/glenn_capped.stl", res=2, split=6, num_processes=6)
```

### 3) Prepare labeled volume for export

```python
import numpy as np
from meshgen.voxels import prepare_voxel_mesh_txt

labels = prepare_voxel_mesh_txt(occ, expected_in_outs={"W","E","N"}, num_type='int')
print(np.unique(labels))  # expect 0,1,2,3,4,5 and optionally 11..16
```

## Notes

- All heavy operations are vectorized; avoid writing per-voxel Python loops around these APIs.
- Ensure inputs are closed surfaces for reliable filling; light repairs are attempted but not guaranteed.
- Install via the `pyproject.toml` metadata: `pip install -e .` (pip >= 21) for core deps, `pip install -e .[vis]` for Mayavi visualization (not available on Python 3.13; use conda or Python 3.10/3.11).

### Face Tagging Semantics

- Face tags are applied after embedding and near‑wall labeling on the fluid boundary planes: for each axis, the first and last indices where fluid exists define the min/max face.
- Mapping: `W=x_min`, `E=x_max`, `F=y_min`, `B=y_max`, `N=z_max`, `S=z_min`.
- Tags (11..16) are applied only to fluid voxels on those planes and override near‑wall bands (3/4/5) there.
