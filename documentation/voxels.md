# Module Guide: meshgen.voxels

The `voxels` module performs voxelization (with or without splitting), filling, stitching, shape normalization, and conversion to labeled volumes suitable for downstream simulation.

## Overview

- Pitch selection: pitch is chosen so the longest axis yields approximately `128 * resolution` cells, aligning with Trimesh VoxelGrid behavior (`N ≈ ceil(extent/pitch) + 1`).
- Splitting: the mesh is partitioned along the leading axis, segments are voxelized independently (optionally in parallel), stitched into a single grid, and then sealed with a one-voxel binary dilation before the global fill. This closes seams introduced by segmentation so the final lattice is identical to the no‑split path.
- Labeling: boolean occupancy is transformed into solver labels: 0 outside, 1 fluid, 2 wall, and near‑wall bands 3/4/5; optionally 11..16 for domain face tags.

## Key Functions

### `voxelize_mesh`

```
voxelize_mesh(name: str, res: int = 1,
              split: int | None = None,
              num_processes: int = 1,
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
             num_processes: int = 1) -> np.ndarray
```

Pipeline: `STL → voxels`. Accepts absolute or relative STL path. Ensures a closed surface if possible (light repairs on normals/winding/holes); then voxelizes. When `split` is provided, faces are partitioned along the leading axis, segments are voxelized (optionally in parallel), stitched, and filled once globally.
This route does not depend on Gmsh, so it works even in environments without the templating tool.

Example:

```python
from meshgen.voxels import voxelize_stl

occ = voxelize_stl("examples/glenn_capped.stl", res=2)
```

### Splitting Helpers

- `split_mesh(mesh, voxel_size, n_segments)` — partitions faces along the leading axis with overlap.
- `voxelize_with_splitting(mesh, voxel_size, split, num_processes, target_bounds)` — segment, voxelize (in parallel if desired), clamp indices to the global lattice, dilate once to seal seams, fill, and normalize shape.

### Filling and Completion

- `fill_mesh_inside_surface(mesh)` — fill internal voids via `scipy.ndimage.binary_fill_holes`.
- `complete_mesh(original_mesh, num_type='bool', expected_in_outs=None)` — enforce LBM-friendly shape; embeds the mesh into a grid where the leading dimension is a multiple of 128 and the others are rounded up to a multiple of 32. If `num_type='int'`, returns integer labels.

### Labeling and Export Prep

- `label_elements(mesh, expected_in_outs=None, num_type='bool')` — convert occupancy to labels:
  - 0 empty/outside; 1 fluid; 2 wall; optionally 11..16 for `{N,E,S,W,B,F}` on the fluid layer.
- `assign_near_walls`, `assign_near_near_walls`, `assign_near_near_near_walls` — set labels 3, 4, and 5 on fluid voxels adjacent to 2, 3, and 4, respectively.
- `prepare_voxel_mesh_txt(mesh, expected_in_outs=None, num_type='int')` — produce the labeled volume for export; pads missing domain faces with a one-voxel wall layer; adds near-wall bands. Accepts any iterable of faces or a mapping of `face -> enabled` flags.

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
