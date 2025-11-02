# Module Guide: meshgen.utilities

Small helpers for visualization, surface extraction, and text export of volumes.

## `extract_surface`

```
extract_surface(arr: np.ndarray) -> np.ndarray
```

Returns a boolean mask of surface voxels from a boolean occupancy array. A surface voxel is a filled voxel adjacent to at least one empty neighbor. Padding is used temporarily to handle boundaries.

Example:

```python
import numpy as np
from meshgen.utilities import extract_surface

occ = np.zeros((8,8,8), dtype=bool)
occ[2:6,2:6,2:6] = True
surf = extract_surface(occ)
print(surf.sum(), occ.sum())
```

## `vis`

```
vis(mesh: np.ndarray) -> None
```

Visualizes the surface voxels of a 3D occupancy or labeled array using Mayavi. Only the surface is rendered as cube glyphs for performance. Requires `mayavi`.

Example:

```python
from meshgen.utilities import vis

# Assume `occ` is a 3D boolean occupancy array
vis(occ)
```



## `array_to_textfile`

```
array_to_textfile(array: np.ndarray, filename: str) -> None
```

Writes a 3D NumPy array to text, one row per voxel: `x y z value`. Implemented with vectorized NumPy to avoid Python loops. Used by `Geometry.save_voxel_mesh_to_text` and `Geometry.save_lbm_mesh_to_text`.

Example:

```python
import numpy as np
from meshgen.utilities import array_to_textfile

labels = np.zeros((16,16,16), dtype=int)
labels[4:12, 4:12, 4:12] = 1
array_to_textfile(labels, "output/geom_example.txt")
```

Notes:
- Pair the geometry file with a `dim_*.txt` file containing `Nx Ny Nz` and an empty `val_*.txt` for the expected triplet format.
- The package metadata lives in `pyproject.toml`; install with `pip install -e .` (pip ≥ 21) for editable development.
