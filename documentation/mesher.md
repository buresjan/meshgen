# Module Guide: meshgen.mesher

The `mesher` module handles turning `.geo` templates into STL surfaces with Gmsh, and generating a boxed STL wrapper used to robustify splitting for the STL route. Gmsh is imported lazily—attempting to call these helpers without `gmsh` installed raises a clear `RuntimeError`, while the STL voxelization pipeline can operate without it.

## Template Editing: `modify_geo_file`

```
modify_geo_file(input_file_path: str, output_file_path: str, **kwargs) -> None
```

Performs placeholder replacement in `.geo` templates. Placeholders follow the form `DEFINE_<NAME>` and are replaced by keyword arguments where `NAME` is given in lowercase in `kwargs`.

Example:

```python
from meshgen.mesher import modify_geo_file

modify_geo_file(
    "meshgen/geo_templates/junction_2d_template.geo",
    "/tmp/junction_2d_filled.geo",
    lower_angle=10,
    upper_angle=-5,
    upper_flare=0.001,
    lower_flare=0.001,
    h=0.001,  # optional if template uses DEFINE_H; inferred elsewhere
)
```

## Surface Generation: `gmsh_surface`

```
gmsh_surface(name_geo: str, dependent: bool = False, **kwargs) -> str
```

Loads `meshgen/geo_templates/{name_geo}_template.geo`, fills placeholders, and runs Gmsh to generate a 2D surface mesh which is exported as an STL in a temporary working directory. Returns the path to the generated STL.

Implementation notes:
- Gmsh sessions are wrapped with `gmsh.initialize()`/`gmsh.finalize()` in a `try/finally` block.
- If the template contains `DEFINE_H` and you do not pass `h`, a heuristic based on `resolution` is used: `h ≈ max(1e-4, 1e-3 / resolution)`.
- The keyword `resolution` is forwarded unchanged, so templates can reference `DEFINE_RESOLUTION` for resolution-aware tweaks (e.g., adding a sub-voxel axial pad in `junction_2d`).

Example:

```python
from meshgen.mesher import gmsh_surface

stl_path = gmsh_surface(
    "junction_2d",
    lower_angle=10,
    upper_angle=-5,
    upper_flare=0.001,
    lower_flare=0.001,
    resolution=2,
)
print("Generated:", stl_path)
```

## Boxed STL Wrapper: `box_stl`

```
box_stl(stl_file: str, x: float, y: float, z: float,
        dx: float, dy: float, dz: float,
        voxel_size: float) -> str
```

Generates a temporary boxed STL that embeds the provided STL inside an axis-aligned box using a Geo template and Gmsh. This is used by the STL voxelization route when splitting is enabled to make plane slicing robust and consistent.

Example:

```python
from meshgen.mesher import box_stl

boxed = box_stl(
    stl_file="examples/glenn_capped.stl",
    x=0.0, y=0.0, z=0.0,
    dx=0.1, dy=0.1, dz=0.1,
    voxel_size=0.001,
)
print("Boxed STL:", boxed)
```

## Tips

- Keep templates parametric with `DEFINE_*` placeholders; avoid hardcoding values.
- Use absolute or relative STL paths in `box_stl`; bare names resolve to `meshgen/stl_models/<name>.stl`.
- Callers must ensure Gmsh is installed before using `gmsh_surface` or `box_stl`; otherwise a descriptive `RuntimeError` is raised.
- Template and STL assets are bundled via the package-data configuration in `pyproject.toml`; editable installs require `pip >= 21` (PEP 517).
- Install via `pip install -e .` for core deps, and add `pip install -e .[vis]` for Mayavi visualization (not available on Python 3.13; use conda or Python 3.10/3.11).
