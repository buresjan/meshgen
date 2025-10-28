Agent Guidance for meshgen

Scope: Root of this repository (applies to all files).

Goals
- Keep both pipelines equivalent:
  - .geo → STL → voxels (Gmsh templated geometries)
  - STL → voxels (closed STL surfaces)
- Both routes must produce the same simulation-ready output format and label semantics.

Code Style and Practices
- Be surgical: modify only what’s needed; do not rename or move files unless required by the task.
- Match the existing Python style (concise, readable, small helpers over big refactors).
- Prefer pure functions and clear parameters; avoid hidden globals or side effects.
- Keep meshes as NumPy arrays with dtype bool for occupancy and dtype int for labeled volumes.
- When adding new APIs, preserve existing ones for backward compatibility.

Pipelines
- .geo route: Use `gmsh_surface` in `meshgen/mesher.py` and voxelize the emitted STL.
- STL route: Accept absolute or relative STL path; ensure the surface is closed (watertight) before voxelization.
- For splitting, segment along the leading axis and preserve full 3D blocks for each segment. Stitch, then fill once globally.
- Pitch and shape: longest axis ≈ 128*resolution; use ceil/floor logic consistent with Trimesh VoxelGrid behavior and normalize stitched results to expected dims.

Labels and Output
- Label semantics expected by downstream simulation:
  - 0 empty/outside, 1 fluid/interior, 2 wall/solid, 3/4/5 near-wall bands.
  - Optional domain wall tags: N/E/S/W/B/F → 11..16.
- Text export uses three files (triplet): `geom_*.txt`, `dim_*.txt`, `val_*.txt`.
- Do not change file naming or ordering without explicit instruction.

Gmsh and Templates
- Always wrap `gmsh.initialize()`/`gmsh.finalize()` in try/finally.
- Modify `.geo` templates via placeholder replacement only; do not hardcode values in the template files.
- Keep template placeholders all-uppercase with `DEFINE_` prefix.

Performance and Parallelism
- `split` controls segment count along the leading axis; `num_processes` controls parallel workers.
- Avoid per-voxel Python loops; rely on NumPy/Scipy vectorization.

Testing Guidance
- Quick shape sanity: longest axis equals 128*resolution for no-split; split path should match no-split.
- Use small resolution (1–2) for fast checks; verify both routes (.geo and STL) produce the same dims.
- Validate label distributions (0/1/2/3/4/5 and 11..16 when requested) and that `dim_*` matches array shape.

Do Not
- Do not commit secrets, large binaries, or generated artifacts.
- Do not change external interfaces without updating README and minimal examples.

Documentation
- When adding features or changing behaviors, update README with:
  - API parameters and defaults
  - Output format and label semantics
  - Pipelines (how to choose between .geo and STL)

Change Management
- Any code or behavior change must be properly documented.
- Update both the root README and the per-module guides under `documentation/` in the same change.
- Examples should be refreshed to reflect new parameters or outputs when applicable.
