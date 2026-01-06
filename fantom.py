#!/usr/bin/env python3
"""
fantom.py

Pipeline:
  1) Load an STL with 4 ends aligned to +/-Y (as expected by tcpc_taper_ends.py)
  2) Optionally taper/extend selected ends and cap all ends
  3) Voxelize using a target voxel count on the longest axis (not "resolution")
  4) Label voxels with standard semantics and distinct inlet/outlet tags
  5) Export geom_/dim_/val_ triplet and optionally visualize in Mayavi
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import trimesh
from scipy import ndimage

import tcpc_taper_ends as taper
from meshgen.utilities import array_to_textfile, extract_surface
from meshgen.voxels import prepare_voxel_mesh_txt, voxelize_with_splitting


# =============================================================================
# USER CONFIG (edit this section)
# =============================================================================

# Input / output
INPUT_STL: str = "fantom.stl"
OUTPUT_STL: str = "fantom_tapered.stl"

# Output text triplet (geom_/dim_/val_) in OUTPUT_DIR with this suffix
OUTPUT_DIR: str = "output"
OUTPUT_TEXT_NAME: str = "fantom.txt"

# -----------------------------------------------------------------------------#
# End manipulation (same semantics as tcpc_taper_ends.py)
# -----------------------------------------------------------------------------#

UNCAP_ALL_ENDS: bool = False
WORKING_ENDS: List[int] = [0, 1, 2]
TAPER_PARAMS: Dict[int, Tuple[float, float]] = {
    0: (0.5, 35.0),
    1: (1.0, 35.0),
    2: (0.4, 35.0),
}

TAPER_PROFILE_DEFAULT: str = "smooth"  # "linear" or "smooth"
TAPER_PROFILE_PER_END: Dict[int, str] = {}
SMOOTH_SEGMENTS: int = 24

EXPECTED_ENDS: int = 4
AXIS: np.ndarray = np.array([0.0, 1.0, 0.0])
NORMAL_THRESHOLD: float = 0.95

LABEL_SORT_AXES: Tuple[str, str, str] = ("y", "x", "z")
LABEL_SORT_DIRECTIONS: Dict[str, float] = {"x": 1.0, "y": 1.0, "z": 1.0}

# -----------------------------------------------------------------------------#
# Voxelization target (longest axis voxel count)
# -----------------------------------------------------------------------------#

LONGEST_AXIS_VOXELS: int = 500
SPLIT: Optional[int] = None
NUM_PROCESSES: int = 1

# -----------------------------------------------------------------------------#
# End labels (distinct tags for 1 max-Y and 3 min-Y openings)
# -----------------------------------------------------------------------------#

END_LABEL_Y_MAX: int = 21
END_LABELS_Y_MIN: Sequence[int] = (22, 23, 24)

# Sort the three min-Y openings by (x, z) centroid order
END_SORT_AXES: Tuple[str, str] = ("x", "z")
END_SORT_DIRECTIONS: Dict[str, float] = {"x": 1.0, "z": 1.0}

# -----------------------------------------------------------------------------#
# Optional padding on non-expected faces
# -----------------------------------------------------------------------------#

PAD_EMPTY_FACES: bool = True
PAD_THICKNESS: int = 1
EXPECTED_OUT_FACES: Sequence[str] = ("y_min", "y_max")
PAD_VALUE: int = 2
HIDE_PAD_IN_MAYAVI: bool = True

# -----------------------------------------------------------------------------#
# Visualization (Mayavi)
# -----------------------------------------------------------------------------#

SHOW_MAYAVI: bool = True
HOLLOW_VIEW: bool = True
SHOW_LABELS: Optional[Iterable[int]] = None  # None => auto-detect present labels


# =============================================================================
# HELPERS
# =============================================================================

def _load_mesh(path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(path)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected a Trimesh, got {type(mesh)} for {path}")
    return mesh


def _voxel_size_from_target(mesh: trimesh.Trimesh, target_voxels: int) -> float:
    if target_voxels is None or int(target_voxels) < 2:
        raise ValueError("LONGEST_AXIS_VOXELS must be an integer >= 2.")
    target_voxels = int(target_voxels)
    extents = mesh.bounds[1] - mesh.bounds[0]
    longest = float(np.max(extents))
    return longest / max(1, target_voxels - 1)


def _voxelize_target(
    mesh: trimesh.Trimesh,
    target_voxels: int,
    split: Optional[int],
) -> Tuple[np.ndarray, float]:
    voxel_size = _voxel_size_from_target(mesh, target_voxels)
    bounds = mesh.bounds
    occ = voxelize_with_splitting(
        mesh,
        voxel_size,
        split,
        num_processes=NUM_PROCESSES,
        target_bounds=bounds,
    )
    return occ, voxel_size


def _component_centroids(mask2d: np.ndarray) -> List[Dict[str, np.ndarray]]:
    structure = np.ones((3, 3), dtype=int)
    labeled, num = ndimage.label(mask2d, structure=structure)
    components: List[Dict[str, np.ndarray]] = []
    for comp_id in range(1, num + 1):
        coords = np.argwhere(labeled == comp_id)
        if coords.size == 0:
            continue
        centroid = coords.mean(axis=0)
        components.append(
            {
                "id": comp_id,
                "coords": coords,
                "centroid": centroid,
                "size": coords.shape[0],
            }
        )
    return components


def _sort_components(
    components: List[Dict[str, np.ndarray]],
    axes: Tuple[str, ...],
    directions: Dict[str, float],
) -> List[Dict[str, np.ndarray]]:
    axis_map = {"x": 0, "z": 1}
    for axis in axes:
        if axis not in axis_map:
            raise ValueError(f"Unsupported axis '{axis}' for 2D plane sort. Use 'x'/'z'.")

    def key(comp: Dict[str, np.ndarray]) -> Tuple[float, ...]:
        coords = comp["centroid"]
        return tuple(float(directions.get(a, 1.0)) * float(coords[axis_map[a]]) for a in axes)

    return sorted(components, key=key)


def _label_plane_components(
    labeled: np.ndarray,
    plane_mask: np.ndarray,
    plane_idx: int,
    target_labels: Sequence[int],
    sort_axes: Tuple[str, ...],
    sort_dirs: Dict[str, float],
    plane_name: str,
) -> None:
    components = _component_centroids(plane_mask)
    if len(components) < len(target_labels):
        raise RuntimeError(
            f"Expected {len(target_labels)} components on {plane_name}, found {len(components)}."
        )
    if len(components) > len(target_labels):
        components = sorted(components, key=lambda c: int(c["size"]), reverse=True)[: len(target_labels)]

    components = _sort_components(components, axes=sort_axes, directions=sort_dirs)

    for comp, label in zip(components, target_labels):
        coords = comp["coords"]
        xs = coords[:, 0]
        zs = coords[:, 1]
        labeled[xs, plane_idx, zs] = int(label)


def label_end_planes(
    labeled: np.ndarray,
    label_y_max: int,
    labels_y_min: Sequence[int],
    sort_axes: Tuple[str, str],
    sort_dirs: Dict[str, float],
) -> np.ndarray:
    fluid_like = np.isin(labeled, [1, 3, 4, 5])
    if not np.any(fluid_like):
        raise RuntimeError("No fluid-like voxels found for end labeling.")

    y_any = np.any(fluid_like, axis=(0, 2))
    y_ids = np.flatnonzero(y_any)
    if y_ids.size == 0:
        raise RuntimeError("No Y-planes contain fluid-like voxels.")

    y_min = int(y_ids[0])
    y_max = int(y_ids[-1])

    plane_max = fluid_like[:, y_max, :]
    plane_min = fluid_like[:, y_min, :]

    _label_plane_components(
        labeled,
        plane_max,
        y_max,
        [label_y_max],
        sort_axes=sort_axes,
        sort_dirs=sort_dirs,
        plane_name="y_max",
    )
    _label_plane_components(
        labeled,
        plane_min,
        y_min,
        list(labels_y_min),
        sort_axes=sort_axes,
        sort_dirs=sort_dirs,
        plane_name="y_min",
    )

    return labeled


def pad_empty_faces(
    mesh: np.ndarray,
    keep_faces: Iterable[str],
    thickness: int = 1,
    value: int = 0,
    return_mask: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    allowed = {"x_min", "x_max", "y_min", "y_max", "z_min", "z_max"}
    keep = set(keep_faces)
    unknown = keep - allowed
    if unknown:
        raise ValueError(f"Unknown face names: {sorted(unknown)}")
    if int(thickness) <= 0:
        return mesh

    pad = [[0, 0], [0, 0], [0, 0]]
    if "x_min" not in keep:
        pad[0][0] = int(thickness)
    if "x_max" not in keep:
        pad[0][1] = int(thickness)
    if "y_min" not in keep:
        pad[1][0] = int(thickness)
    if "y_max" not in keep:
        pad[1][1] = int(thickness)
    if "z_min" not in keep:
        pad[2][0] = int(thickness)
    if "z_max" not in keep:
        pad[2][1] = int(thickness)

    padded = np.pad(mesh, pad, mode="constant", constant_values=int(value))
    if not return_mask:
        return padded

    mask = np.ones(padded.shape, dtype=bool)
    x0, y0, z0 = pad[0][0], pad[1][0], pad[2][0]
    x1 = x0 + mesh.shape[0]
    y1 = y0 + mesh.shape[1]
    z1 = z0 + mesh.shape[2]
    mask[x0:x1, y0:y1, z0:z1] = False
    return padded, mask


def export_triplet(mesh: np.ndarray, output_dir: str, name: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    geom_file = os.path.join(output_dir, f"geom_{name}")
    dim_file = os.path.join(output_dir, f"dim_{name}")
    val_file = os.path.join(output_dir, f"val_{name}")

    array_to_textfile(mesh, geom_file)
    with open(dim_file, "w") as f:
        sx, sy, sz = mesh.shape
        f.write(f"{sx} {sy} {sz}\n")
    open(val_file, "w").close()


def visualize_labels(
    labeled: np.ndarray,
    label_colors: Dict[int, Tuple[float, float, float]],
    label_names: Dict[int, str],
    show_labels: Optional[Iterable[int]] = None,
    hollow: bool = True,
) -> None:
    from mayavi import mlab

    present = np.unique(labeled)
    if show_labels is None:
        labels = [int(v) for v in present if int(v) != 0]
    else:
        labels = [int(v) for v in show_labels if int(v) in present and int(v) != 0]

    for label in labels:
        mask = labeled == label
        if hollow:
            mask = extract_surface(mask)
        if not np.any(mask):
            continue
        coords = np.argwhere(mask)
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        color = label_colors.get(label, (0.7, 0.7, 0.7))
        mlab.points3d(
            x,
            y,
            z,
            mode="cube",
            color=color,
            scale_mode="none",
            scale_factor=1,
        )

    y0 = 0.95
    dy = 0.04
    for i, label in enumerate(labels):
        name = label_names.get(label, f"label {label}")
        color = label_colors.get(label, (0.7, 0.7, 0.7))
        mlab.text(0.02, y0 - i * dy, f"{label}: {name}", color=color, width=0.3)

    mlab.view(azimuth=45, elevation=45)
    mlab.show()


def main() -> None:
    for i in WORKING_ENDS:
        if i not in (0, 1, 2, 3):
            raise ValueError(f"WORKING_ENDS contains invalid end index {i}; expected 0..3.")
    if len(END_LABELS_Y_MIN) != 3:
        raise ValueError("END_LABELS_Y_MIN must contain exactly three labels.")
    if int(PAD_THICKNESS) < 0:
        raise ValueError("PAD_THICKNESS must be >= 0.")

    mesh = _load_mesh(INPUT_STL)
    print(f"Loaded mesh: {INPUT_STL}")
    print(f"  vertices: {len(mesh.vertices)}")
    print(f"  faces:    {len(mesh.faces)}")
    print(f"  watertight: {mesh.is_watertight}")
    bounds = mesh.bounds
    extents = bounds[1] - bounds[0]
    print(
        "  bounds:   "
        f"x=({bounds[0][0]: .3f}, {bounds[1][0]: .3f}) "
        f"y=({bounds[0][1]: .3f}, {bounds[1][1]: .3f}) "
        f"z=({bounds[0][2]: .3f}, {bounds[1][2]: .3f})"
    )
    print(
        "  extents:  "
        f"dx={extents[0]: .3f}, dy={extents[1]: .3f}, dz={extents[2]: .3f}"
    )

    ends = taper._detect_ends(mesh, expected_ends=EXPECTED_ENDS, axis=AXIS, normal_threshold=NORMAL_THRESHOLD)
    ends = taper._assign_labels(ends, sort_axes=LABEL_SORT_AXES, directions=LABEL_SORT_DIRECTIONS)

    print("\nDetected ends (label -> centroid, outward normal, loop vertices):")
    for e in ends:
        c = e.centroid
        n = e.outward_normal / (np.linalg.norm(e.outward_normal) + taper.EPS)
        print(
            f"  End {e.label}: centroid=({c[0]: .3f}, {c[1]: .3f}, {c[2]: .3f}), "
            f"normal=({n[0]: .3f}, {n[1]: .3f}, {n[2]: .3f}), "
            f"rim_n={len(e.loop_vertices)}, has_cap={e.has_cap}"
        )

    print("\nWorking ends and parameters:")
    for idx in WORKING_ENDS:
        if idx not in TAPER_PARAMS:
            raise ValueError(f"End {idx} is in WORKING_ENDS but missing from TAPER_PARAMS.")
        s, L = TAPER_PARAMS[idx]
        prof = taper._normalize_profile_name(TAPER_PROFILE_PER_END.get(idx, TAPER_PROFILE_DEFAULT))
        segs = 1 if prof == "linear" else max(2, int(SMOOTH_SEGMENTS))
        print(f"  End {idx}: scale={s}, length={L}, profile={prof}, segments={segs}")

    out = taper._apply_taper_and_cap(
        mesh=mesh,
        ends=ends,
        working_ends=WORKING_ENDS,
        taper_params=TAPER_PARAMS,
        profile_default=TAPER_PROFILE_DEFAULT,
        profile_per_end=TAPER_PROFILE_PER_END,
        smooth_segments=SMOOTH_SEGMENTS,
        uncap_all=UNCAP_ALL_ENDS,
        cap_tri_tol=taper.CAP_TRI_TOL,
    )

    if hasattr(out, "is_watertight") and not out.is_watertight:
        try:
            trimesh.repair.fix_normals(out)
            trimesh.repair.fix_winding(out)
            trimesh.repair.fill_holes(out)
        except Exception:
            pass

    print("\nOutput mesh:")
    print(f"  vertices: {len(out.vertices)}")
    print(f"  faces:    {len(out.faces)}")
    print(f"  watertight: {out.is_watertight}")

    out.export(OUTPUT_STL)
    print(f"\nSaved STL: {OUTPUT_STL}")

    occ, voxel_size = _voxelize_target(out, LONGEST_AXIS_VOXELS, SPLIT)
    print(f"\nVoxelized occupancy shape: {occ.shape}")
    out_bounds = out.bounds
    out_extents = out_bounds[1] - out_bounds[0]
    sx, sy, sz = occ.shape
    dx = out_extents[0] / max(1, (sx - 1))
    dy = out_extents[1] / max(1, (sy - 1))
    dz = out_extents[2] / max(1, (sz - 1))
    print(f"  voxel pitch target: {voxel_size: .6f}")
    print(f"  voxel spacing: dx={dx: .6f}, dy={dy: .6f}, dz={dz: .6f}")

    labeled = prepare_voxel_mesh_txt(occ, expected_in_outs=None, num_type="int")
    labeled = label_end_planes(
        labeled,
        label_y_max=END_LABEL_Y_MAX,
        labels_y_min=END_LABELS_Y_MIN,
        sort_axes=END_SORT_AXES,
        sort_dirs=END_SORT_DIRECTIONS,
    )
    pad_mask = None
    if PAD_EMPTY_FACES:
        labeled, pad_mask = pad_empty_faces(
            labeled,
            keep_faces=EXPECTED_OUT_FACES,
            thickness=PAD_THICKNESS,
            value=PAD_VALUE,
            return_mask=True,
        )

    export_triplet(labeled, OUTPUT_DIR, OUTPUT_TEXT_NAME)
    print(f"Exported geom_/dim_/val_ to {OUTPUT_DIR}")
    print("Labels present:", np.unique(labeled))

    if SHOW_MAYAVI:
        viz_mesh = labeled
        if HIDE_PAD_IN_MAYAVI and pad_mask is not None:
            viz_mesh = labeled.copy()
            viz_mesh[pad_mask] = 0
        label_colors = {
            1: (0.2, 0.6, 1.0),
            2: (0.7, 0.7, 0.7),
            3: (0.1, 0.8, 0.2),
            4: (0.9, 0.6, 0.2),
            5: (0.95, 0.85, 0.2),
            END_LABEL_Y_MAX: (0.9, 0.2, 0.2),
            int(END_LABELS_Y_MIN[0]): (0.6, 0.2, 0.9),
            int(END_LABELS_Y_MIN[1]): (0.2, 0.8, 0.8),
            int(END_LABELS_Y_MIN[2]): (0.9, 0.5, 0.1),
        }
        label_names = {
            1: "fluid",
            2: "wall",
            3: "near-wall",
            4: "near-wall-2",
            5: "near-wall-3",
            END_LABEL_Y_MAX: "end_y_max",
            int(END_LABELS_Y_MIN[0]): "end_y_min_0",
            int(END_LABELS_Y_MIN[1]): "end_y_min_1",
            int(END_LABELS_Y_MIN[2]): "end_y_min_2",
        }
        visualize_labels(
            viz_mesh,
            label_colors=label_colors,
            label_names=label_names,
            show_labels=SHOW_LABELS,
            hollow=HOLLOW_VIEW,
        )


if __name__ == "__main__":
    main()
