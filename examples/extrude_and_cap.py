#!/usr/bin/env python3
import argparse
from collections import Counter, defaultdict

import numpy as np
import trimesh


# ---------------------------------------------------------------------
# Boundary loop detection and characterization
# ---------------------------------------------------------------------

def find_boundary_loops(mesh: trimesh.Trimesh):
    """
    Find boundary loops (each loop is an ordered list of vertex indices).
    Boundary edges are those that belong to exactly one face.
    """
    faces = mesh.faces

    # All directed edges from faces
    edges = np.vstack([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]],
    ])

    # Sort each edge so we can count undirected edges
    edges_sorted = np.sort(edges, axis=1)
    edge_tuples = [tuple(e) for e in edges_sorted]

    c = Counter(edge_tuples)
    boundary_edges = [e for e, count in c.items() if count == 1]

    # Build vertex adjacency along boundary
    adj = defaultdict(list)
    for a, b in boundary_edges:
        adj[a].append(b)
        adj[b].append(a)

    loops = []
    visited_edges = set()

    for a, b in boundary_edges:
        if (a, b) in visited_edges or (b, a) in visited_edges:
            continue

        loop = [a, b]
        visited_edges.add((a, b))
        visited_edges.add((b, a))

        cur = b
        prev = a

        while True:
            neighbors = adj[cur]
            next_v = None
            for n in neighbors:
                if n != prev and ((cur, n) not in visited_edges and (n, cur) not in visited_edges):
                    next_v = n
                    break

            if next_v is None:
                # Open chain; break out
                break

            if next_v == loop[0]:
                # Closed loop
                break

            loop.append(next_v)
            visited_edges.add((cur, next_v))
            visited_edges.add((next_v, cur))
            prev, cur = cur, next_v

        loops.append(loop)

    return loops


def characterize_loops(mesh: trimesh.Trimesh, loops):
    """
    For each loop, compute:
    - center
    - size (axis-aligned bounding box extents)
    - dominant plane normal axis = argmin(size)
    """
    verts = mesh.vertices
    infos = []

    for idx, loop in enumerate(loops):
        pts = verts[loop]
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        size = maxs - mins
        axis = int(np.argmin(size))  # axis ~ constant => plane normal
        center = pts.mean(axis=0)

        info = {
            "index": idx,
            "n": len(loop),
            "axis": axis,
            "center": center,
            "size": size,
            "mins": mins,
            "maxs": maxs,
        }
        infos.append(info)

    return infos


def select_end_loop2(mesh: trimesh.Trimesh, loops, loop_infos, end_name, tol_ratio=0.02):
    """
    Select the loop corresponding to a given open end:
        'YZ_maxX', 'YZ_minX', 'XZ_minY', 'XZ_maxY', 'XY_minZ', 'XY_maxZ'

    Strategy:
    - pick loops whose 'axis' (plane normal) matches
    - whose center along that axis is near the global min/max
    - among those, pick the one with the largest vertex count
    """
    bounds = mesh.bounds
    axis_map = {
        "YZ_maxX": (0, "max"),
        "YZ_minX": (0, "min"),
        "XZ_minY": (1, "min"),
        "XZ_maxY": (1, "max"),
        "XY_minZ": (2, "min"),
        "XY_maxZ": (2, "max"),
    }

    if end_name not in axis_map:
        raise ValueError(f"Unknown end_name {end_name}")

    axis, extreme = axis_map[end_name]
    target_val = bounds[1, axis] if extreme == "max" else bounds[0, axis]
    axis_extent = bounds[1, axis] - bounds[0, axis]
    tol = axis_extent * tol_ratio

    candidates = []
    for info in loop_infos:
        if info["axis"] != axis:
            continue
        dist = abs(info["center"][axis] - target_val)
        if dist <= tol:
            candidates.append(info)

    if not candidates:
        raise RuntimeError(f"No candidate loops found for {end_name}")

    best = max(candidates, key=lambda inf: inf["n"])
    return best


# ---------------------------------------------------------------------
# 2D triangulation helpers (ear clipping)
# ---------------------------------------------------------------------

def polygon_area_2d(points):
    """
    Signed area (positive for CCW) of a 2D polygon.
    points: (N, 2)
    """
    area = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return 0.5 * area


def point_in_triangle_2d(p, a, b, c, eps=1e-9):
    """
    Barycentric point-in-triangle test in 2D.
    """
    x, y = p
    x1, y1 = a
    x2, y2 = b
    x3, y3 = c

    denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    if abs(denom) < eps:
        return False

    l1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denom
    l2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denom
    l3 = 1.0 - l1 - l2

    return (l1 >= -eps) and (l2 >= -eps) and (l3 >= -eps)


def triangulate_polygon_2d(points, vertex_indices):
    """
    Triangulate a simple 2D polygon via ear clipping.

    points: (N, 2) in order
    vertex_indices: list of N indices into the 3D mesh vertex array,
                    same order as points.
    Returns a list of (i, j, k) faces in terms of original 3D vertex indices.
    """
    n = len(points)
    if n < 3:
        return []
    if n == 3:
        return [(vertex_indices[0], vertex_indices[1], vertex_indices[2])]

    # Ensure CCW orientation
    area = polygon_area_2d(points)
    if area < 0:
        points = points[::-1].copy()
        vertex_indices = vertex_indices[::-1]

    V = list(range(len(points)))
    faces = []
    max_iter = n * n
    iter_count = 0

    while len(V) > 3 and iter_count < max_iter:
        ear_found = False

        for idx in range(len(V)):
            i0 = V[idx - 1]
            i1 = V[idx]
            i2 = V[(idx + 1) % len(V)]

            p0 = points[i0]
            p1 = points[i1]
            p2 = points[i2]

            # Convexity test (z-component of 2D cross product > 0 for CCW)
            cross_z = (p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0])
            if cross_z <= 1e-10:
                continue

            # Check no other vertex lies inside that triangle
            ear = True
            for j in V:
                if j in (i0, i1, i2):
                    continue
                if point_in_triangle_2d(points[j], p0, p1, p2):
                    ear = False
                    break

            if not ear:
                continue

            # We have an ear
            faces.append((vertex_indices[i0], vertex_indices[i1], vertex_indices[i2]))
            V.pop(idx)
            ear_found = True
            break

        if not ear_found:
            # Degenerate or pathological; stop
            break

        iter_count += 1

    if len(V) == 3:
        i0, i1, i2 = V
        faces.append((vertex_indices[i0], vertex_indices[i1], vertex_indices[i2]))

    return faces


# ---------------------------------------------------------------------
# Capping a loop
# ---------------------------------------------------------------------

def cap_loop(mesh: trimesh.Trimesh, loop, axis):
    """
    Cap a single boundary loop lying approximately in a plane orthogonal to `axis`.
    Adds triangles in-place; loop is an ordered list of vertex indices.
    """
    verts = mesh.vertices
    loop = list(loop)

    plane_axes = [0, 1, 2]
    plane_axes.remove(axis)

    pts3d = verts[loop]
    pts2d = pts3d[:, plane_axes]

    # Remove consecutive duplicates in 2D
    cleaned_indices = []
    cleaned_pts2d = []
    prev = None
    for vidx, p in zip(loop, pts2d):
        if prev is None or np.linalg.norm(p - prev) > 1e-9:
            cleaned_indices.append(vidx)
            cleaned_pts2d.append(p)
            prev = p

    if len(cleaned_indices) < 3:
        return []

    cleaned_pts2d = np.asarray(cleaned_pts2d)
    faces = triangulate_polygon_2d(cleaned_pts2d, cleaned_indices)
    if not faces:
        return []

    new_faces = np.vstack([mesh.faces, np.array(faces, dtype=np.int64)])
    mesh.faces = new_faces
    return faces


def cap_open_ends(mesh: trimesh.Trimesh, ends=None):
    """
    Cap the four major open ends by default:
        YZ_maxX, XZ_minY, XZ_maxY, XY_maxZ

    `ends` can be overridden with a subset/superset of these identifiers.
    """
    if ends is None:
        ends = ["YZ_maxX", "XZ_minY", "XZ_maxY", "XY_maxZ"]

    mesh = mesh.copy()
    loops = find_boundary_loops(mesh)
    loop_infos = characterize_loops(mesh, loops)

    capped_stats = {}

    for end_name in ends:
        try:
            info = select_end_loop2(mesh, loops, loop_infos, end_name)
        except Exception as exc:
            capped_stats[end_name] = f"failed to find loop: {exc}"
            continue

        loop = loops[info["index"]]
        faces = cap_loop(mesh, loop, axis=info["axis"])
        capped_stats[end_name] = len(faces)

    return mesh, capped_stats


# ---------------------------------------------------------------------
# Extrusion with smooth narrowing
# ---------------------------------------------------------------------

def extrude_narrow_end(mesh: trimesh.Trimesh,
                       end_name="YZ_maxX",
                       length=14.0,
                       scale_target=0.75,
                       segments=10):
    """
    Extrude one open end and smoothly narrow its cross-section.

    end_name: which end to extrude:
        'YZ_maxX', 'XZ_minY', 'XZ_maxY', 'XY_maxZ', etc.
    length: extrusion length (in STL units)
    scale_target: scale factor at the tip (e.g. 0.75 = 75%)
    segments: number of interpolation steps for smooth taper
    """
    mesh = mesh.copy()

    loops = find_boundary_loops(mesh)
    loop_infos = characterize_loops(mesh, loops)

    target_info = select_end_loop2(mesh, loops, loop_infos, end_name)
    loop = loops[target_info["index"]]
    axis = target_info["axis"]

    bounds = mesh.bounds
    axis_map = {
        "YZ_maxX": (0, "max"),
        "YZ_minX": (0, "min"),
        "XZ_minY": (1, "min"),
        "XZ_maxY": (1, "max"),
        "XY_minZ": (2, "min"),
        "XY_maxZ": (2, "max"),
    }
    if end_name not in axis_map:
        raise ValueError(f"Unknown end_name {end_name}")

    axis_from_name, extreme = axis_map[end_name]
    if axis_from_name != axis:
        # Basic sanity check: our loop classification should match the requested end
        raise RuntimeError(f"Selected loop axis {axis} doesn't match {axis_from_name} for {end_name}")

    direction = 1.0 if extreme == "max" else -1.0

    verts = mesh.vertices.copy()
    faces = mesh.faces.copy()

    loop = list(loop)
    pts = verts[loop]
    center = pts.mean(axis=0)

    plane_axes = [0, 1, 2]
    plane_axes.remove(axis)

    offsets = pts[:, plane_axes] - center[plane_axes]  # in-plane offsets of rim

    rings = [loop]
    for k in range(1, segments + 1):
        t = k / float(segments)

        # Smoothstep profile: 3t^2 - 2t^3
        smooth = 3.0 * t * t - 2.0 * t * t * t
        s = 1.0 + (scale_target - 1.0) * smooth

        axis_offset = direction * length * t
        new_pts = pts.copy()

        # Shift along axis
        new_pts[:, axis] = pts[:, axis] + axis_offset

        # Scale in cross-sectional plane
        new_pts[:, plane_axes] = center[plane_axes] + s * offsets

        idx_start = len(verts)
        verts = np.vstack([verts, new_pts])
        new_ring = list(range(idx_start, idx_start + len(loop)))
        rings.append(new_ring)

    # Connect rings with quads -> triangles
    faces_list = [faces]
    ring_len = len(loop)

    for k in range(0, segments):
        ring_a = rings[k]
        ring_b = rings[k + 1]
        assert len(ring_a) == len(ring_b) == ring_len

        for i in range(ring_len):
            a0 = ring_a[i]
            a1 = ring_a[(i + 1) % ring_len]
            b0 = ring_b[i]
            b1 = ring_b[(i + 1) % ring_len]

            faces_list.append([[a0, a1, b1],
                               [a0, b1, b0]])

    faces2 = np.vstack(faces_list)

    new_mesh = trimesh.Trimesh(vertices=verts, faces=faces2, process=True)

    meta = {
        "target_loop_index": target_info["index"],
        "axis": axis,
        "extreme": extreme,
    }
    return new_mesh, meta


# ---------------------------------------------------------------------
# High-level convenience function
# ---------------------------------------------------------------------

def process_mesh(input_path,
                 output_path,
                 end_name="YZ_maxX",
                 length=14.0,
                 scale_target=0.75,
                 segments=10):
    """
    Load STL, extrude chosen end with smooth taper, cap the four major ends,
    and export STL.
    """
    mesh = trimesh.load_mesh(input_path)

    extruded_mesh, meta = extrude_narrow_end(
        mesh,
        end_name=end_name,
        length=length,
        scale_target=scale_target,
        segments=segments,
    )

    capped_mesh, capped_info = cap_open_ends(extruded_mesh)

    capped_mesh.export(output_path)
    print("Extrusion meta:", meta)
    print("Capped faces per end:", capped_info)


# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extrude and smoothly narrow a chosen open end of a mesh, then cap the main outlets."
    )
    parser.add_argument("input", help="Input STL file")
    parser.add_argument("output", help="Output STL file")

    parser.add_argument(
        "--end",
        default="YZ_maxX",
        choices=["YZ_maxX", "XZ_minY", "XZ_maxY", "XY_maxZ"],
        help="Which open end to extrude",
    )
    parser.add_argument(
        "--length",
        type=float,
        default=14.0,
        help="Extrusion length in STL units (≈ mm)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.75,
        help="Final cross-section scaling factor (e.g. 0.75 = 75%% of original)",
    )
    parser.add_argument(
        "--segments",
        type=int,
        default=10,
        help="Number of interpolation segments for smooth taper",
    )

    args = parser.parse_args()

    process_mesh(
        args.input,
        args.output,
        end_name=args.end,
        length=args.length,
        scale_target=args.scale,
        segments=args.segments,
    )
