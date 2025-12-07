#!/usr/bin/env python3
"""Minimal voxelization + Mayavi view for master_combined_capped.stl."""

import os

import trimesh as trm

from meshgen.geometry import Geometry


def describe_physical_dimensions(path):
    mesh = trm.load(path)
    if isinstance(mesh, trm.Scene):
        mesh = trm.util.concatenate(tuple(mesh.geometry.values()))

    bounds = mesh.bounds
    extents = bounds[1] - bounds[0]
    print("Physical bounds (STL units):")
    print(
        f"  min: [{bounds[0,0]:.4f}, {bounds[0,1]:.4f}, {bounds[0,2]:.4f}] "
        f"max: [{bounds[1,0]:.4f}, {bounds[1,1]:.4f}, {bounds[1,2]:.4f}]"
    )
    print(
        "Physical size (x, y, z): "
        f"{extents[0]:.4f} x {extents[1]:.4f} x {extents[2]:.4f}"
    )


if __name__ == "__main__":
    stl_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, "master_combined_capped.stl")
    )

    describe_physical_dimensions(stl_path)

    geom = Geometry(stl_path=stl_path, resolution=3)
    geom.generate_voxel_mesh()
    geom.visualize()
