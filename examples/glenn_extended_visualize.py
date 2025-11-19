#!/usr/bin/env python3
"""
Example: voxelize the glenn_extended STL at resolution 4 and visualize it.

This drives the STL → voxels route with the provided glenn_extended.stl
surface (two outlets on E/W, inlet on N). The resulting lattice is exported
as the usual geom_/dim_/val_ triplet before opening a Mayavi viewer.

Usage:
  conda activate meshgen
  python examples/glenn_extended_visualize.py
"""

import os

from meshgen.geometry import Geometry


def main():
    expected_faces = {"N", "E", "W"}  # inlet @N, outlets @E/W
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    stl_path = os.path.join(repo_root, "glenn_extended.stl")

    geom = Geometry(
        stl_path=stl_path,
        resolution=4,
        split=None,
        num_processes=1,
        output_dir="examples/output",
        expected_in_outs=expected_faces,
    )

    geom.generate_voxel_mesh()
    mesh = geom.get_voxel_mesh()
    if mesh is not None:
        print("glenn_extended voxel shape:", mesh.shape)

    geom.save_voxel_mesh_to_text("glenn_extended.txt")
    geom.visualize()


if __name__ == "__main__":
    main()
