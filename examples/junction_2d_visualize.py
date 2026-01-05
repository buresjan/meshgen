#!/usr/bin/env python3
"""
Example: visualize and export the junction_2d geometry (single process).

This runs the .geo → STL → voxels route, saves the STL used for
voxelization into output/, writes the simulation-ready triplet text
output (geom_/dim_/val_), and opens a Mayavi window to show the
voxelized surface.

Usage:
  conda activate meshgen
  python examples/junction_2d_visualize.py
"""

import os
import shutil

from meshgen.geometry import Geometry
from meshgen.mesher import gmsh_surface


def main():
    output_dir = "output"
    resolution = 5  # keep small for a fast run
    geo_kwargs = {
        "lower_angle": 4,   # degrees
        "upper_angle": -3,  # degrees
        "upper_flare": 0.001,  # meters
        "lower_flare": 0.001,  # meters
        "offset": 0.001,  # meters
    }

    os.makedirs(output_dir, exist_ok=True)

    # Generate the STL from the .geo template so it matches voxelization.
    stl_temp_path = gmsh_surface("junction_2d", True, resolution=resolution, **geo_kwargs)
    stl_output_path = os.path.join(output_dir, "junction_2d_voxelization.stl")
    shutil.copyfile(stl_temp_path, stl_output_path)
    print(f"Saved voxelization STL to {stl_output_path}")

    geom = Geometry(
        name="junction_2d",
        resolution=resolution,
        split=None,          # do not split (no parallel voxelization)
        num_processes=1,     # single process
        output_dir=output_dir, # write triplet into output/
        stl_path=stl_output_path,
        **geo_kwargs,
    )

    # Generate voxels
    geom.generate_voxel_mesh()
    vm = geom.get_voxel_mesh()
    if vm is not None:
        print("Voxel mesh shape:", vm.shape)

    # Save simulation-ready text triplet into output/
    geom.save_voxel_mesh_to_text("junction_2d.txt")

    # Show voxelization with Mayavi
    geom.visualize()


if __name__ == "__main__":
    main()
