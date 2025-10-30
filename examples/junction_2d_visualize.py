#!/usr/bin/env python3
"""
Example: visualize and export the junction_2d geometry (single process).

This runs the .geo → STL → voxels route, saves the simulation-ready
triplet text output (geom_/dim_/val_), and opens a Mayavi window to
show the voxelized surface.

Usage:
  conda activate meshgen
  python examples/junction_2d_visualize.py
"""

from meshgen.geometry import Geometry


def main():
    # Small angles/flare for a quick, representative geometry
    geom = Geometry(
        name="junction_2d",
        resolution=5,        # keep small for a fast run
        split=None,          # do not split (no parallel voxelization)
        num_processes=1,     # single process
        output_dir="output", # write triplet into output/
        lower_angle=0,      # degrees
        upper_angle=-10,      # degrees
        upper_flare=0.001,   # meters
        lower_flare=0.001,   # meters
        offset=0.02,   # meters
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
