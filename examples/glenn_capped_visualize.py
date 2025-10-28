#!/usr/bin/env python3
"""
Example: visualize and export the glenn_capped STL (single process).

This runs the STL → voxels route, saves the simulation-ready triplet
text output (geom_/dim_/val_), and opens a Mayavi window to show the
voxelized surface. Only the W, E, N faces are tagged for domain
boundary labels (11/12/11..16 mapping) on the fluid layer as required.

Usage:
  conda activate meshgen
  python examples/glenn_capped_visualize.py
"""

import os
from meshgen.geometry import Geometry


def main():
    # Label only West, East, North faces for domain tagging
    expected_faces = {"W", "E", "N"}

    stl_path = os.path.join(os.path.dirname(__file__), "glenn_capped.stl")
    geom = Geometry(
        stl_path=stl_path,            # path relative to this file
        resolution=4,                   # longest axis ~256 voxels for a quick run
        split=None,                     # no splitting
        num_processes=1,                # single process
        output_dir="examples/output",           # write triplet into output/
        expected_in_outs=expected_faces,
    )

    # Generate voxels
    geom.generate_voxel_mesh()
    vm = geom.get_voxel_mesh()
    if vm is not None:
        print("Voxel mesh shape:", vm.shape)

    # Save simulation-ready text triplet into output/
    geom.save_voxel_mesh_to_text("glenn_capped.txt")

    # Show voxelization with Mayavi
    geom.visualize()


if __name__ == "__main__":
    main()
