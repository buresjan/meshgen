# Mesh Generation and Voxelization Toolkit

## Overview

This package provides tools for generating and voxelizing 3D geometries using templates in the `.geo` format, working with STL models, and visualizing the results. It is designed to facilitate workflows involving geometry generation, voxelization, and preparation for numerical simulations like Lattice Boltzmann Method (LBM). The toolkit leverages `gmsh` for meshing and `trimesh` for handling 3D meshes, offering parallel processing for large datasets.

### Key Features

- **Geometry Generation**: Generate 3D geometries based on `.geo` templates.
- **Voxelization**: Convert complex 3D geometries into voxelized representations.
- **Parallel Processing**: Accelerate voxelization by splitting the mesh into segments and processing them in parallel.
- **Visualization**: Visualize the voxelized mesh using Mayavi.
- **Flexible Storage**: Save voxelized meshes as binary `.npy` files or as human-readable `.txt` files.

## Requirements

The package requires the following Python libraries:

- `gmsh`
- `trimesh`
- `numpy`
- `scipy`
- `mayavi`
- `tqdm`

## Installation

To install this package locally via `pip` (without publishing to PyPI), follow these steps:

1. Clone the repository or download the source code.
2. Navigate to the project directory and run the following command:

```bash
pip install .
```

### Installation for Development

If you want to install the package in development mode (editable installation), use the following command:

```bash
pip install -e .
```

This allows you to modify the source code while using the package without needing to reinstall it.

## Usage

Here is a quick example of how to use the `Geometry` class to generate, voxelize, and visualize a 3D geometry.

```python
from meshgen.geometry import Geometry

# Initialize a Geometry instance with parameters
geom = Geometry(name="tcpc_classic", resolution=5, split=5 * 128, num_processes=8)

# Generate the voxel mesh based on the .geo template
geom.generate_voxel_mesh()

# Save the voxel mesh as a binary .npy file
geom.save_voxel_mesh("tcpc_voxel_mesh.npy")

# Save the voxel mesh as a human-readable text file
geom.save_voxel_mesh_to_text("tcpc_voxel_mesh.txt")

# Visualize the voxel mesh
geom.visualize()

```
