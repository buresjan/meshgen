# Installation

This repo supports two installation paths: Conda (recommended for Mayavi/VTK) and
Python venv (pip).

## Option A: Conda environment (recommended)

1) Create the environment from `environment.yml` at the repo root:

```bash
conda env create -f environment.yml
# or, if you have mamba:
# mamba env create -f environment.yml
```

2) Activate it:

```bash
conda activate meshgen
```

The environment installs all runtime deps (numpy, scipy, trimesh, gmsh, mayavi,
vtk, pyqt, tqdm) and installs this repo in editable mode for development.

Notes:
- If you are on a headless server and do not need visualization, you can remove
  `mayavi`, `vtk`, and `pyqt` from `environment.yml` before creating the
  environment.
- Gmsh is provided from conda-forge with the Python API. Ensure OpenGL support
  is available for Mayavi.

## Option B: Python venv (pip)

1) Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2) Upgrade pip and install the package in editable mode:

```bash
python -m pip install --upgrade pip
pip install -e .
```

Optional:

```bash
# Development deps
pip install -e .[dev]
```

Notes:
- If you want to control dependencies manually (for example, to skip `gmsh` or
  visualization stacks), install with `--no-deps` and then add only what you
  need:

```bash
pip install -e . --no-deps
pip install numpy scipy trimesh tqdm
# Add gmsh and/or mayavi if you need them
```

- Ensure you are using `pip >= 21` so editable installs work with the PEP 517
  backend.
