from setuptools import find_packages, setup

setup(
    name="meshgen",
    version="0.1.0",
    author="Jan Bureš",
    description="Mesh Generation and Voxelization Toolkit",
    url="https://github.com/buresjan/meshgen",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[      # Core dependencies
        "numpy",
        "scipy",
        "trimesh",
        "gmsh",
        "mayavi",
        "tqdm"
    ],
    extras_require={        # Development dependencies
        "dev": [
            "black",
        ]
    },
    python_requires=">=3.8",  # Specify the compatible Python version
)

