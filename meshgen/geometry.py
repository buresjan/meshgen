from voxels import voxelize_mesh, generate_lbm_mesh, prepare_voxel_mesh_txt
import os
import numpy as np
from utilities import array_to_textfile


class Geometry:
    def __init__(
        self,
        name,
        resolution,
        split=None,
        num_processes=1,
        output_dir="output",
        expected_in_outs=None,
        **kwargs,
    ):
        """
        Initialize the Geometry class with the specified parameters.

        Parameters:
        name (str): The base name of the .geo template file (without extension).
        resolution (int): The resolution for voxelization.
        split (int, optional): Number of segments to split the mesh into before voxelization.
                               If None, the mesh is voxelized without splitting. Default is None.
        num_processes (int, optional): Number of processes to use for parallel voxelization.
                                       Default is 1 (no parallelism).
        output_dir (str, optional): Directory where output files will be stored. Default is 'output'.
        **kwargs: Additional parameters for customizing the geometry.
        """
        self.name = name
        self.resolution = resolution
        self.split = split
        self.num_processes = num_processes
        self.output_dir = output_dir
        self.kwargs = kwargs

        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Store the voxelized geometry as an attribute
        self.voxelized_mesh = None

        # Store the lbm geometry as an attribute
        self.lbm_mesh = None

        self.expected_in_outs = expected_in_outs

    def generate_voxel_mesh(self):
        """
        Generate the voxelized mesh for the geometry based on the .geo template.
        """
        print(f"Generating voxel mesh for geometry '{self.name}'...")
        self.voxelized_mesh = voxelize_mesh(
            self.name,
            res=self.resolution,
            split=self.split,
            num_processes=self.num_processes,
            **self.kwargs,
        )
        print(f"Voxel mesh generation complete. Shape: {self.voxelized_mesh.shape}")

    def save_voxel_mesh(self, filename="voxel_mesh.npy"):
        """
        Save the voxelized mesh to a file in the specified output directory.

        Parameters:
        filename (str, optional): Name of the file to save the voxel mesh. Default is 'voxel_mesh.npy'.
        """
        if self.voxelized_mesh is not None:
            file_path = os.path.join(self.output_dir, filename)
            np.save(file_path, self.voxelized_mesh)
            print(f"Voxel mesh saved to {file_path}")
        else:
            print(
                "Error: No voxel mesh to save. Generate it first using 'generate_voxel_mesh'."
            )

    def save_voxel_mesh_to_text(self, filename="voxel_mesh.txt"):
        """
        Save the voxelized mesh as a text file in the specified output directory.

        Parameters:
        filename (str, optional): Name of the text file to save the voxel mesh. Default is 'voxel_mesh.txt'.
        """
        if self.voxelized_mesh is not None:
            output_mesh = prepare_voxel_mesh_txt(
                self.voxelized_mesh,
                expected_in_outs=self.expected_in_outs,
                num_type="int",
            )
            file_path = os.path.join(self.output_dir, filename)
            array_to_textfile(output_mesh, file_path)
            print(f"Voxel mesh saved as text to {file_path}")
        else:
            print(
                "Error: No voxel mesh to save. Generate it first using 'generate_voxel_mesh'."
            )

    def load_voxel_mesh(self, filename="voxel_mesh.npy"):
        """
        Load the voxelized mesh from a file.

        Parameters:
        filename (str, optional): Name of the file to load the voxel mesh from. Default is 'voxel_mesh.npy'.
        """
        file_path = os.path.join(self.output_dir, filename)
        if os.path.exists(file_path):
            self.voxelized_mesh = np.load(file_path)
            print(f"Voxel mesh loaded from {file_path}")
        else:
            print(f"Error: File {file_path} does not exist.")

    def visualize(self):
        """
        Visualize the voxelized mesh using the visualization function from utilities.
        """
        if self.voxelized_mesh is not None:
            from utilities import vis

            # output_mesh = prepare_voxel_mesh_txt(self.voxelized_mesh, expected_in_outs=self.expected_in_outs,
            #                                      num_type='int')
            #
            # boolean_array = (output_mesh == 3)
            vis(self.voxelized_mesh)
        else:
            print("Error: No voxel mesh to visualize. Generate or load it first.")

    def get_voxel_mesh(self):
        """
        Return the voxelized mesh.

        Returns:
        numpy.ndarray: The voxelized mesh array.
        """
        if self.voxelized_mesh is not None:
            return self.voxelized_mesh
        else:
            print("Error: No voxel mesh available. Generate or load it first.")
            return None

    def generate_lbm_mesh(self):
        """
        Generate the voxelized mesh for the geometry based on the .geo template.
        """
        print(f"Generating LBM mesh for geometry '{self.name}'...")

        if self.voxelized_mesh is None:
            self.lbm_mesh = None
        else:
            self.lbm_mesh = generate_lbm_mesh(
                self.voxelized_mesh, expected_in_outs=self.expected_in_outs
            )

            print(f"LBM mesh generation complete. Shape: {self.lbm_mesh.shape}")

    def save_lbm_mesh_to_text(self, filename="lbm_mesh.txt"):
        """
        Save the voxelized mesh as a text file in the specified output directory.

        Parameters:
        filename (str, optional): Name of the text file to save the voxel mesh. Default is 'voxel_mesh.txt'.
        """
        if self.voxelized_mesh is not None:
            file_path = os.path.join(self.output_dir, filename)
            array_to_textfile(self.lbm_mesh, file_path)
            print(f"Voxel mesh saved as text to {file_path}")
        else:
            print(
                "Error: No voxel mesh to save. Generate it first using 'generate_voxel_mesh'."
            )


if __name__ == "__main__":
    offsets = [
        -0.02,
        -0.01652893,
        -0.01338843,
        -0.01057851,
        -0.00809917,
        -0.00595041,
        -0.00413223,
        -0.00264463,
        -0.0014876,
        -0.00066116,
        -0.00016529,
        0.0,
        0.00016529,
        0.00066116,
        0.0014876,
        0.00264463,
        0.00413223,
        0.00595041,
        0.00809917,
        0.01057851,
        0.01338843,
        0.01652893,
        0.02,
    ]

    for offset in offsets:
        # Scale and round the offset to a fixed number of decimals
        scaled_offset = round(offset * 1e6)  # Scale to micro-units for readability
        # Format the identifier with consistent padding
        if offset < 0:
            identifier = f"RES_3_OFF_M{abs(scaled_offset):08d}"  # Zero-padded 8 digits
        else:
            identifier = f"RES_3_OFF_{scaled_offset:08d}"  # Zero-padded 8 digits

        geom = Geometry(
            name="junction_1d",
            resolution=3,
            split=3 * 128,
            num_processes=4,
            offset=offset,
            h=0.0005,
            expected_in_outs={"W", "E", "S", "N"},
        )

        geom.generate_voxel_mesh()
        geom.save_voxel_mesh_to_text(identifier + ".txt")