import os
import h5py
import random
import numpy as np
from torch.utils.data import Dataset

class DataLoader_structure_oscillation(Dataset):
    def __init__(
        self,
        data_path: str,
        mode: str = "train",
        frames_num: int = 1000,
        dt: int = 4,
        numeric = np.float32
    ):
        self.data_path = os.path.join(data_path, "TF_fsi2_results")
        self.mode = mode
        self.frames_num = frames_num
        self.dt = dt
        self.numeric = numeric

        self._get_params()
        self._read_mesh()
        self._read_data()
        self._shuffle_data()

        if mode == "train":
            start, end = 0, int(0.8 * len(self.all_current_frames))
        elif mode == "eval":
            start, end = int(0.8 * len(self.all_current_frames)), int(0.9 * len(self.all_current_frames))
        else:
            start, end = int(0.9 * len(self.all_current_frames)), len(self.all_current_frames)

        self.all_current_frames = self.all_current_frames[start : end]
        self.all_next_frames = self.all_next_frames[start : end]
    
    def _get_params(self):
        self.mu_list = [1.0, 5, 10.0]
        self.x1_list = [-4.0, -2,0, 0.0, 2.0, 6.0, 4.0]
        self.x2_list = [-4.0, -2.0, 0, 2.0, 4.0, 6.0]

    def _read_mesh(self):
        mesh_path = os.path.join(self.data_path, "mesh.h5")
        f = h5py.File(mesh_path, 'r')

        # point positions
        self.mesh_coordinates = np.array(f["domains"]["coordinates"])
        
        # domain types, 1-fluid 2-solid
        self.mesh_values = np.array(f["domains"]["values"], dtype = np.int64)
        
        # cell index
        self.mesh_topology = np.array(f["domains"]["topology"], dtype = np.int64)

        self.mesh_fluid_idx, self.mesh_solid_idx = set(), set()

        for i in range(self.mesh_topology.shape[0]):
            if self.mesh_values[i] == 1:
                for j in range(self.mesh_topology.shape[1]):
                    self.mesh_fluid_idx.add(self.mesh_topology[i, j])
            else:
                for j in range(self.mesh_topology.shape[1]):
                    self.mesh_solid_idx.add(self.mesh_topology[i, j])

        self.mesh_interface_idx = self.mesh_fluid_idx & self.mesh_solid_idx

        self.mesh_fluid_idx -= self.mesh_interface_idx

        self.mesh_solid_idx -= self.mesh_interface_idx

    def _read_data(self):
        self.all_current_frames = []
        self.all_next_frames = []

        for mu in self.mu_list:
            for x1 in self.x1_list:
                for x2 in self.x2_list:
                    sample_path = os.path.join(
                        self.data_path, 
                        "mu=" + str(mu),
                        "x1=" + str(x1),
                        "x2=" + str(x2),
                        "Visualization"
                    )
                    if not os.path.exists(sample_path):
                        continue
                    
                    # displacement
                    displacement_path = os.path.join(
                        sample_path,
                        "displacement.h5"
                    )
                    displacement = h5py.File(displacement_path, 'r')["VisualisationVector"]

                    # pressure
                    pressure_path = os.path.join(
                        sample_path,
                        "pressure.h5"
                    )
                    pressure = h5py.File(pressure_path, 'r')["VisualisationVector"]

                    # velocity
                    velocity_path = os.path.join(
                        sample_path,
                        "velocity.h5"
                    )
                    velocity = h5py.File(velocity_path, 'r')["VisualisationVector"]
                    
                    frames = []
                    for i in range(0, self.frames_num, self.dt):
                        frame_coordinate = self.mesh_coordinates + np.array(displacement[str(i)])[:, :2]
                        frame_pressure = np.array(pressure[str(i)])
                        frame_velocity = np.array(velocity[str(i)])[:, :2]

                        frames.append(
                            np.concatenate(
                                (
                                    frame_coordinate,
                                    frame_pressure, 
                                    frame_velocity
                                ), 
                                axis = -1
                            )
                        )
                    self.all_current_frames += frames[:-1]
                    self.all_next_frames += frames[1:]


    def _shuffle_data(self, seed = 42):
        random.seed(seed)
        paired = list(zip(self.all_current_frames, self.all_next_frames))
        random.shuffle(paired)
        self.all_current_frames, self.all_next_frames = zip(*paired)
        self.all_current_frames = list(self.all_current_frames)
        self.all_next_frames = list(self.all_next_frames)


    def __getitem__(self, index):
        # fluid, solid, interface
        current_frame = self.all_current_frames[index]
        current_solid = current_frame[sorted(self.mesh_solid_idx)][:, :2]
        current_fluid = current_frame[sorted(self.mesh_fluid_idx)]
        current_interface = current_frame[sorted(self.mesh_interface_idx)][:, :-2]

        next_frame = self.all_next_frames[index]
        next_solid = next_frame[sorted(self.mesh_solid_idx)][:, :2]
        next_fluid = next_frame[sorted(self.mesh_fluid_idx)]
        next_interface = next_frame[sorted(self.mesh_interface_idx)][:, :-2]

        return current_solid.astype(self.numeric), \
                current_fluid.astype(self.numeric), \
                current_interface.astype(self.numeric), \
                next_solid.astype(self.numeric), \
                next_fluid.astype(self.numeric), \
                next_interface.astype(self.numeric)


    def __len__(self):
        return len(self.all_current_frames)
    