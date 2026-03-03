import os
import pickle
import numpy as np
from torch.utils.data import Dataset

class DataLoader_Venous_Valve(Dataset):
    def __init__(
        self,
        data_path: str,
        mode: str = "train",
        frames_num: int = 101,
        dt: int = 1,
        numeric = np.float32
    ):
        self.data_path = os.path.join(data_path, f"venous_valve_{mode}.pkl")
        self.mode = mode
        self.frames_num = frames_num
        self.dt = dt
        self.numeric = numeric

        self.f = pickle.load(open(self.data_path, "rb"))

    def __getitem__(self, index):
        # solid, fluid, interface
        f = self.f
        solid_frames = np.concatenate(
            (
                f[index]["solid"]["position"],
                f[index]["solid"]["stress"],
            ), axis = -1
        )

        fluid_frames = np.concatenate(
            (
                f[index]["fluid"]["position"],
                f[index]["fluid"]["pressure"],
                f[index]["fluid"]["velocity"]
            ), axis = -1
        )
        
        interface_frames = np.concatenate(
            (
                f[index]["interface"]["position"],
                f[index]["interface"]["stress"],
                f[index]["interface"]["pressure"],
                f[index]["interface"]["velocity"]
            ), axis = -1
        )

        current_solid_frames = solid_frames[0 : self.frames_num - self.dt : self.dt]
        current_fluid_frames = fluid_frames[0 : self.frames_num - self.dt : self.dt]
        current_interface_frames = interface_frames[0 : self.frames_num - self.dt : self.dt]

        next_solid_frames = solid_frames[self.dt : self.frames_num : self.dt]
        next_fluid_frames = fluid_frames[self.dt : self.frames_num : self.dt]
        next_interface_frames = interface_frames[self.dt : self.frames_num : self.dt]

        target_solid_frames = next_solid_frames - current_solid_frames
        target_fluid_frames = next_fluid_frames - current_fluid_frames
        target_interface_frames = next_interface_frames - current_interface_frames

        return current_solid_frames.astype(self.numeric), \
                current_fluid_frames.astype(self.numeric), \
                current_interface_frames.astype(self.numeric), \
                target_solid_frames.astype(self.numeric), \
                target_fluid_frames.astype(self.numeric), \
                target_interface_frames.astype(self.numeric), \
                next_solid_frames.astype(self.numeric), \
                next_fluid_frames.astype(self.numeric), \
                next_interface_frames.astype(self.numeric)
                
    def __len__(self):
        return len(self.f)
        
