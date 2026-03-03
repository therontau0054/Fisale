import os
import pickle
import numpy as np
from torch.utils.data import Dataset

class DataLoader_Flexible_Wing(Dataset):
    def __init__(
        self,
        data_path: str,
        mode: str = "train",
        numeric = np.float32
    ):
        self.data_path = os.path.join(data_path, f"flexible_wing_{mode}.pkl")
        self.mode = mode
        self.numeric = numeric

        self.f = pickle.load(open(self.data_path, "rb"))

    def __getitem__(self, index):
        # solid, fluid, interface
        f = self.f
        init_solid = f[index]["solid"]["init_position"]
        init_solid = np.concatenate(
            (
                init_solid,
                np.array(
                    f[index]["wing_material"]
                ).reshape(1, -1).repeat(init_solid.shape[0], axis = 0)
            ), axis = -1
        )

        init_fluid = f[index]["fluid"]["init_position"]
        init_fluid = np.concatenate(
            (
                init_fluid,
                np.array(
                    f[index]["attack_angle"]
                ).reshape(1, -1).repeat(init_fluid.shape[0], axis = 0),
                np.array(
                    f[index]["wind_velocity"]
                ).reshape(1, -1).repeat(init_fluid.shape[0], axis = 0)
            ), axis = -1
        )

        init_interface = f[index]["interface"]["init_position"]
        init_interface = np.concatenate(
            (
                init_interface,
                np.array(
                    f[index]["wing_material"]
                ).reshape(1, -1).repeat(init_interface.shape[0], axis = 0),
                np.array(
                    f[index]["attack_angle"]
                ).reshape(1, -1).repeat(init_interface.shape[0], axis = 0),
                np.array(
                    f[index]["wind_velocity"]
                ).reshape(1, -1).repeat(init_interface.shape[0], axis = 0)
            ), axis = -1
        )

        final_solid = np.concatenate(
            (
                f[index]["solid"]["final_position"],
                f[index]["solid"]["stress"]
            ), axis = -1
        )

        final_fluid = np.concatenate(
            (
                f[index]["fluid"]["final_position"],
                f[index]["fluid"]["pressure"],
                f[index]["fluid"]["velocity"]
            ), axis = -1
        )
        
        final_interface = np.concatenate(
            (
                f[index]["interface"]["final_position"],
                f[index]["interface"]["stress"],
                f[index]["interface"]["pressure"],
                f[index]["interface"]["velocity"]
            ), axis = -1
        )

        return init_solid.astype(self.numeric), \
                init_fluid.astype(self.numeric), \
                init_interface.astype(self.numeric), \
                final_solid.astype(self.numeric), \
                final_fluid.astype(self.numeric), \
                final_interface.astype(self.numeric)

    def __len__(self):
        return len(self.f)
        
