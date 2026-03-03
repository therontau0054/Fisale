import torch.nn as nn

from .ale_grid import ALEModule

from typing import List, Dict


__all__ = ["Fisale"]

class Fisale(nn.Module):
    def __init__(
        self, 
        dim = 2, 
        input_quantity_dims: Dict = None, 
        output_quantity_dims: Dict = None,
        grid_num = 4,
        hidden_dims: List = [],
        grid_shapes: List[List] = None,
        coupling_steps: int = 3,
        neighbors_nums: List = None,
        heads_num: int = 8,
        mlp_ratio: int = 2,
        mlp_layers: int = 2,
        dropout = 0,
        act = "gelu"
    ):
        super().__init__()
        assert grid_num == len(hidden_dims) == len(grid_shapes)
        assert len(grid_shapes[0]) == dim

        if output_quantity_dims is None:
            output_quantity_dims = input_quantity_dims
        
        self.ale_module = ALEModule(
            dim = dim,
            input_quantity_dims = input_quantity_dims,
            output_quantity_dims = output_quantity_dims,
            grid_num = grid_num,
            hidden_dims = hidden_dims,
            grid_shapes = grid_shapes,
            coupling_steps = coupling_steps,
            neighbors_nums = neighbors_nums,
            heads_num = heads_num,
            mlp_ratio = mlp_ratio,
            mlp_layers = mlp_layers,
            dropout = dropout,
            act = act
        )
                
    def forward(self, solid, fluid, interface):
        output_solid, output_fluid, output_interface = \
            self.ale_module(solid, fluid, interface)

        return output_solid, output_fluid, output_interface
    