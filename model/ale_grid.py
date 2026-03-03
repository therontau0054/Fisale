import torch
import torch.nn as nn

from .attention import AttentionLayer, FFN
from .projection import Projection, Deprojection
from typing import List, Dict
import torch_geometric.nn as nng

__all__ = ["ALEGrid", "ALEModule"]

class ALEGrid(nn.Module):
    def __init__(self, dim = 2, hidden_dim = 64, grid_shape = None, neighbors_num = 5):
        super().__init__()

        assert dim == len(grid_shape)
        self.neighbors_num = neighbors_num
    
        if dim == 2:
            self.grid = nn.Parameter(
                torch.stack(
                    torch.meshgrid(
                        torch.linspace(-3.5, 3.5, grid_shape[0]), 
                        torch.linspace(-3.5, 3.5, grid_shape[1]), 
                        indexing = "ij"
                    ), 
                    dim = -1
                ).reshape(1, -1, 2),
                requires_grad = False
            )

        elif dim == 3:
            self.grid = nn.Parameter(
                torch.stack(
                    torch.meshgrid(
                        torch.linspace(-3.5, 3.5, grid_shape[0]), 
                        torch.linspace(-3.5, 3.5, grid_shape[1]), 
                        torch.linspace(-3.5, 3.5, grid_shape[2]),
                        indexing = "ij"
                    ), 
                    dim = -1
                ).reshape(1, -1, 3),
                requires_grad = False
            )

        self.fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim)
        )

        self.w = nn.Sequential(
            nn.LayerNorm(self.grid.shape[1]),
            nn.Linear(self.grid.shape[1], self.grid.shape[1])
        )

    @staticmethod
    def knn(k, points):
        B, N, C = points.shape
        x = points.view(-1, C)
        batch_tensor = torch.arange(B, device = x.device).repeat_interleave(N)
        neighbor_idxes = nng.knn_graph(x, k = k, batch = batch_tensor)
        with torch.no_grad():
            src, dst = neighbor_idxes
            assert (batch_tensor[src] == batch_tensor[dst]).all(), "KNN batch error"
        return neighbor_idxes
    
    def forward(self, u):
        offsets = []
        for _u in u:
            direction_vec = _u[:, :, None] - self.grid[:, None, :] # B, N, M, D
            weight = (self.w(-torch.cdist(_u, self.grid, p = 2) ** 2)).softmax(1).unsqueeze(-1)
            offsets.append(torch.sum(weight * direction_vec, dim = 1))
        self.new_grid = self.fc(self.grid + sum(offsets))
        self.neighbor_idxes = self.knn(self.neighbors_num, self.new_grid)
        return self.new_grid, self.neighbor_idxes
    

class ALEGridUpdate(nn.Module):
    def __init__(
        self,
        hidden_dim = 64,
        heads_num = 8
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim)
        )

        self.aggr_p = nng.GATConv(
            in_channels = hidden_dim,
            out_channels = hidden_dim // heads_num,
            heads = heads_num,
            residual = True
        )
        self.ln_p = nn.LayerNorm(hidden_dim)

        self.aggr_u = nng.GATConv(
            in_channels = hidden_dim,
            out_channels = hidden_dim // heads_num,
            heads = heads_num,
            residual = True
        )
        self.ln_u = nn.LayerNorm(hidden_dim)
    
    def forward(self, u_proj, ps_proj, pf_proj, pb_proj, edge_index):

        p_proj = self.fc(torch.cat(
            (
                ps_proj, pf_proj, pb_proj
            ), 
            dim = -1
        )).reshape(-1, self.hidden_dim)

        update_u_proj = self.aggr_p(self.ln_p(p_proj), edge_index)
    
        return self.aggr_u(self.ln_u(update_u_proj), edge_index).reshape(u_proj.shape)

class ALEBlock(nn.Module):
    def __init__(
        self,
        hidden_dim = 64,
        grid_length = 144,
        heads_num = 8,
        mlp_ratio = 2,
        dropout = 0,
        act = "gelu"
    ):
        super().__init__()
        self.solid_proj_op = Projection(hidden_dim = hidden_dim, grid_length = grid_length)
        self.fluid_proj_op = Projection(hidden_dim = hidden_dim, grid_length = grid_length)
        self.interface_proj_op = Projection(hidden_dim = hidden_dim, grid_length = grid_length)

        self.update_solid = AttentionLayer(
            query_dim = hidden_dim,
            key_dim = hidden_dim,
            value_dim = hidden_dim,
            output_dim = hidden_dim,
            heads_num = heads_num,
            mlp_ratio = mlp_ratio,
            dropout = dropout,
            act = act,
            attn_type = "linear",
            linear_type = "galerkin",
            experts_num = 0
        )

        self.update_grid = ALEGridUpdate(
            hidden_dim = hidden_dim,
            heads_num = heads_num
        )

        self.update_fluid = AttentionLayer(
            query_dim = hidden_dim,
            key_dim = hidden_dim,
            value_dim = hidden_dim,
            output_dim = hidden_dim,
            heads_num = heads_num,
            mlp_ratio = mlp_ratio,
            dropout = dropout,
            act = act,
            attn_type = "linear",
            linear_type = "galerkin",
            experts_num = 0
        )

        self.interface_coupling = AttentionLayer(
            query_dim = hidden_dim,
            key_dim = hidden_dim,
            value_dim = hidden_dim,
            output_dim = hidden_dim,
            heads_num = heads_num,
            mlp_ratio = mlp_ratio,
            dropout = dropout,
            act = act,
            attn_type = "linear",
            linear_type = "galerkin",
            experts_num = 0
        )

        self.solid_deproj_op = Deprojection(hidden_dim = hidden_dim)
        self.fluid_deproj_op = Deprojection(hidden_dim = hidden_dim)
        self.interface_deproj_op = Deprojection(hidden_dim = hidden_dim)


        self.ffn_ale_grid = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            FFN(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, layer_num = 0, act = act)
        )


    def forward(self, solid_feature, fluid_feature, interface_feature, ale_grid, ale_edge):
        solid_proj, solid_weight = self.solid_proj_op(solid_feature, ale_grid)
        fluid_proj, fluid_weight = self.fluid_proj_op(fluid_feature, ale_grid)
        interface_proj, interface_weight = self.interface_proj_op(interface_feature, ale_grid)

        pos_embedding = ale_grid

        update_solid_proj, update_interface_proj = torch.chunk(
            self.update_solid(
                torch.cat((pos_embedding + solid_proj, pos_embedding + interface_proj), dim = 1),
                torch.cat((pos_embedding + solid_proj, pos_embedding + fluid_proj, pos_embedding + interface_proj), dim = 1),
                torch.cat((pos_embedding + solid_proj, pos_embedding + fluid_proj, pos_embedding + interface_proj), dim = 1),
            ),
            chunks = 2, 
            dim = 1
        )

        update_ale_grid = self.update_grid(
            ale_grid, update_solid_proj, fluid_proj, update_interface_proj, ale_edge
        )

        pos_embedding = update_ale_grid
        update_fluid_proj, update_interface_proj = torch.chunk(
            self.update_fluid(
                torch.cat((pos_embedding + fluid_proj, pos_embedding + update_interface_proj), dim = 1),
                torch.cat((pos_embedding + update_solid_proj, pos_embedding + fluid_proj, pos_embedding + update_interface_proj), dim = 1),
                torch.cat((pos_embedding + update_solid_proj, pos_embedding + fluid_proj, pos_embedding + update_interface_proj), dim = 1),
            ),
            chunks = 2, 
            dim = 1
        )

        update_solid_proj, update_fluid_proj, update_interface_proj = torch.chunk(
            self.interface_coupling(
                torch.cat((pos_embedding + update_solid_proj, pos_embedding + update_fluid_proj, pos_embedding + update_interface_proj), dim = 1),
                torch.cat((pos_embedding + update_solid_proj, pos_embedding + update_fluid_proj, pos_embedding + update_interface_proj), dim = 1),
                torch.cat((pos_embedding + update_solid_proj, pos_embedding + update_fluid_proj, pos_embedding + update_interface_proj), dim = 1),
            ),
            chunks = 3, 
            dim = 1
        )  

        update_solid_feature = self.solid_deproj_op(update_solid_proj, solid_weight) + solid_feature
        update_fluid_feature = self.fluid_deproj_op(update_fluid_proj, fluid_weight) + fluid_feature
        update_interface_feature = self.interface_deproj_op(update_interface_proj, interface_weight) + interface_feature

        update_ale_grid = update_ale_grid + ale_grid

        update_ale_grid = self.ffn_ale_grid(update_ale_grid) + update_ale_grid

        return update_solid_feature, update_fluid_feature, update_interface_feature, update_ale_grid


class ALEModule(nn.Module):
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
        heads_num = 8,
        mlp_ratio = 2,
        mlp_layers = 2,
        dropout = 0,
        act = "gelu"
    ):
        super().__init__()

        self.dim = dim
        self.grid_num = grid_num
        self.coupling_steps = coupling_steps
        self.hidden_dims = hidden_dims

        if output_quantity_dims is None:
            output_quantity_dims = input_quantity_dims
        self.output_quantity_dims = output_quantity_dims

        self.ale_grids = nn.ModuleList([
            ALEGrid(
                dim = dim, 
                hidden_dim = hidden_dims[i], 
                grid_shape = grid_shapes[i], 
                neighbors_num = neighbors_nums[i]
            )
            for i in range(grid_num)
        ])

        self.solid_pre_ffns = nn.ModuleList(
            [
                FFN(
                    dim + input_quantity_dims["solid"], 
                    hidden_dims[i] * mlp_ratio, 
                    hidden_dims[i], 
                    layer_num = 0, 
                    act = act
                )
                for i in range(grid_num)
            ]
        )

        self.fluid_pre_ffns = nn.ModuleList(
            [
                FFN(
                    dim + input_quantity_dims["fluid"], 
                    hidden_dims[i] * mlp_ratio, 
                    hidden_dims[i], 
                    layer_num = 0, 
                    act = act
                )
                for i in range(grid_num)
            ]
        )

        self.interface_pre_ffns = nn.ModuleList(
            [
                FFN(
                    dim + input_quantity_dims["interface"], 
                    hidden_dims[i] * mlp_ratio, 
                    hidden_dims[i], 
                    layer_num = 0, 
                    act = act
                )
                for i in range(grid_num)
            ]
        )

        grid_lengths = [1] * grid_num

        for i in range(grid_num):
            for g in grid_shapes[i]:
                grid_lengths[i] *= g


        self.ale_blocks = nn.ModuleList([
            nn.ModuleList([
                ALEBlock(
                    hidden_dim = hidden_dims[i],
                    grid_length = grid_lengths[i],
                    heads_num = heads_num,
                    mlp_ratio = mlp_ratio,
                    dropout = dropout,
                    act = act
                ) for _ in range(coupling_steps)
            ])
            for i in range(grid_num)
        ])

        self.ffn_solid = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(sum(hidden_dims)),
                    FFN(
                        sum(hidden_dims), 
                        sum(hidden_dims) * mlp_ratio,
                        sum(hidden_dims),
                        layer_num = 0,
                        act = act
                    )
                ) for _ in range(coupling_steps)
            ]
        )

        self.ffn_fluid = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(sum(hidden_dims)),
                    FFN(
                        sum(hidden_dims), 
                        sum(hidden_dims) * mlp_ratio,
                        sum(hidden_dims),
                        layer_num = 0,
                        act = act
                    )
                ) for _ in range(coupling_steps)
            ]
        )

        self.ffn_interface = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(sum(hidden_dims)),
                    FFN(
                        sum(hidden_dims), 
                        sum(hidden_dims) * mlp_ratio,
                        sum(hidden_dims),
                        layer_num = 0,
                        act = act
                    )
                ) for _ in range(coupling_steps)
            ]
        )

        self.fc_out = FFN(
            sum(hidden_dims), 
            max(hidden_dims) * mlp_ratio, 
            dim + max(output_quantity_dims.values()),
            layer_num = mlp_layers,
            act = act
        )


    def forward(self, solid, fluid, interface):
        solid_u, fluid_u = solid[:, :, :self.dim], fluid[:, :, :self.dim]
        interface_u = interface[:, :, :self.dim]

        ale_grids = [
            self.ale_grids[i]([solid_u, fluid_u, interface_u])
            for i in range(self.grid_num)
        ]

        ale_edges = [ale_grids[i][1] for i in range(self.grid_num)]
        ale_grids = [ale_grids[i][0] for i in range(self.grid_num)]

        solid_features = [self.solid_pre_ffns[i](solid) for i in range(self.grid_num)]
        fluid_features = [self.fluid_pre_ffns[i](fluid) for i in range(self.grid_num)]
        interface_features = [self.interface_pre_ffns[i](interface) for i in range(self.grid_num)]

        for j in range(self.coupling_steps):
            for i in range(self.grid_num):
                solid_features[i], fluid_features[i], interface_features[i], ale_grids[i] = \
                    self.ale_blocks[i][j](solid_features[i], fluid_features[i], interface_features[i], ale_grids[i], ale_edges[i])
            
            solid_features = list(map(lambda x, y: x + y, solid_features, torch.split(self.ffn_solid[j](torch.cat(solid_features, dim = -1)), self.hidden_dims, dim = -1)))
            fluid_features = list(map(lambda x, y: x + y, fluid_features, torch.split(self.ffn_fluid[j](torch.cat(fluid_features, dim = -1)), self.hidden_dims, dim = -1)))
            interface_features = list(map(lambda x, y: x + y, interface_features, torch.split(self.ffn_interface[j](torch.cat(interface_features, dim = -1)), self.hidden_dims, dim = -1)))

        n_solid, n_fluid = solid.shape[1], fluid.shape[1]
         
        output = self.fc_out(
            torch.cat(
                (
                    torch.cat(solid_features, dim = -1),
                    torch.cat(fluid_features, dim = -1),
                    torch.cat(interface_features, dim = -1)
                ), dim = 1
            )
        )

        update_solid = output[:, :n_solid, :self.dim + self.output_quantity_dims["solid"]]
        update_fluid = output[:, n_solid : n_solid + n_fluid, :self.dim + self.output_quantity_dims["fluid"]]
        update_interface = output[:, n_solid + n_fluid:, :self.dim + self.output_quantity_dims["interface"]]

        return update_solid, update_fluid, update_interface
    