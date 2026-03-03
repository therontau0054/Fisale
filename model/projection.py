import torch
import torch.nn as nn

__all__ = ["Projection", "Deprojection"]

class Projection(nn.Module):
    def __init__(self, hidden_dim = 64, grid_length = 144):
        super().__init__()
        self.placeholder = nn.Parameter(
            torch.randn(1, 1, hidden_dim, dtype = torch.float) / hidden_dim
        )

        self.ww = nn.Sequential(
            nn.LayerNorm(grid_length), 
            nn.Linear(grid_length, grid_length)
        )
 
        self.wq = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
 
        self.wk = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, feature, ale_grid):
        q = self.wq(feature + self.placeholder)
        k = self.wk(ale_grid)
        weight = self.ww(q @ k.transpose(-1, -2))
        w = weight.softmax(-2).transpose(-1, -2)
        return w @ feature, weight
    
    
class Deprojection(nn.Module):
    def __init__(self, hidden_dim = 64):
        super().__init__()

        self.fc_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, p_proj, weight):
        _up = self.fc_out(weight.softmax(-1) @ p_proj)
        return _up
