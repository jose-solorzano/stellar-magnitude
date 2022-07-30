import torch
from torch import nn


class ExtinctionModel(nn.Module):
    def __init__(self, hidden=20):
        super().__init__()
        self.k1 = nn.Parameter(torch.FloatTensor([[0.5]]))
        self.k2 = nn.Parameter(torch.FloatTensor([[0.5]]))
        # self.positional_ext_model = nn.Sequential(
        #     nn.Linear(4, hidden),
        #     nn.GELU(),
        #     nn.Linear(hidden, 1),
        # )

    def forward(self, distance: torch.Tensor, lat_rad: torch.Tensor, long_rad: torch.Tensor):
        lat_sin = torch.sin(lat_rad)
        # lat_cos = torch.cos(lat_rad)
        # long_sin = torch.sin(long_rad)
        # long_cos = torch.cos(long_rad)
        # pos_x = torch.hstack([lat_sin, lat_cos, long_sin, long_cos])
        # pos_ext = self.positional_ext_model(pos_x)
        k2_t_lat_sin = 3.0 * self.k2 * lat_sin + torch.randn_like(lat_sin) * 1e-10
        dist_kp = distance / 1000.0
        return torch.abs(self.k1 * torch.tanh(dist_kp * k2_t_lat_sin) / k2_t_lat_sin)
