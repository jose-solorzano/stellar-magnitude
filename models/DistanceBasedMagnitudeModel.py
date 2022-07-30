import torch
from torch import nn

from models.ExtinctionModel import ExtinctionModel


class DistanceBasedMagnitudeModel(nn.Module):
    def __init__(self, num_color_metrics: int, est_d_param: float, est_log_d_param: float, est_bias: float,
                 log_d_param_factor=0.3, hidden=40):
        super().__init__()
        self.logd_param_factor = log_d_param_factor
        self.logd_param = nn.Parameter(torch.FloatTensor([[est_log_d_param * log_d_param_factor]]))
        self.est_bias = torch.as_tensor(est_bias - 0.5).float()
        self.ext_model = ExtinctionModel()
        self.color_model = nn.Sequential(
            nn.Linear(num_color_metrics + 1, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, distance: torch.Tensor, color_metrics: torch.Tensor, lat_rad: torch.Tensor, long_rad: torch.Tensor):
        log_distance = torch.log(distance)
        extinction = self.ext_model(distance, lat_rad, long_rad)
        std_cm = (color_metrics - 0.50) / 0.35
        cm_input = torch.hstack([std_cm, extinction])
        return self.est_bias + extinction + \
            log_distance * self.logd_param / self.logd_param_factor + \
            self.color_model(cm_input)
