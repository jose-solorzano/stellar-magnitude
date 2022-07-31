import torch
from torch import nn

from models.ExtinctionModel import ExtinctionModel


class BandBasedMagnitudeModel(nn.Module):
    def __init__(self, est_band_params: torch.Tensor, num_bands: int, hidden=40):
        super().__init__()
        num_bands = est_band_params.size(0)
        self.linear_band_model = nn.Linear(num_bands, 1, bias=False)
        self.linear_band_model.weight = nn.Parameter(est_band_params.unsqueeze(0))
        self.non_linear_band_model = nn.Sequential(
            nn.Linear(num_bands, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, band_magnitudes: torch.Tensor, distance: torch.Tensor, color_metrics: torch.Tensor, lat_rad: torch.Tensor, long_rad: torch.Tensor):
        scaled_band_magnitudes = (band_magnitudes - 10.5) / 2.0
        return self.linear_band_model(band_magnitudes) +\
            self.non_linear_band_model(scaled_band_magnitudes)
