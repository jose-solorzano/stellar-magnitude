import math
import unittest

import torch

from models.ExtinctionModel import ExtinctionModel


class TestExtinctionModel(unittest.TestCase):
    def test_extinction_gp(self):
        lat_rad = torch.FloatTensor([[0]])
        long_rad = torch.FloatTensor([[0]])
        em = ExtinctionModel()
        for d in range(0, 1500, 100):
            distance = torch.FloatTensor([[d]])
            extinction = em(distance, lat_rad, long_rad)
            print(f'D={d}, E={extinction.item():.4f}')

    def test_extinction_vertical(self):
        lat_rad = torch.FloatTensor([[math.sin(-math.pi / 2)]])
        long_rad = torch.FloatTensor([[-0.5]])
        em = ExtinctionModel()
        for d in range(0, 1500, 100):
            distance = torch.FloatTensor([[d]])
            extinction = em(distance, lat_rad, long_rad)
            print(f'D={d}, E={extinction.item():.4f}')
