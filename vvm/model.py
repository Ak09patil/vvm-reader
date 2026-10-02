"""Tiny multi-task CNN: crop (3x64x64) -> continuous s (main) + 4-class stage (auxiliary).

Decision rule everywhere: class = stage_of(s_pred). The class head only regularises training.
hybrid=True also feeds the classical IVP features [log_ratio, measured_flag] to the heads.
"""
import torch
import torch.nn as nn


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2))


class TinyVVMNet(nn.Module):
    def __init__(self, hybrid=False, width=(16, 32, 64, 96)):
        super().__init__()
        self.hybrid = hybrid
        layers, cin = [], 3
        for c in width:
            layers.append(block(cin, c))
            cin = c
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        d = cin + (2 if hybrid else 0)
        self.reg = nn.Sequential(nn.Linear(d, 64), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(64, 1))
        self.cls = nn.Linear(d, 4)

    def forward(self, x, extra=None):
        h = self.features(x)
        if self.hybrid:
            h = torch.cat([h, extra], 1)
        return torch.sigmoid(self.reg(h)).squeeze(1), self.cls(h)


def n_params(m):
    return sum(p.numel() for p in m.parameters())
