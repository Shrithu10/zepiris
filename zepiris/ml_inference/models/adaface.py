"""AdaFace IResNet-50 backbone with SE blocks for face recognition (CVPR 2022)."""

from __future__ import annotations

from collections import namedtuple

import torch
import torch.nn as nn
from torch.nn import (
    BatchNorm1d,
    BatchNorm2d,
    Conv2d,
    Dropout,
    Linear,
    Module,
    PReLU,
    Sequential,
)

# ---------------------------------------------------------------------------
# Block spec
# ---------------------------------------------------------------------------

_BlockSpec = namedtuple("BlockSpec", ["in_channel", "depth", "stride"])


def _make_layer(in_channel: int, depth: int, num_units: int, stride: int = 2) -> list[_BlockSpec]:
    return [_BlockSpec(in_channel, depth, stride)] + [
        _BlockSpec(depth, depth, 1) for _ in range(num_units - 1)
    ]


def _get_blocks(num_layers: int) -> list[list[_BlockSpec]]:
    if num_layers == 50:
        return [
            _make_layer(64, 64, 3),
            _make_layer(64, 128, 4),
            _make_layer(128, 256, 14),
            _make_layer(256, 512, 3),
        ]
    if num_layers == 100:
        return [
            _make_layer(64, 64, 3),
            _make_layer(64, 128, 13),
            _make_layer(128, 256, 30),
            _make_layer(256, 512, 3),
        ]
    raise ValueError(f"Unsupported num_layers: {num_layers}. Choose 50 or 100.")


# ---------------------------------------------------------------------------
# SE module
# ---------------------------------------------------------------------------


class _SEModule(Module):
    """Squeeze-and-Excitation channel attention block."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(channels, channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.avg_pool(x)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.sigmoid(scale)
        return x * scale


# ---------------------------------------------------------------------------
# Residual block (IR-SE variant)
# ---------------------------------------------------------------------------


class _IRSEBottleneck(Module):
    """Improved residual block with SE attention (used in AdaFace IR-SE models)."""

    def __init__(self, in_channel: int, depth: int, stride: int) -> None:
        super().__init__()
        if in_channel == depth:
            self.shortcut = nn.MaxPool2d(1, stride)
        else:
            self.shortcut = Sequential(
                Conv2d(in_channel, depth, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(depth),
            )
        self.res = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, kernel_size=3, stride=1, padding=1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, kernel_size=3, stride=stride, padding=1, bias=False),
            BatchNorm2d(depth),
            _SEModule(depth),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(x) + self.shortcut(x)


# ---------------------------------------------------------------------------
# IResNet backbone
# ---------------------------------------------------------------------------


class AdaFaceIR50(Module):
    """IResNet-50 with SE blocks — the AdaFace recognition backbone.

    Accepts 112×112 aligned RGB face crops normalized to [-1, 1].
    Returns a 512-d L2-normalized embedding vector per image.
    """

    def __init__(self) -> None:
        super().__init__()

        self.input_layer = Sequential(
            Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(64),
            PReLU(64),
        )

        blocks = _get_blocks(50)
        modules: list[Module] = []
        for layer in blocks:
            for spec in layer:
                modules.append(_IRSEBottleneck(spec.in_channel, spec.depth, spec.stride))
        self.body = Sequential(*modules)

        # 112 / (2^4) = 7  →  spatial map is 7×7 after four stride-2 layers
        self.output_layer = Sequential(
            BatchNorm2d(512),
            Dropout(0.4),
            nn.Flatten(),
            Linear(512 * 7 * 7, 512),
            BatchNorm1d(512, affine=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Aligned face tensor, shape (B, 3, 112, 112), values in [-1, 1]

        Returns:
            torch.Tensor: L2-normalized embeddings, shape (B, 512)
        """
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        norm = x.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
        return x / norm
