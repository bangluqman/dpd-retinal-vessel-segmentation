"""Frozen U-Net, S2D-replacement, PMR-2, DPD-2, and DPD-4 definitions."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int | None = None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class _UNetCore(nn.Module):
    def __init__(self, n_channels: int = 1, n_classes: int = 1,
                 base_channels: int = 64, bilinear: bool = True):
        super().__init__()
        factor = 2 if bilinear else 1
        c = base_channels
        self.inc = DoubleConv(n_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        self.down4 = Down(c * 8, c * 16 // factor)
        self.up1 = Up(c * 16, c * 8 // factor, bilinear)
        self.up2 = Up(c * 8, c * 4 // factor, bilinear)
        self.up3 = Up(c * 4, c * 2 // factor, bilinear)
        self.up4 = Up(c * 2, c, bilinear)
        self.outc = nn.Conv2d(c, n_classes, 1)

    def decode(self, x1, x2, x3, x4, x5):
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


class StandardUNet(_UNetCore):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2)
        x4 = self.down3(x3); x5 = self.down4(x4)
        return self.decode(x1, x2, x3, x4, x5)


class SpaceToDepthResidual(nn.Module):
    """Four spatial phases projected beside, not instead of, max pooling."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels * 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        nn.init.zeros_(self.projection[-1].weight)
        nn.init.zeros_(self.projection[-1].bias)

    def forward(self, source: torch.Tensor, standard_downsampled: torch.Tensor):
        pb, pr = source.shape[-2] % 2, source.shape[-1] % 2
        if pb or pr:
            source = F.pad(source, [0, pr, 0, pb], mode="replicate")
        phases = F.pixel_unshuffle(source, 2)
        h, w = standard_downsampled.shape[-2:]
        residual = self.projection(phases[..., :h, :w])
        return F.relu(standard_downsampled + residual, inplace=False)


class PooledResidualControl(nn.Module):
    """Parameter-count-matched pooled residual control (not capacity proof)."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 1, bias=False) for _ in range(4)
        ])
        self.normalization = nn.BatchNorm2d(out_channels)
        nn.init.zeros_(self.normalization.weight)
        nn.init.zeros_(self.normalization.bias)

    def forward(self, pooled_feature: torch.Tensor, standard_downsampled: torch.Tensor):
        residual = sum(layer(pooled_feature) for layer in self.projections) / 4.0
        return F.relu(standard_downsampled + self.normalization(residual), inplace=False)


class SpaceToDepthReplacement(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.compression = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
        )

    def forward(self, source: torch.Tensor):
        if min(source.shape[-2:]) < 2:
            raise ValueError("Space-to-depth requires spatial dimensions >=2")
        h, w = source.shape[-2] // 2, source.shape[-1] // 2
        pb, pr = source.shape[-2] % 2, source.shape[-1] % 2
        if pb or pr:
            source = F.pad(source, [0, pr, 0, pb], mode="replicate")
        return self.compression(F.pixel_unshuffle(source, 2)[..., :h, :w])


class EarlyS2DReplacementUNet(_UNetCore):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        c = kwargs.get("base_channels", 64)
        with torch.random.fork_rng(devices=[]):
            self.s2d_down1 = SpaceToDepthReplacement(c)
            self.s2d_down2 = SpaceToDepthReplacement(c * 2)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1.block[1](self.s2d_down1(x1))
        x3 = self.down2.block[1](self.s2d_down2(x2))
        x4 = self.down3(x3); x5 = self.down4(x4)
        return self.decode(x1, x2, x3, x4, x5)


class DPDUNet(_UNetCore):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        c = kwargs.get("base_channels", 64)
        with torch.random.fork_rng(devices=[]):
            self.detail_down1 = SpaceToDepthResidual(c, c * 2)
            self.detail_down2 = SpaceToDepthResidual(c * 2, c * 4)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.detail_down1(x1, self.down1(x1))
        x3 = self.detail_down2(x2, self.down2(x2))
        x4 = self.down3(x3); x5 = self.down4(x4)
        return self.decode(x1, x2, x3, x4, x5)


class PMR2UNet(_UNetCore):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        c = kwargs.get("base_channels", 64)
        with torch.random.fork_rng(devices=[]):
            self.pooled_residual_down1 = PooledResidualControl(c, c * 2)
            self.pooled_residual_down2 = PooledResidualControl(c * 2, c * 4)

    def forward(self, x):
        x1 = self.inc(x)
        p2 = self.down1.block[0](x1)
        x2 = self.pooled_residual_down1(p2, self.down1.block[1](p2))
        p3 = self.down2.block[0](x2)
        x3 = self.pooled_residual_down2(p3, self.down2.block[1](p3))
        x4 = self.down3(x3); x5 = self.down4(x4)
        return self.decode(x1, x2, x3, x4, x5)


class DPDAllStagesUNet(_UNetCore):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        c = kwargs.get("base_channels", 64)
        factor = 2 if kwargs.get("bilinear", True) else 1
        with torch.random.fork_rng(devices=[]):
            self.detail_down1 = SpaceToDepthResidual(c, c * 2)
            self.detail_down2 = SpaceToDepthResidual(c * 2, c * 4)
            self.detail_down3 = SpaceToDepthResidual(c * 4, c * 8)
            self.detail_down4 = SpaceToDepthResidual(c * 8, c * 16 // factor)

    def forward(self, x):
        x1 = self.inc(x); x2 = self.detail_down1(x1, self.down1(x1))
        x3 = self.detail_down2(x2, self.down2(x2))
        x4 = self.detail_down3(x3, self.down3(x3))
        x5 = self.detail_down4(x4, self.down4(x4))
        return self.decode(x1, x2, x3, x4, x5)


MODEL_REGISTRY = {
    "unet_standard": StandardUNet,
    "unet_s2d_replace_early_ablation": EarlyS2DReplacementUNet,
    "unet_pmr2": PMR2UNet,
    "unet_dpd_v3": DPDUNet,
    "unet_dpd_all4_ablation": DPDAllStagesUNet,
}


def build_model(name: str, base_channels: int = 64) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown architecture {name!r}; choose {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](n_channels=1, n_classes=1,
                                base_channels=base_channels, bilinear=True)


def trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

