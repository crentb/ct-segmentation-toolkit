"""
model.py — U-Net architecture for 2.5D Noise2Inverse denoising.

Defines a compact 2D U-Net used as the denoiser. The variant used by the project
is `unet_ns_gn` ("no-skip, group-norm"): a U-Net WITHOUT the usual skip
connections between encoder and decoder, using Group Normalization and LeakyReLU
throughout. Empirically this no-skip design is a robust denoiser across different
CT samples (skip connections tend to let high-frequency noise bypass the network).

The network takes a stack of adjacent slices as input channels (5 for the 2.5D
setup) and outputs a single denoised slice. Helper modules:
    unet_box_gn        - double 3x3 conv block (conv -> GroupNorm -> LeakyReLU, x2)
    unet_bottleneck_gn - single 3x3 conv block at the bottleneck
    unet_down          - 2x2 max-pool downsampling
    unet_up            - nearest-neighbour 2x upsampling
    unet_ns_gn         - the full encoder/bottleneck/decoder network

Author:  Cameron Renteria <crentb23@gmail.com>
License: Apache-2.0 (see LICENSE)
"""

import torch
import torch.nn as nn


class unet_box_gn(torch.nn.Module):
    """Double convolution block: (Conv3x3 -> GroupNorm -> LeakyReLU) applied twice.

    Preserves spatial size (padding=1) while mapping `in_ch` -> `out_ch` channels.
    `groups` sets the number of GroupNorm groups.
    """

    def __init__(self, in_ch, out_ch, groups):
        super().__init__()
        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=groups, num_channels=out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=groups, num_channels=out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class unet_bottleneck_gn(torch.nn.Module):
    """Single convolution block (Conv3x3 -> GroupNorm -> LeakyReLU) for the bottleneck."""

    def __init__(self, in_ch, out_ch, groups):
        super().__init__()
        self.bn_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=groups, num_channels=out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.bn_conv(x)


class unet_up(torch.nn.Module):
    """Upsampling block: nearest-neighbour 2x upsampling (no learned parameters).

    The `ch` argument is accepted for interface symmetry but is unused.
    """

    def __init__(
        self,
        ch,
    ):
        super().__init__()
        self.down_scale = torch.nn.Sequential(torch.nn.Upsample(scale_factor=2, mode="nearest"))

    def forward(self, x):
        return self.down_scale(x)


class unet_down(torch.nn.Module):
    """Downsampling block: 2x2 max pooling (halves the spatial resolution)."""

    def __init__(self, ch):
        super().__init__()
        self.maxpool = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.maxpool(x)


class unet_ns_gn(torch.nn.Module):
    """No-skip U-Net with Group Normalization (the project's denoiser).

    Args:
        start_filter_size : base channel width; deeper stages use multiples of it.
        ich               : number of input channels (5 for 2.5D: five adjacent slices).
        och               : number of output channels (1 = a single denoised slice).
        channels_per_group: channels per GroupNorm group (sets num_groups internally).

    The forward pass is a straight encoder -> bottleneck -> decoder with NO skip
    connections; spatial resolution is restored purely by the upsampling stages.
    """

    def __init__(self, start_filter_size, ich=1, och=1, channels_per_group=8):
        super().__init__()
        # Stem: 1x1 conv lifting the `ich` input channels up to start_filter_size.
        self.in_box = torch.nn.Sequential(
            torch.nn.Conv2d(ich, start_filter_size, kernel_size=1, padding=0),
            nn.GroupNorm(
                num_groups=int((start_filter_size) / channels_per_group),
                num_channels=start_filter_size,
            ),
            nn.LeakyReLU(0.1, inplace=True),
        )
        # --- Encoder: progressively widen channels and halve spatial size ---
        self.box1 = unet_box_gn(
            start_filter_size,
            start_filter_size * 4,
            groups=int((start_filter_size * 4) / channels_per_group),
        )
        self.down1 = unet_down(start_filter_size * 4)

        self.box2 = unet_box_gn(
            start_filter_size * 4,
            start_filter_size * 8,
            groups=int((start_filter_size * 8) / channels_per_group),
        )
        self.down2 = unet_down(start_filter_size * 8)

        self.box3 = unet_box_gn(
            start_filter_size * 8,
            start_filter_size * 16,
            groups=int((start_filter_size * 16) / channels_per_group),
        )
        self.down3 = unet_down(start_filter_size * 16)

        # --- Bottleneck (lowest resolution) ---
        self.bottleneck = unet_bottleneck_gn(
            start_filter_size * 16,
            start_filter_size * 16,
            groups=int((start_filter_size * 16) / channels_per_group),
        )

        # --- Decoder: upsample back to full resolution (NO skip connections) ---
        self.up1 = unet_up(start_filter_size * 16)
        self.box4 = unet_box_gn(
            start_filter_size * 16,
            start_filter_size * 8,
            groups=int((start_filter_size * 8) / channels_per_group),
        )

        self.up2 = unet_up(start_filter_size * 8)
        self.box5 = unet_box_gn(
            start_filter_size * 8,
            start_filter_size * 4,
            groups=int((start_filter_size * 4) / channels_per_group),
        )

        self.up3 = unet_up(start_filter_size * 4)
        self.box6 = unet_box_gn(
            start_filter_size * 4,
            start_filter_size * 4,
            groups=int((start_filter_size * 4) / channels_per_group),
        )

        # Output head: project down to `och` output channel(s).
        self.out_layer = torch.nn.Sequential(
            torch.nn.Conv2d(start_filter_size * 4, start_filter_size * 2, kernel_size=1, padding=0),
            nn.GroupNorm(
                num_groups=int((start_filter_size * 2) / channels_per_group),
                num_channels=start_filter_size * 2,
            ),
            nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(start_filter_size * 2, och, kernel_size=1, padding=0),
        )

    def forward(self, x):
        output = self.in_box(x)

        output = self.box1(output)
        output = self.down1(output)

        output = self.box2(output)
        output = self.down2(output)

        output = self.box3(output)
        output = self.down3(output)

        output = self.bottleneck(output)

        output = self.up1(output)

        output = self.box4(output)
        output = self.up2(output)

        output = self.box5(output)
        output = self.up3(output)

        output = self.box6(output)

        output = self.out_layer(output)

        return output
