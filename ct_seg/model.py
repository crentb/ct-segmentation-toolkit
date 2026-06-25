"""
U-Net with Skip Connections for Semantic Segmentation of CT Images
==================================================================
Standard U-Net architecture with skip connections (concatenation),
Group Normalization, and configurable number of output classes.
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Conv3x3 -> GN -> LeakyReLU -> Conv3x3 -> GN -> LeakyReLU"""

    def __init__(self, in_ch, out_ch, groups=8):
        super().__init__()
        # Ensure groups doesn't exceed channel count
        g = min(groups, out_ch)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_groups=g, num_channels=out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_groups=g, num_channels=out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetSegmentation(nn.Module):
    """
    U-Net with skip connections for multi-class segmentation.

    Parameters
    ----------
    in_channels : int
        Number of input channels (1 for grayscale, 5 for 2.5D).
    num_classes : int
        Number of segmentation classes.
    base_filters : int
        Number of filters in the first encoder level. Doubles at each level.
    channels_per_group : int
        Channels per group for Group Normalization.
    """

    def __init__(self, in_channels=1, num_classes=4, base_filters=32, channels_per_group=8):
        super().__init__()
        f = base_filters

        # Encoder
        self.enc1 = DoubleConv(in_channels, f, groups=min(channels_per_group, f))
        self.enc2 = DoubleConv(f, f * 2, groups=min(channels_per_group, f * 2))
        self.enc3 = DoubleConv(f * 2, f * 4, groups=min(channels_per_group, f * 4))
        self.enc4 = DoubleConv(f * 4, f * 8, groups=min(channels_per_group, f * 8))

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(f * 8, f * 16, groups=min(channels_per_group, f * 16))

        # Decoder (input channels = skip + upsampled)
        self.up4 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec4 = DoubleConv(f * 16 + f * 8, f * 8, groups=min(channels_per_group, f * 8))

        self.up3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec3 = DoubleConv(f * 8 + f * 4, f * 4, groups=min(channels_per_group, f * 4))

        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec2 = DoubleConv(f * 4 + f * 2, f * 2, groups=min(channels_per_group, f * 2))

        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec1 = DoubleConv(f * 2 + f, f, groups=min(channels_per_group, f))

        # Output
        self.out_conv = nn.Conv2d(f, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)

    def predict(self, x):
        """Run forward pass and return class predictions (argmax)."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    model = UNetSegmentation(in_channels=1, num_classes=4, base_filters=32)
    print(f"Parameters: {count_parameters(model):,}")
    x = torch.randn(1, 1, 256, 256)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")  # [1, 4, 256, 256]
    pred = model.predict(x)
    print(f"Prediction: {pred.shape}")  # [1, 256, 256]
