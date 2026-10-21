import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleGaussianDecoder(nn.Module):
    """
    Decodes features at a single multi-scale level into Gaussian parameters.
    Each level produces: color(3) + opacity(1) + scale(3) + rotation(4) + conf(1) = 12 channels.
    """
    def __init__(self, in_channels, features=256):
        super().__init__()

        # Lightweight conv decoder: project -> refine -> output
        self.proj = nn.Conv2d(in_channels, features, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1)

        # Output: color(3) + opacity(1) + scale(3) + rotation(4) + conf(1) = 12
        out_dim = 12
        self.out_conv = nn.Conv2d(features // 2, out_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        Args:
            x: [B, C, H_patch, W_patch] - features from one multi-scale level
        Returns:
            gs_params: [B, 12, H_patch, W_patch]
                Channels: color(0:3), opacity(3:4), scale(4:7), rotation(7:11), conf(11:12)
        """
        x = F.relu(self.proj(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.out_conv(x)
        return x
