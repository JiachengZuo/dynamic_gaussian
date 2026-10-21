import torch
import torch.nn as nn
import torch.nn.functional as F


class AssignHead(nn.Module):
    """
    Assign Head: predicts per-position utility score S_i and Gaussian count n_i
    from fused temporal features.

    Input:  [B, C_fused, H_patch, W_patch]
    Output: S_i [B, H_patch, W_patch, 1], n_i [B, H_patch, W_patch, 1]
    """
    def __init__(self, in_channels, hidden_ch=256, eps=1e-6):
        super().__init__()
        self.eps = eps

        self.conv1 = nn.Conv2d(in_channels, hidden_ch, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_ch, hidden_ch // 2, kernel_size=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(hidden_ch // 2, 2, kernel_size=1, stride=1, padding=0)

        self.gaussian_count_max = 4  # one per multi-scale level

    def forward(self, fused_features):
        """
        Args:
            fused_features: [B, C_fused, H_patch, W_patch]
        Returns:
            S_i: utility score [B, H_patch, W_patch, 1], always > 0
            n_i: continuous gaussian count [B, H_patch, W_patch, 1], in range [0, 4]
        """
        x = F.relu(self.conv1(fused_features))
        x = F.relu(self.conv2(x))
        x = self.conv_out(x)  # [B, 2, H_patch, W_patch]

        # Channel 0 → S_i: softplus ensures positivity, +eps avoids zero
        S_i = F.softplus(x[:, 0:1, :, :]) + self.eps  # [B, 1, H_patch, W_patch]

        # Channel 1 → n_i: sigmoid scaled to [0, 4], continuous for training gradient flow
        n_i = self.gaussian_count_max * torch.sigmoid(x[:, 1:2, :, :])  # [B, 1, H_patch, W_patch]

        # Permute to [B, H_patch, W_patch, 1]
        S_i = S_i.permute(0, 2, 3, 1)
        n_i = n_i.permute(0, 2, 3, 1)

        return S_i, n_i
