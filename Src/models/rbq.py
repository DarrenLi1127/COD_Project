import torch
import torch.nn as nn
import torch.nn.functional as F


class RBQ(nn.Module):
    def __init__(self, vit_dim=768, out_dim=256, window_size=4, num_heads=8):
        super().__init__()
        self.vit_dim = vit_dim
        self.out_dim = out_dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (out_dim // num_heads) ** -0.5

        # Dimension reduction for ViT features
        self.vit_reduction = nn.Linear(vit_dim, out_dim)

        # Normalization layers
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)

        # Attention projections
        self.q = nn.Linear(out_dim, out_dim)
        self.k = nn.Linear(out_dim, out_dim)
        self.v = nn.Linear(out_dim, out_dim)
        self.proj = nn.Linear(out_dim, out_dim)

    def window_partition(self, x, window_size):
        """Partition into windows"""
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size * window_size, C)
        return windows

    def window_reverse(self, windows, window_size, H, W):
        """Reverse window partition"""
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H, W, -1)
        return x

    def forward(self, vit_feat, cnn_feat):
        """
        Args:
            vit_feat (torch.Tensor): ViT features [B, H, W, C_vit]
            cnn_feat (torch.Tensor): CNN features [B, H, W, C_out]
        Returns:
            torch.Tensor: Enhanced features [B, H, W, C_out]
        """
        B, H, W, _ = vit_feat.shape
        shortcut = cnn_feat

        # Reduce ViT feature dimension
        vit_feat = self.vit_reduction(vit_feat)

        # Normalize inputs
        vit_feat = self.norm1(vit_feat)
        cnn_feat = self.norm2(cnn_feat)

        # Window partition
        vit_windows = self.window_partition(vit_feat, self.window_size)
        cnn_windows = self.window_partition(cnn_feat, self.window_size)

        # Project to Q, K, V
        q = self.q(vit_windows)
        k = self.k(cnn_windows)
        v = self.v(cnn_windows)

        # Reshape for multi-head attention
        q = q.reshape(-1, self.window_size * self.window_size, self.num_heads, self.out_dim // self.num_heads).permute(
            0, 2, 1, 3)
        k = k.reshape(-1, self.window_size * self.window_size, self.num_heads, self.out_dim // self.num_heads).permute(
            0, 2, 1, 3)
        v = v.reshape(-1, self.window_size * self.window_size, self.num_heads, self.out_dim // self.num_heads).permute(
            0, 2, 1, 3)

        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(-1, self.window_size * self.window_size, self.out_dim)

        # Apply projection
        x = self.proj(x)

        # Reverse window partition
        x = self.window_reverse(x, self.window_size, H, W)

        # Add residual connection
        x = x + shortcut

        return x


if __name__ == '__main__':
    # Test RBQ module
    B, H, W = 2, 96, 96
    vit_dim, out_dim = 768, 256
    window_size = 4

    rbq = RBQ(vit_dim=vit_dim, out_dim=out_dim, window_size=window_size)
    vit_feat = torch.randn(B, H, W, vit_dim)
    cnn_feat = torch.randn(B, H, W, out_dim)

    output = rbq(vit_feat, cnn_feat)
    print(f"Output shape: {output.shape}")