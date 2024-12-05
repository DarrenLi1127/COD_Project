import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureAlign(nn.Module):
    """Feature alignment module to handle dimension inconsistency"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        return x


class Decoder(nn.Module):
    """
    Decoder network that fuses enhanced multi-scale features
    """

    def __init__(self, vit_dim=768, cnn_dims=[256, 512, 1024, 2048], feature_dim=256):
        super().__init__()
        self.feature_dim = feature_dim

        # Feature alignment layers for ViT features
        self.vit_align = FeatureAlign(vit_dim, feature_dim)

        # Feature alignment layers for CNN features at different scales
        self.cnn_aligns = nn.ModuleList([
            FeatureAlign(dim, feature_dim) for dim in cnn_dims
        ])

        # Final prediction head
        self.pred_head = nn.Sequential(
            nn.Conv2d(feature_dim * 4, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, 1, 1)
        )

    def _upsample_add(self, x, y):
        """Upsample x and add it to y"""
        return F.interpolate(x, size=y.shape[2:], mode='bilinear', align_corners=False) + y

    def forward(self, vit_feat, cnn_feats):
        """
        Args:
            vit_feat (torch.Tensor): Features from ViT [B, H*W, C]
            cnn_feats (list): List of CNN features at different scales
                              Each tensor has shape [B, Ci, Hi, Wi]
        Returns:
            torch.Tensor: Final prediction map
            list: Enhanced feature maps at different scales
        """
        B = vit_feat.shape[0]
        H, W = cnn_feats[-1].shape[2:]  # Use the size of the smallest feature map

        # Reshape ViT features to spatial form (24x24 patches for img_size=384)
        vit_feat = vit_feat.transpose(1, 2)  # [B, C, N]
        vit_feat = vit_feat.reshape(B, 768, 24, 24)

        # Match CNN scale (e.g. 12x12)
        vit_feat = F.interpolate(vit_feat, size=(H, W), mode='bilinear', align_corners=False)
        vit_feat = self.vit_align(vit_feat)

        # Align CNN features
        aligned_feats = [align(feat) for feat, align in zip(cnn_feats, self.cnn_aligns)]

        # Progressive feature fusion (from deep to shallow)
        enhanced_feats = []
        cur_feat = aligned_feats[-1] + vit_feat
        enhanced_feats.append(cur_feat)

        for feat in reversed(aligned_feats[:-1]):
            cur_feat = self._upsample_add(cur_feat, feat)
            enhanced_feats.append(cur_feat)

        # Reverse to shallow-to-deep order
        enhanced_feats = enhanced_feats[::-1]

        # Concatenate all enhanced features
        base_size = enhanced_feats[0].shape[2:]
        cat_feats = [F.interpolate(feat, size=base_size, mode='bilinear', align_corners=False)
                     for feat in enhanced_feats]
        cat_feats = torch.cat(cat_feats, dim=1)

        # Final prediction
        pred = self.pred_head(cat_feats)
        pred = F.interpolate(pred, scale_factor=4, mode='bilinear', align_corners=False)

        return pred, enhanced_feats


if __name__ == '__main__':
    # Test the decoder
    batch_size = 2
    img_size = 352
    feature_sizes = [img_size // 4, img_size // 8, img_size // 16, img_size // 32]

    vit_feat = torch.randn(batch_size, (img_size // 16) ** 2, 768)
    cnn_feats = [
        torch.randn(batch_size, dim, size, size)
        for dim, size in zip([256, 512, 1024, 2048], feature_sizes)
    ]

    decoder = Decoder()
    pred, enhanced_feats = decoder(vit_feat, cnn_feats)

    print(f"Prediction shape: {pred.shape}")
    print(f"Number of enhanced feature maps: {len(enhanced_feats)}")
    for i, feat in enumerate(enhanced_feats):
        print(f"Enhanced feature map {i} shape: {feat.shape}")
