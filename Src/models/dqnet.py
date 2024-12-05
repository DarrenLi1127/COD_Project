import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit import ViTEncoder
from .resnet import ResNetBackbone, ResNetFeaturePyramid
from .rbq import RBQ
from .decoder import Decoder


class DQnet(nn.Module):
    def __init__(self, pretrained=True, vit_model='vit_base_patch16_384', feature_dim=256, window_size=4):
        super().__init__()

        # Initialize backbones
        self.vit = ViTEncoder(pretrained=pretrained, model_name=vit_model)
        self.resnet = ResNetBackbone(pretrained=pretrained)
        self.feature_pyramid = ResNetFeaturePyramid(self.resnet, out_channels=feature_dim)

        # Initialize RBQ modules for different scales
        self.rbq_modules = nn.ModuleList([
            RBQ(vit_dim=self.vit.embed_dim, out_dim=feature_dim, window_size=window_size)
            for _ in range(4)
        ])

        # Initialize decoder
        self.decoder = Decoder(
            vit_dim=self.vit.embed_dim,
            cnn_dims=[feature_dim] * 4,
            feature_dim=feature_dim
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input image [B, C, H, W]
        Returns:
            dict: Dictionary containing predictions and enhanced features
        """
        B = x.shape[0]

        # Get ViT features
        vit_features = self.vit(x)  # [B, N, C]

        # Get ResNet features through feature pyramid
        cnn_features = self.feature_pyramid(x)  # List of [B, feature_dim, H, W]

        # Apply RBQ at each scale
        enhanced_features = []
        patch_size = x.shape[-1] // 16  # 384/16 = 24 for ViT patch size

        for i, (feat, rbq) in enumerate(zip(cnn_features, self.rbq_modules)):
            h, w = feat.shape[2:]

            # Reshape and interpolate ViT features to match current scale
            cur_vit_feat = vit_features.transpose(1, 2)  # [B, C, N]
            cur_vit_feat = cur_vit_feat.reshape(B, -1, patch_size, patch_size)  # [B, C, H', W']
            cur_vit_feat = F.interpolate(cur_vit_feat, size=(h, w), mode='bilinear', align_corners=False)
            cur_vit_feat = cur_vit_feat.permute(0, 2, 3, 1)  # [B, H, W, C]

            # Prepare CNN features
            cur_cnn_feat = feat.permute(0, 2, 3, 1)  # [B, H, W, C]

            # Apply RBQ
            enhanced_feat = rbq(cur_vit_feat, cur_cnn_feat)
            enhanced_feat = enhanced_feat.permute(0, 3, 1, 2)  # [B, C, H, W]
            enhanced_features.append(enhanced_feat)

        # Get predictions from decoder
        pred, decoder_feats = self.decoder(vit_features, enhanced_features)

        # Only return the main prediction and enhanced features
        return {
            'pred': torch.sigmoid(pred),
            'enhanced_feats': enhanced_features
        }


if __name__ == '__main__':
    # Test the model
    model = DQnet(pretrained=True)
    x = torch.randn(2, 3, 384, 384)
    outputs = model(x)

    print("Model outputs:")
    print(f"Main prediction shape: {outputs['pred'].shape}")
    print("Enhanced feature shapes:")
    for i, feat in enumerate(outputs['enhanced_feats']):
        print(f"Feature {i}: {feat.shape}")
