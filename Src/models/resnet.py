import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class ResNetBackbone(nn.Module):
    """Modified ResNet50 for feature extraction"""

    def __init__(self, pretrained=True):
        super().__init__()

        # Load pretrained ResNet50
        if pretrained:
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            model = resnet50(weights=None)

        # Remove fully connected layer and average pooling
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool

        # Extract residual blocks
        self.layer1 = model.layer1  # 1/4
        self.layer2 = model.layer2  # 1/8
        self.layer3 = model.layer3  # 1/16
        self.layer4 = model.layer4  # 1/32

        # Store output channels for each stage
        self.channels = {
            'stage1': 256,  # layer1
            'stage2': 512,  # layer2
            'stage3': 1024,  # layer3
            'stage4': 2048,  # layer4
        }

        self._freeze_bn()

    def _freeze_bn(self):
        """Freeze all BatchNorm layers"""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input image [B, C, H, W]
        Returns:
            list: List of feature maps at different scales
        """
        # Initial convolution blocks
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x0 = self.maxpool(x)  # 1/4

        # Residual blocks
        x1 = self.layer1(x0)  # 1/4
        x2 = self.layer2(x1)  # 1/8
        x3 = self.layer3(x2)  # 1/16
        x4 = self.layer4(x3)  # 1/32

        return [x1, x2, x3, x4]

    def get_channels(self):
        """Return output channels for each stage"""
        return self.channels


class ResNetFeaturePyramid(nn.Module):
    """Feature Pyramid for ResNet with proper channel reduction"""

    def __init__(self, backbone, out_channels=256):
        super().__init__()
        self.backbone = backbone
        channels = backbone.get_channels()

        # Lateral connections for each stage
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(channels[f'stage{i}'], out_channels, 1)
            for i in range(1, 5)
        ])

        # Output convolutions for each stage
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(4)
        ])

    def _upsample_add(self, x, y):
        """Upsample x and add it to y"""
        return F.interpolate(x, size=y.shape[2:], mode='bilinear', align_corners=False) + y

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input image [B, C, H, W]
        Returns:
            list: List of processed feature maps at different scales
        """
        # Get backbone features
        backbone_features = self.backbone(x)

        # Process features from deep to shallow
        processed_features = []
        last_feature = None

        for i, (feat, lat_conv, out_conv) in enumerate(
                zip(backbone_features[::-1], self.lateral_convs[::-1], self.output_convs[::-1])
        ):
            # Reduce channels
            feat = lat_conv(feat)

            # Add upsampled feature if not the deepest layer
            if last_feature is not None:
                feat = self._upsample_add(last_feature, feat)

            # Apply output convolution
            feat = out_conv(feat)
            processed_features.append(feat)
            last_feature = feat

        # Reverse the list to maintain shallow to deep order
        return processed_features[::-1]


if __name__ == '__main__':
    # Test the ResNet backbone and feature pyramid
    batch_size = 2
    img_size = 384

    backbone = ResNetBackbone(pretrained=True)
    feature_pyramid = ResNetFeaturePyramid(backbone)

    x = torch.randn(batch_size, 3, img_size, img_size)
    features = feature_pyramid(x)

    print("Feature map shapes:")
    for i, feat in enumerate(features):
        print(f"Level {i}: {feat.shape}")