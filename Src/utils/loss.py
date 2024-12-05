import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCELoss(nn.Module):
    """Weighted Binary Cross Entropy Loss"""

    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        """
        Args:
            pred (torch.Tensor): Predicted map [B, 1, H, W]
            gt (torch.Tensor): Ground truth map [B, 1, H, W]
        Returns:
            torch.Tensor: Weighted BCE loss
        """
        # Calculate weights based on surrounding environment
        gt_center = F.avg_pool2d(gt, kernel_size=3, stride=1, padding=1)
        weights = torch.abs(gt - gt_center)

        bce = F.binary_cross_entropy(pred, gt, reduction='none')
        weighted_bce = (bce * (1 + weights)).mean()

        return weighted_bce


class WeightedIoULoss(nn.Module):
    """Weighted IoU Loss"""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, gt):
        """
        Args:
            pred (torch.Tensor): Predicted map [B, 1, H, W]
            gt (torch.Tensor): Ground truth map [B, 1, H, W]
        Returns:
            torch.Tensor: Weighted IoU loss
        """
        # Calculate weights based on surrounding environment
        gt_center = F.avg_pool2d(gt, kernel_size=3, stride=1, padding=1)
        weights = torch.abs(gt - gt_center)

        # Calculate intersection and union
        intersection = (pred * gt).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + gt.sum(dim=(2, 3)) - intersection

        # Calculate IoU
        iou = (intersection + self.eps) / (union + self.eps)

        # Apply weights
        weighted_iou = (iou * (1 + weights.mean(dim=(2, 3)))).mean()

        return 1 - weighted_iou


class SignificanceAwareLoss(nn.Module):
    """Significance-Aware Loss for detail querying supervision"""

    def __init__(self, beta=2.0):
        super().__init__()
        self.beta = beta
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, enhanced_feats, pred, gt):
        """
        Args:
            enhanced_feats (list): List of enhanced feature maps
            pred (torch.Tensor): Predicted map [B, 1, H, W]
            gt (torch.Tensor): Ground truth map [B, 1, H, W]
        Returns:
            torch.Tensor: SAL loss
        """
        total_loss = 0
        num_scales = len(enhanced_feats)

        for i, feat in enumerate(enhanced_feats):
            # Calculate significance weights
            feat_magnitude = torch.abs(feat).mean(dim=1, keepdim=True)
            weights = torch.sigmoid(feat_magnitude)

            # Upsample weights to ground truth size
            weights = F.interpolate(weights, size=gt.shape[2:], mode='bilinear', align_corners=False)

            # Calculate BCE loss with significance weights
            scale_pred = F.interpolate(pred, size=gt.shape[2:], mode='bilinear', align_corners=False)
            bce_loss = self.bce(scale_pred, gt)
            weighted_bce = bce_loss * (1 + self.beta * weights)

            # Add to total loss with scale-specific weight
            scale_weight = 0.8 ** i  # Lower weight for deeper layers
            total_loss += scale_weight * weighted_bce.mean()

        return total_loss / num_scales


class DQnetLoss(nn.Module):
    """Combined loss for DQnet"""

    def __init__(self, beta=2.0):
        super().__init__()
        self.w_bce = WeightedBCELoss()
        self.w_iou = WeightedIoULoss()
        self.sal = SignificanceAwareLoss(beta=beta)

    def forward(self, pred_dict, gt):
        """
        Args:
            pred_dict (dict): Dictionary containing:
                - 'pred': Final prediction [B, 1, H, W]
                - 'enhanced_feats': Enhanced feature maps
            gt (torch.Tensor): Ground truth map [B, 1, H, W]
        Returns:
            dict: Dictionary containing individual losses and total loss
        """
        # Main supervision
        bce_loss = self.w_bce(pred_dict['pred'], gt)
        iou_loss = self.w_iou(pred_dict['pred'], gt)

        # Detail querying supervision (Significance-Aware Loss)
        sal_loss = self.sal(pred_dict['enhanced_feats'], pred_dict['pred'], gt)

        # No auxiliary predictions
        total_loss = bce_loss + iou_loss + sal_loss

        return {
            'total': total_loss,
            'bce': bce_loss,
            'iou': iou_loss,
            'sal': sal_loss
        }


if __name__ == '__main__':
    # Test the losses
    batch_size = 2
    size = 352

    # Create dummy predictions and ground truth
    pred = torch.rand(batch_size, 1, size, size)
    gt = torch.randint(0, 2, (batch_size, 1, size, size)).float()
    enhanced_feats = [torch.rand(batch_size, 256, size // 4, size // 4) for _ in range(4)]

    pred_dict = {
        'pred': pred,
        'enhanced_feats': enhanced_feats
    }

    # Test losses
    criterion = DQnetLoss()
    losses = criterion(pred_dict, gt)

    print("Losses:")
    for name, value in losses.items():
        print(f"{name}: {value.item():.4f}")
