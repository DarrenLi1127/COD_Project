import torch
import numpy as np
import torch.nn.functional as F


def EM_hybrid(pred, gt):
    """
        Compute the E-Measurement proposed in: Cognitive Vision Inspired Object Segmentation Metric and Loss Function
        Adapted from: https://github.com/GewelsJI/Hybrid-Eloss
    Args:
        pred: numpy.ndarray or torch.Tensor
        gt: numpy.ndarray or torch.Tensor
    Return: 
        scalar loss value
    """
    # Convert numpy arrays to torch tensors if needed
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred).float()
    if isinstance(gt, np.ndarray):
        gt = torch.from_numpy(gt).float()
    
    # Ensure inputs are 4D tensors (batch, channel, height, width)
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
    if gt.dim() == 2:
        gt = gt.unsqueeze(0)
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
    if gt.dim() == 3:
        gt = gt.unsqueeze(0)
    
    # adaptive weighting masks
    weit = 1 + 5 * torch.abs(F.avg_pool2d(gt, kernel_size=31, stride=1, padding=15) - gt)

    # weighted binary cross entropy loss function
    wbce = F.binary_cross_entropy_with_logits(pred, gt, reduction='none')
    wbce = ((weit * wbce).sum(dim=(2, 3)) + 1e-8) / (weit.sum(dim=(2, 3)) + 1e-8)

    # weighted e loss function
    pred = torch.sigmoid(pred)
    mpred = pred.mean(dim=(2, 3)).view(pred.shape[0], pred.shape[1], 1, 1).repeat(1, 1, pred.shape[2], pred.shape[3])
    phiFM = pred - mpred

    mmask = gt.mean(dim=(2, 3)).view(gt.shape[0], gt.shape[1], 1, 1).repeat(1, 1, gt.shape[2], gt.shape[3])
    phiGT = gt - mmask

    EFM = (2.0 * phiFM * phiGT + 1e-8) / (phiFM * phiFM + phiGT * phiGT + 1e-8)
    QFM = (1 + EFM) * (1 + EFM) / 4.0
    eloss = 1.0 - QFM.mean(dim=(2, 3))

    # weighted iou loss function
    inter = ((pred * gt) * weit).sum(dim=(2, 3))
    union = ((pred + gt) * weit).sum(dim=(2, 3))
    wiou = 1.0 - (inter + 1 + 1e-8) / (union - inter + 1 + 1e-8)

    return (wbce + eloss + wiou).mean()

def EM(pred, gt):
    """
    Compute the Enhanced Alignment measure
    
    Parameters:
    pred : numpy.ndarray
        Binary foreground map
    gt : numpy.ndarray
        Binary ground truth
    
    Returns:
    float
        The Enhanced alignment score
    """
    # Convert to boolean
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    
    # Convert to double for computations
    dFM = pred.astype(float)
    dGT = gt.astype(float)
    
    # Special cases
    if np.sum(gt) == 0:  # if gt is completely black
        enhanced_matrix = 1.0 - dFM
    elif np.sum(~gt) == 0:  # if gt is completely white
        enhanced_matrix = dFM
    else:
        # Normal case:
        # 1. compute alignment matrix
        align_matrix = AlignmentTerm(dFM, dGT)
        # 2. compute enhanced alignment matrix
        enhanced_matrix = EnhancedAlignmentTerm(align_matrix)
    
    # 3. Emeasure score
    h, w = gt.shape
    score = np.sum(enhanced_matrix) / (w * h - 1 + np.finfo(float).eps)
    
    return score

def AlignmentTerm(dFM, dGT):
    """
    Compute the alignment term
    """
    # compute global mean
    mu_FM = np.mean(dFM)
    mu_GT = np.mean(dGT)
    
    # compute the bias matrix
    align_FM = dFM - mu_FM
    align_GT = dGT - mu_GT
    
    # compute alignment matrix
    align_Matrix = 2 * (align_GT * align_FM) / (align_GT * align_GT + align_FM * align_FM + np.finfo(float).eps)
    
    return align_Matrix

def EnhancedAlignmentTerm(align_Matrix):
    """
    Enhanced Alignment Term function. f(x) = 1/4*(1 + x)^2)
    """
    return ((align_Matrix + 1) ** 2) / 4