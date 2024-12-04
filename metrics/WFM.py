import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt

def WFM(pred:np.ndarray, gt:np.ndarray) -> float:
    """
    Compute the Weighted F-beta measure (as proposed in "How to Evaluate Foreground Maps?" [Margolin et. al - CVPR'14])
    Adapted from: https://github.com/jiwei0921/Saliency-Evaluation-Toolbox/blob/master/saliency_evaluation/WFb.m
    
    Args:
        pred: Binary/Non binary foreground map with values in the range [0 1]. Type: numpy.ndarray (float)
        gt: Binary ground truth. Type: numpy.ndarray (bool)
    
    Returns:
        Q: The Weighted F-beta score
    """
    
    # Check input    
    if np.max(pred) > 1 or np.min(pred) < 0:
        raise ValueError('pred should be in the range of [0 1]')
    
    dGT = gt.astype(np.float64)  # Use float64 for computations
    E = np.abs(pred - dGT)
    
    # Calculate distance transform and indices
    Dst, IDXT = distance_transform_edt(1 - dGT, return_indices=True)
    
    # Create gaussian kernel
    size = 7
    sigma = 5
    x = np.linspace(-(size-1)/2, (size-1)/2, size)
    x, y = np.meshgrid(x, x)
    K = np.exp(-(x**2 + y**2)/(2*sigma**2))
    K = K / np.sum(K)
    
    # Pixel dependency
    Et = E.copy()
    
    # To deal correctly with the edges of the foreground region
    not_GT = ~gt
    Et[not_GT] = Et[IDXT[0][not_GT], IDXT[1][not_GT]]
    
    # Apply gaussian filter
    EA = ndimage.convolve(Et, K, mode='reflect')
    
    MIN_E_EA = E.copy()
    mask = np.logical_and(gt, EA < E)
    MIN_E_EA[mask] = EA[mask]
    
    # Pixel importance
    B = np.ones_like(gt, dtype=np.float64)
    B[not_GT] = 2 - np.exp(np.log(1-0.5)/5 * Dst[not_GT])
    
    Ew = MIN_E_EA * B
    
    TPw = np.sum(dGT) - np.sum(Ew[gt])
    FPw = np.sum(Ew[not_GT])
    
    # Weighted Recall
    R = 1 - np.mean(Ew[gt])
    
    # Weighted Precision
    eps = np.finfo(np.float64).eps
    P = TPw / (eps + TPw + FPw)
    
    # Beta = 1
    Q = (2 * R * P) / (eps + R + P)
    return Q