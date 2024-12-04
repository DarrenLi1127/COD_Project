import cv2
import numpy as np


def reciprocalDissimilarity(gt: np.ndarray, pred: np.ndarray, lamd=0.5) -> float:
    # O_{BG/FG} = 1/D_{BG/FG}
    # D_{BG/FG} = 2*avg(x_{BG/FG})/(x_{BG/FG}^2 + 1 + 2*lambda*sigma_{BG/FG})
    epsilon = np.finfo(float).eps
    x = pred[gt].mean()
    sigma_x = pred[gt].std()
    score = 2 * x / (x ** 2 + 1 + 2 * lamd * sigma_x + epsilon)
    return score


def objectSimilarity(gt: np.ndarray, pred: np.ndarray) -> float:

    # compute mju, the ratio of foreground area to all pixels of gt
    gt = gt.flatten()
    pred = pred.flatten()
    mju = np.sum(gt) / gt.size
    pred_fg = pred.copy()
    pred_fg[~gt] = 0
    o_fg = reciprocalDissimilarity(gt, pred_fg)

    pred_bg = 1 - pred.copy()
    pred_bg[gt] = 0
    o_bg = reciprocalDissimilarity(~gt, pred_bg)
    return mju * o_fg + (1 - mju) * o_bg


def centroid(GT):
    """
    Compute the centroid of the ground truth mask

    Args:
        GT: 2D binary numpy array
    Returns:
        X, Y: coordinates of centroid
    """
    rows, cols = GT.shape

    if GT.sum() == 0:
        X = cols // 2
        Y = rows // 2
    else:
        total = GT.sum()
        i = np.arange(1, cols + 1)
        j = np.arange(1, rows + 1)

        X = round(np.sum(GT.sum(axis=0) * i) / total)
        Y = round(np.sum(GT.sum(axis=1) * j) / total)

    return X, Y


def divideGT(GT, X, Y):
    """
    Divide the ground truth into 4 regions according to the centroid

    Args:
        GT: 2D binary numpy array
        X, Y: centroid coordinates
    Returns:
        LT, RT, LB, RB: four regions of GT
        w1, w2, w3, w4: corresponding weights
    """
    hei, wid = GT.shape
    area = wid * hei

    # Copy the 4 regions
    LT = GT[:Y, :X]
    RT = GT[:Y, X:]
    LB = GT[Y:, :X]
    RB = GT[Y:, X:]

    # Calculate weights
    w1 = (X * Y) / area
    w2 = ((wid - X) * Y) / area
    w3 = (X * (hei - Y)) / area
    w4 = 1.0 - w1 - w2 - w3

    return LT, RT, LB, RB, w1, w2, w3, w4


def Divideprediction(prediction, X, Y):
    """
    Divide the prediction into 4 regions according to the centroid

    Args:
        prediction: 2D numpy array
        X, Y: centroid coordinates
    Returns:
        LT, RT, LB, RB: four regions of prediction
    """
    hei, wid = prediction.shape

    LT = prediction[:Y, :X]
    RT = prediction[:Y, X:]
    LB = prediction[Y:, :X]
    RB = prediction[Y:, X:]

    return LT, RT, LB, RB


def ssim(prediction, GT):
    """
    Compute the region similarity between foreground maps and ground truth

    Args:
        prediction: 2D numpy array with values in [0,1]
        GT: 2D binary numpy array
    Returns:
        Q: similarity score
    """
    dGT = GT.astype(float)

    hei, wid = prediction.shape
    N = wid * hei

    # Compute means
    x = prediction.mean()
    y = dGT.mean()

    # Compute variances
    sigma_x2 = ((prediction - x)**2).sum() / (N - 1 + np.finfo(float).eps)
    sigma_y2 = ((dGT - y)**2).sum() / (N - 1 + np.finfo(float).eps)

    # Compute covariance
    sigma_xy = ((prediction - x) * (dGT - y)).sum() / \
                (N - 1 + np.finfo(float).eps)

    alpha = 4 * x * y * sigma_xy
    beta = (x**2 + y**2) * (sigma_x2 + sigma_y2)

    if alpha != 0:
        Q = alpha / (beta + np.finfo(float).eps)
    elif alpha == 0 and beta == 0:
        Q = 1.0
    else:
        Q = 0

    return Q


def regionSimilarity(gt: np.ndarray, pred: np.ndarray) -> float:
    """
    Compute the region similarity between the foreground map and ground truth

    Args:
        prediction: 2D numpy array with values in [0,1]
        GT: 2D binary numpy array
    Returns:
        Q: region similarity score
    """
    # Find the centroid of the GT
    X, Y = centroid(gt)
    
    # Divide GT into 4 regions
    GT_1, GT_2, GT_3, GT_4, w1, w2, w3, w4 = divideGT(gt, X, Y)
    
    # Divide prediction into 4 regions
    prediction_1, prediction_2, prediction_3, prediction_4 = Divideprediction(pred, X, Y)
    
    # Compute the ssim score for each region
    Q1 = ssim(prediction_1, GT_1)
    Q2 = ssim(prediction_2, GT_2)
    Q3 = ssim(prediction_3, GT_3)
    Q4 = ssim(prediction_4, GT_4)
    
    # Combine the 4 scores
    Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4
    
    return Q

def SM(pred:np.ndarray, gt:np.ndarray, alpha=0.5) -> float:
    """
    Compute the structure similarity between the ground truth and the prediction.
    Adapted from : https://github.com/DengPingFan/S-measure
    """
    return alpha * objectSimilarity(gt, pred) + (1 - alpha) * regionSimilarity(gt, pred)