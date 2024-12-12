import os
import cv2
from tqdm import tqdm
from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure

for _data_name in ['COD10K']:
    print("eval-dataset: {}".format(_data_name))
    pred_root = '/school/CSCI_2470/COD_Project/results/refined_archive_80/masks' 
    mask_root = '/school/CSCI_2470/COD_Project/data/Test/GT_Object'
    pred_name_list = sorted(os.listdir(pred_root))
    ious = []
    FM = Fmeasure()
    WFM = WeightedFmeasure()
    SM = Smeasure()
    EM = Emeasure()
    M = MAE()
    for pred_name in tqdm(pred_name_list, total=len(pred_name_list)):
        mask_path = os.path.join(mask_root, pred_name).replace('.jpg', '.png')
        pred_path = os.path.join(pred_root, pred_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        
        # compute iou
        intersection = (mask & pred).sum()
        union = (mask | pred).sum()
        iou = intersection / union
        ious.append(iou)
        
        mask_height, mask_width = mask.shape

        # change pred size
        pred = cv2.resize(pred, (mask_width, mask_height))
        FM.step(pred=pred, gt=mask)
        WFM.step(pred=pred, gt=mask)
        SM.step(pred=pred, gt=mask)
        EM.step(pred=pred, gt=mask)
        M.step(pred=pred, gt=mask)

    fm = FM.get_results()["fm"]
    wfm = WFM.get_results()["wfm"]
    sm = SM.get_results()["sm"]
    em = EM.get_results()["em"]
    mae = M.get_results()["mae"]
    iou = sum(ious) / len(ious)

    results = {
        "Smeasure": sm,
        "wFmeasure": wfm,
        "MAE": mae,
        "adpEm": em["adp"],
        "meanEm": em["curve"].mean(),
        "maxEm": em["curve"].max(),
        "adpFm": fm["adp"],
        "meanFm": fm["curve"].mean(),
        "maxFm": fm["curve"].max(),
        "IoU": iou
    }

    print(results)

print("Eval finished!")
