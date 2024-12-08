import os
import cv2
from tqdm import tqdm
from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure

method='HGINet' 
for _data_name in ['COD10K']:
    print("eval-dataset: {}".format(_data_name))
    mask_root = '/school/CSCI_2470/COD_Project/data/Test/{}/'.format("GT_Object") # change path
    pred_root = '/school/CSCI_2470/COD_Project/results/predictions/' # change path
    mask_name_list = sorted(os.listdir(mask_root))
    ious = []
    FM = Fmeasure()
    WFM = WeightedFmeasure()
    SM = Smeasure()
    EM = Emeasure()
    M = MAE()
    for mask_name in tqdm(mask_name_list, total=len(mask_name_list)):
        mask_path = os.path.join(mask_root, mask_name)
        pred_path = os.path.join(pred_root, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        
        # # compute iou
        # intersection = (mask & pred).sum()
        # union = (mask | pred).sum()
        # iou = intersection / union
        # ious.append(iou)
        
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
    }

    print(results)

print("Eval finished!")
