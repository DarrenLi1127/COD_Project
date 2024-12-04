import os
import torch
from utils.Dataloader import CamObjDataset, get_loader
import numpy as np

# Set up data paths
base_path = '/home/hyli/school/COD_Project/data/ours'
image_root = os.path.join(base_path, 'Train/Image/')
gt_root = os.path.join(base_path, 'Train/GT_Object/')

# Create data loader
train_loader = get_loader(
    image_root=image_root,
    gt_root=gt_root,
    batchsize=1,  # Load just one image
    trainsize=384,
    shuffle=False
)

# Get first image and its ground truth
for images, gts in train_loader:
    print("\nFirst Training Image:")
    print(f"Image shape: {images.shape}")  # Will be [1, 3, 384, 384]
    print("\nSample of normalized image values (10x10 patch):")
    print("Red channel:")
    print(images[0, 0, :10, :10])  # First 10x10 pixels of red channel

    print("\nCorresponding Ground Truth:")
    print(f"GT shape: {gts.shape}")  # Will be [1, 1, 384, 384]
    print("\nSample of ground truth values (10x10 patch, 1=object, 0=background):")
    print(gts[0, 0, :10, :10])

    break