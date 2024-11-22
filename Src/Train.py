import os
import sys
import torch
from utils.Dataloader import CamObjDataset, get_loader

# Set up data paths and parameters
root = '../../COD10K-v3/COD10K-v3'

if os.path.exists(root):
    print(f"Found COD10K-v3 directory: {os.path.abspath(root)}")
else:
    print(f"COD10K-v3 directory not found at: {os.path.abspath(root)}")
    sys.exit(1)  # Exit the script with a non-zero status code

# Define paths to images and ground truth
image_root = os.path.join(root, 'Train', 'Image')
gt_root = os.path.join(root, 'Train', 'GT_Object')

# Ensure that image_root and gt_root end with a '/'
if not image_root.endswith('/'):
    image_root += '/'
if not gt_root.endswith('/'):
    gt_root += '/'

# Check if the image and ground truth directories exist
if not os.path.exists(image_root):
    print(f"Image directory not found at: {os.path.abspath(image_root)}")
    sys.exit(1)

if not os.path.exists(gt_root):
    print(f"Ground truth directory not found at: {os.path.abspath(gt_root)}")
    sys.exit(1)

# Set parameters
batch_size = 4
train_size = 352  # As used in your Dataloader.py

# Create data loader
data_loader = get_loader(
    image_root=image_root,
    gt_root=gt_root,
    batchsize=batch_size,
    trainsize=train_size,
    shuffle=True,
    num_workers=0,  # Set to 0 for stable training
    pin_memory=True
)

# Iterate over data loader and print some data
for i, (images, gts) in enumerate(data_loader):
    print(f"Batch {i + 1}:")
    print(f"Images shape: {images.shape}")  # Should be [batch_size, 3, train_size, train_size]
    print(f"Ground Truths shape: {gts.shape}")  # Should be [batch_size, 1, train_size, train_size]
    # You can print the actual tensor values if needed
    print(f"Images tensor: {images}")
    print(f"Ground Truth tensor: {gts}")
    # Break after the first batch to prevent printing too much data
    if i == 0:
        break
