import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import argparse
from tqdm import tqdm

from models.dqnet import DQnet
from utils.dataloader import test_loader_faster
from configs.config import Config


class Tester:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)

        # Initialize model
        self.model = DQnet(
            pretrained=True,
            vit_model=config.vit_model,
            feature_dim=config.feature_dim,
            window_size=config.window_size
        ).to(self.device)

    def load_model(self, checkpoint_path):
        """Load trained model"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model checkpoint from {checkpoint_path}")
        self.model.eval()

    def predict_single_image(self, image_path, save_path=None):
        """Predict on a single image"""
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img = img.resize((self.config.img_size, self.config.img_size))
        img_tensor = self.config.test_transform(img).unsqueeze(0)

        # Get original image size for resizing prediction
        orig_size = Image.open(image_path).size[::-1]  # (H, W)

        # Predict
        with torch.no_grad():
            img_tensor = img_tensor.to(self.device)
            output = self.model(img_tensor)
            pred = output['pred']

            # Resize to original size
            pred = F.interpolate(pred, size=orig_size, mode='bilinear', align_corners=False)
            pred = pred.squeeze().cpu().numpy()

        # Save prediction if requested
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            pred_255 = (pred * 255).astype(np.uint8)
            cv2.imwrite(save_path, pred_255)

        return pred

    def test_batch(self, test_dir, save_dir):
        """Test on a directory of images"""
        # Create test dataloader
        test_loader = test_loader_faster(
            image_root=test_dir,
            testsize=self.config.img_size
        )
        test_loader = torch.utils.data.DataLoader(
            test_loader,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Test loop
        print("Starting batch testing...")
        with torch.no_grad():
            for images, image_paths in tqdm(test_loader):
                # Get image name
                image_name = os.path.basename(image_paths[0])
                save_path = os.path.join(save_dir, image_name.replace('.jpg', '.png'))

                # Predict
                images = images.to(self.device)
                outputs = self.model(images)
                pred = outputs['pred']

                # Get original image size
                orig_size = Image.open(image_paths[0]).size[::-1]

                # Resize prediction to original size
                pred = F.interpolate(pred, size=orig_size, mode='bilinear', align_corners=False)
                pred = pred.squeeze().cpu().numpy()

                # Save prediction
                pred_255 = (pred * 255).astype(np.uint8)
                cv2.imwrite(save_path, pred_255)


def main():
    parser = argparse.ArgumentParser(description='Test DQnet')
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'batch'],
                        help='Test mode: single image or batch')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input image or directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save prediction(s)')

    args = parser.parse_args()

    # Load config
    config = Config()

    # Create tester
    tester = Tester(config)

    # Load model
    tester.load_model(args.checkpoint)

    try:
        if args.mode == 'single':
            # Test single image
            print(f"Processing image: {args.input}")
            tester.predict_single_image(args.input, args.output)
            print(f"Saved prediction to: {args.output}")

        else:
            # Test batch
            print(f"Processing images from: {args.input}")
            tester.test_batch(args.input, args.output)
            print(f"Saved predictions to: {args.output}")

    except Exception as e:
        print(f"Testing failed with error: {str(e)}")
        raise


if __name__ == '__main__':
    main()