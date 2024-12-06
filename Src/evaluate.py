import os
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import logging
from datetime import datetime

from models.dqnet import DQnet
from utils.dataloader import test_dataset
from configs.config import Config

# Import metrics
from metrics.SM import SM
from metrics.EM import EM
from metrics.WFM import WFM


class Evaluator:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)

        # Setup logging
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join(config.logs_dir, 'eval_' + timestamp)
        os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'eval.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging

        # Initialize model
        self.model = DQnet(
            pretrained=True,
            vit_model=config.vit_model,
            feature_dim=config.feature_dim,
            window_size=config.window_size
        ).to(self.device)

        # Load test dataset
        self.test_dataset = test_dataset(
            image_root=os.path.join(config.test_path['image']),
            gt_root=os.path.join(config.test_path['gt']),
            testsize=config.img_size
        )

    def load_model(self, checkpoint_path):
        """Load trained model"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info(f"Loaded model checkpoint from {checkpoint_path}")

    def evaluate(self, checkpoint_path, save_predictions=True):
        """Evaluate model on test dataset"""
        self.load_model(checkpoint_path)
        self.model.eval()

        # Create directory for saving predictions
        save_path = os.path.join(self.config.results_dir, 'predictions')
        if save_predictions:
            os.makedirs(save_path, exist_ok=True)

        # Metrics recording
        metrics = {
            'SM': [],
            'EM': [],
            'WFM': [],
            'MAE': []
        }

        self.logger.info("Starting evaluation...")
        with torch.no_grad():
            for i in tqdm(range(self.test_dataset.size)):
                # Load data
                image, gt, name = self.test_dataset.load_data()
                gt_np = np.array(gt).astype(np.bool)

                # Forward pass
                image = image.to(self.device)
                outputs = self.model(image)
                pred = outputs['pred']

                # Post-process prediction
                pred = F.interpolate(pred, size=gt_np.shape, mode='bilinear', align_corners=False)
                pred_np = pred.squeeze().cpu().numpy()

                # Calculate metrics
                sm_score = SM(pred_np, gt_np)
                em_score = EM(pred_np, gt_np)
                wfm_score = WFM(pred_np, gt_np)
                mae_score = np.mean(np.abs(pred_np - gt_np))

                # Record metrics
                metrics['SM'].append(sm_score)
                metrics['EM'].append(em_score)
                metrics['WFM'].append(wfm_score)
                metrics['MAE'].append(mae_score)

                # Save prediction
                if save_predictions:
                    pred_255 = (pred_np * 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(save_path, name), pred_255)

        # Calculate mean scores
        mean_metrics = {k: np.mean(v) for k, v in metrics.items()}

        # Log results
        self.logger.info("\nEvaluation Results:")
        self.logger.info(f"Structure Measure (SM): {mean_metrics['SM']:.4f}")
        self.logger.info(f"Enhanced Measure (EM): {mean_metrics['EM']:.4f}")
        self.logger.info(f"Weighted F-measure (WFM): {mean_metrics['WFM']:.4f}")
        self.logger.info(f"Mean Absolute Error (MAE): {mean_metrics['MAE']:.4f}")

        # Save detailed results
        results_file = os.path.join(self.config.results_dir, 'evaluation_results.txt')
        with open(results_file, 'w') as f:
            f.write(f"Checkpoint: {checkpoint_path}\n\n")
            f.write("Mean Metrics:\n")
            for k, v in mean_metrics.items():
                f.write(f"{k}: {v:.4f}\n")

        return mean_metrics


def main():
    # Load config
    config = Config()

    # Create evaluator
    evaluator = Evaluator(config)

    # Specify checkpoint path
    checkpoint_path = os.path.join(config.checkpoints_dir, 'best_model.pth')

    try:
        # Run evaluation
        metrics = evaluator.evaluate(
            checkpoint_path=checkpoint_path,
            save_predictions=True
        )
    except KeyboardInterrupt:
        logging.info("Evaluation interrupted by user")
    except Exception as e:
        logging.error(f"Evaluation failed with error: {str(e)}")
        raise


if __name__ == '__main__':
    main()