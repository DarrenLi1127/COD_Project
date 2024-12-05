import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from datetime import datetime

from models.dqnet import DQnet
from utils.loss import DQnetLoss
from utils.dataloader import get_loader
from configs.config import Config


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)

        # Setup logging
        self.setup_logging()

        # Initialize model
        self.model = DQnet(
            pretrained=True,
            vit_model=config.vit_model,
            feature_dim=config.feature_dim,
            window_size=config.window_size
        ).to(self.device)

        # Initialize loss
        self.criterion = DQnetLoss(beta=config.sal_beta).to(self.device)

        # Create parameter groups with layer decay
        param_groups = self.create_optimizer_param_groups(
            self.model,
            config.learning_rate,
            config.layer_decay
        )

        # Initialize optimizer with parameter groups
        self.optimizer = optim.AdamW(param_groups)

        # Initialize learning rate scheduler
        self.scheduler = optim.lr_scheduler.PolynomialLR(
            self.optimizer,
            total_iters=config.num_epochs,
            power=config.lr_power
        )

        # Setup data loader using the correct paths from config
        self.train_loader = get_loader(
            image_root=config.train_path['image'],
            gt_root=config.train_path['gt'],
            batchsize=config.batch_size,
            trainsize=config.img_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True
        )

        # Best model tracking
        self.best_loss = float('inf')

    def create_optimizer_param_groups(self, model, lr, layer_decay):
        """Create parameter groups for layer-wise learning rate decay"""
        param_groups = []
        processed_params = set()

        # First handle ViT parameters
        num_layers = len(model.vit.vit.blocks)
        layer_scales = {}

        # Pre-compute decay rates for each layer
        for layer_id in range(num_layers + 2):
            scale = layer_decay ** (num_layers - layer_id)
            layer_scales[layer_id] = scale

        # Process ViT parameters
        for name, param in model.vit.named_parameters():
            if not param.requires_grad:
                continue

            # Add to processed params
            processed_params.add(param)

            # Determine layer id and scale
            if any(n in name for n in ['pos_embed', 'cls_token', 'patch_embed']):
                scale = layer_scales[0]
            elif 'blocks.' in name:
                block_id = int(name.split('blocks.')[1].split('.')[0])
                scale = layer_scales[block_id + 1]
            else:
                scale = layer_scales[num_layers + 1]

            param_group = {
                'params': [param],
                'lr': lr * scale,
                'weight_decay': self.config.weight_decay
            }
            param_groups.append(param_group)

        # Add remaining parameters (ResNet, RBQ, decoder) without layer decay
        remaining_params = []
        for name, param in model.named_parameters():
            if param.requires_grad and param not in processed_params:
                remaining_params.append(param)

        if remaining_params:
            param_groups.append({
                'params': remaining_params,
                'lr': lr,
                'weight_decay': self.config.weight_decay
            })

        return param_groups


    def setup_logging(self):
        """Setup logging configuration"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join(self.config.logs_dir, timestamp)
        os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'train.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging

    def save_checkpoint(self, epoch, loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss
        }

        checkpoint_path = os.path.join(
            self.config.checkpoints_dir,
            f'checkpoint_epoch_{epoch}.pth'
        )
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = os.path.join(
                self.config.checkpoints_dir,
                'best_model.pth'
            )
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model with loss: {loss:.4f}")

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0

        with tqdm(self.train_loader, desc=f'Epoch {epoch}') as pbar:
            for batch_idx, (images, gts) in enumerate(pbar):
                # Move data to device
                images = images.to(self.device)
                gts = gts.to(self.device)

                # Forward pass
                outputs = self.model(images)
                losses = self.criterion(outputs, gts)
                total_loss = losses['total']

                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                # Update progress bar
                epoch_loss += total_loss.item()
                pbar.set_postfix({
                    'loss': f"{total_loss.item():.4f}",
                    'lr': f"{self.scheduler.get_last_lr()[0]:.6f}"
                })

        epoch_loss /= len(self.train_loader)
        return epoch_loss

    def train(self):
        """Main training loop"""
        self.logger.info(f"Starting training with config:\n{self.config}")

        for epoch in range(self.config.num_epochs):
            # Train for one epoch
            epoch_loss = self.train_epoch(epoch)

            # Update learning rate
            self.scheduler.step()

            # Log progress
            self.logger.info(
                f"Epoch {epoch}: Loss={epoch_loss:.4f}, "
                f"LR={self.scheduler.get_last_lr()[0]:.6f}"
            )

            # Save checkpoint
            is_best = epoch_loss < self.best_loss
            if is_best:
                self.best_loss = epoch_loss
            self.save_checkpoint(epoch, epoch_loss, is_best)


if __name__ == '__main__':
    # Create trainer and start training
    config = Config()
    trainer = Trainer(config)

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("Training interrupted by user. Saving checkpoint...")
        trainer.save_checkpoint(-1, float('inf'), is_best=False)
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        raise