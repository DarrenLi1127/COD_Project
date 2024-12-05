import os
from torchvision import transforms

class Config:
    # Project paths
    project_root = '/home/hyli/school/COD_Project'
    train_path = {
        'image': os.path.join(project_root, 'data/ours/Train/Image/'),
        'gt': os.path.join(project_root, 'data/ours/Train/GT_Object/')
    }
    test_path = {
        'image': os.path.join(project_root, 'data/ours/Test/Image/'),
        'gt': os.path.join(project_root, 'data/ours/Test/GT_Object/')
    }
    checkpoints_dir = os.path.join(project_root, 'checkpoints')
    logs_dir = os.path.join(project_root, 'logs')
    results_dir = os.path.join(project_root, 'results')

    # Create directories if they don't exist
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Model settings
    vit_model = 'vit_base_patch16_384'
    feature_dim = 256
    window_size = 4
    img_size = 384  # Changed to match ViT requirements

    # Training settings
    device = 'cuda'
    num_epochs = 10
    batch_size = 8  # Reduced due to larger image size
    num_workers = 4

    # Optimizer settings
    learning_rate = 1e-3
    weight_decay = 0.01
    lr_power = 0.9  # For poly learning rate schedule
    layer_decay = 0.75  # For ViT fine-tuning

    # Loss settings
    sal_beta = 2.0  # Beta for Significance-Aware Loss
    bce_weight = 1.0
    iou_weight = 1.0
    sal_weight = 1.0

    # Data transforms
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225])
    ])

    def __str__(self):
        """Pretty print configurations"""
        config_str = 'Configurations\n' + '=' * 50 + '\n'
        for attr in dir(self):
            if not attr.startswith('__'):
                config_str += f'{attr}: {getattr(self, attr)}\n'
        return config_str

config = Config()

if __name__ == '__main__':
    print(config)