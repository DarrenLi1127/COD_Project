import torch
import torch.nn as nn
import timm
from einops import rearrange


class ViTEncoder(nn.Module):
    def __init__(self, pretrained=True, model_name='vit_base_patch16_384'):
        super().__init__()

        # Load pretrained ViT model
        self.vit = timm.create_model(model_name, pretrained=pretrained)

        # Remove classification head and keep only the feature extractor
        self.vit.head = nn.Identity()

        # Get embedding dimension
        self.embed_dim = self.vit.embed_dim
        self.patch_size = self.vit.patch_embed.patch_size[0]

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input image [B, C, H, W]
        Returns:
            torch.Tensor: ViT features [B, num_patches, embed_dim]
        """
        # Get features from ViT
        B = x.shape[0]
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.vit.pos_drop(x + self.vit.pos_embed)

        # Pass through transformer blocks
        for blk in self.vit.blocks:
            x = blk(x)
        x = self.vit.norm(x)

        # Remove CLS token
        x = x[:, 1:, :]

        return x

    def get_last_selfattention(self, x):
        """Get attention maps from last layer for visualization
        Args:
            x (torch.Tensor): Input image [B, C, H, W]
        Returns:
            torch.Tensor: Attention maps [B, num_heads, num_patches, num_patches]
        """
        B = x.shape[0]
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.vit.pos_drop(x + self.vit.pos_embed)

        for i, blk in enumerate(self.vit.blocks):
            if i < len(self.vit.blocks) - 1:
                x = blk(x)
            else:
                # Return attention map from last block
                return blk.attn.get_attention_map(x)


class LayerDecayValueAssigner:
    """Assign different learning rates for different layers"""

    def __init__(self, layer_decay=0.75):
        self.layer_decay = layer_decay

    def get_layer_decay(self, layer_id, num_layers):
        """Calculate decay rate for a specific layer"""
        return self.layer_decay ** (num_layers - layer_id)

    def get_decay_dict(self, model):
        """Get parameter groups with different learning rates"""
        parameter_group_names = {}
        parameter_group_vars = {}

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Determine layer id from parameter name
            if 'pos_embed' in name or 'cls_token' in name:
                layer_id = 0
            elif 'patch_embed' in name:
                layer_id = 0
            elif 'blocks' in name:
                layer_id = int(name.split('.')[1]) + 1
            else:
                layer_id = len(model.blocks)

            decay_rate = self.get_layer_decay(layer_id, len(model.blocks))

            group_name = f"layer_{layer_id}"
            if group_name not in parameter_group_names:
                parameter_group_names[group_name] = {
                    "decay_rate": decay_rate,
                    "params": []
                }
                parameter_group_vars[group_name] = {
                    "decay_rate": decay_rate,
                    "params": []
                }

            parameter_group_vars[group_name]["params"].append(param)
            parameter_group_names[group_name]["params"].append(name)

        return list(parameter_group_vars.values())


if __name__ == '__main__':
    # Test the ViT encoder
    model = ViTEncoder(pretrained=True)
    x = torch.randn(2, 3, 384, 384)
    features = model(x)
    print(f"Output features shape: {features.shape}")

    # Test attention map extraction
    attention = model.get_last_selfattention(x)
    print(f"Attention map shape: {attention.shape}")

    # Test layer decay assignment
    layer_decay = LayerDecayValueAssigner()
    param_groups = layer_decay.get_decay_dict(model.vit)
    print(f"Number of parameter groups: {len(param_groups)}")