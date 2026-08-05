

import torch
import torch.nn as nn
import timm

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for channel recalibration.
    """
    def __init__(self, in_features, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, in_features)
        self.fc2 = nn.Linear(in_features, reduction)  # Handles bottleneck reduction
        self.fc3 = nn.Linear(reduction, in_features)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        se = self.fc1(x)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.relu(se)
        se = self.fc3(se)
        se = self.sigmoid(se)
        return x * se


class SpatialCBAM_Token(nn.Module):
    """
    Spatial Attention Module for token-level feature refinement in Vision Transformers.
    """
    def __init__(self, kernel_size=7):
        super(SpatialCBAM_Token, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=2, keepdim=True)
        max_out, _ = torch.max(x, dim=2, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=2)
        x_cat = x_cat.transpose(1, 2)
        out = self.conv1(x_cat)
        attn_map = self.sigmoid(out.transpose(1, 2))
        return x * attn_map


class ViT_Hybrid_Refined(nn.Module):
    """
    Hybrid Deep Learning Model combining Vision Transformer (ViT) with
    Spatial CBAM and Channel Squeeze-and-Excitation (SE) Modules.
    """
    def __init__(self, vit_backbone, num_classes=2):
        super(ViT_Hybrid_Refined, self).__init__()
        self.vit = vit_backbone
        embed_dim = 768
        self.spatial_cbam = SpatialCBAM_Token(kernel_size=7)
        self.se = SEBlock(embed_dim, reduction=16)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Extract features/tokens from the pre-trained ViT backbone
        x_tokens = self.vit.forward_features(x)
        x_refined = self.spatial_cbam(x_tokens)
        
        # Isolate the primary CLS token
        cls_token = x_refined[:, 0]
        cls_refined = self.se(cls_token)
        
        out = self.classifier(cls_refined)
        return out


def get_model(num_classes=2, device='cpu'):
    """
    Instantiates and returns the Refined Hybrid Model on the target device.
    """
    print("Building Refined Hybrid Model (ViT + SpatialCBAM + SE)...")
    vit_backbone = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
    model = ViT_Hybrid_Refined(vit_backbone, num_classes=num_classes)
    model.to(device)
    return model
