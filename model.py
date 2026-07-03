import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, resnet50, EfficientNet_B0_Weights, ResNet50_Weights

class AttentionFusion(nn.Module):
    """
    Proposed Adaptive Weighted / Attention-based Fusion module.
    It takes features from two backbones, computes an attention weight for each,
    and returns the weighted sum/fused vector.
    """
    def __init__(self, in_features_1: int, in_features_2: int, fused_features: int = 1024):
        super().__init__()
        # Project both to same dimension
        self.proj1 = nn.Linear(in_features_1, fused_features)
        self.proj2 = nn.Linear(in_features_2, fused_features)
        
        # Attention score generator
        # Concatenate projected features and predict a weight between 0 and 1
        self.attention_fc = nn.Sequential(
            nn.Linear(fused_features * 2, fused_features // 2),
            nn.ReLU(),
            nn.Linear(fused_features // 2, 2), # 2 outputs for the 2 branches
            nn.Softmax(dim=1)
        )
        
    def forward(self, feat1, feat2):
        # feat1 shape: (B, in_features_1), feat2 shape: (B, in_features_2)
        p1 = self.proj1(feat1)
        p2 = self.proj2(feat2)
        
        combined = torch.cat([p1, p2], dim=1)
        
        # attention weights shape: (B, 2)
        weights = self.attention_fc(combined)
        w1 = weights[:, 0].unsqueeze(1) # (B, 1)
        w2 = weights[:, 1].unsqueeze(1) # (B, 1)
        
        # Adaptive weighted fusion
        fused = (p1 * w1) + (p2 * w2)
        return fused

class MultiBackboneFundusModel(nn.Module):
    """
    Multi-Input Generation & Feature Extraction model.
    Branch 1: Full Fundus -> EfficientNetB0
    Branch 2: Optic Disc ROI -> ResNet50
    """
    def __init__(self, num_classes: int = 5):
        super().__init__()
        
        # Branch 1: EfficientNetB0 (Primary)
        self.branch1 = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        # Remove classifier head from efficientnet
        in_features_1 = self.branch1.classifier[1].in_features
        self.branch1.classifier = nn.Identity()
        
        # Branch 2: ResNet50 (Secondary)
        self.branch2 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # Remove classifier head from resnet
        in_features_2 = self.branch2.fc.in_features
        self.branch2.fc = nn.Identity()
        
        fused_dim = 1024
        self.fusion = AttentionFusion(in_features_1, in_features_2, fused_features=fused_dim)
        
        # Dense Layers
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=0.4), # Regularization
            nn.Linear(512, num_classes)
        )
        
    def forward(self, full_image, od_roi):
        # full_image -> Branch 1 (EfficientNet)
        features_1 = self.branch1(full_image) # (B, in_features_1)
        
        # od_roi -> Branch 2 (ResNet50)
        features_2 = self.branch2(od_roi) # (B, in_features_2)
        
        # Feature Fusion
        fused_features = self.fusion(features_1, features_2)
        
        # Output Layer (Softmax is generally applied inside CrossEntropyLoss in PyTorch)
        # So we output logits here.
        logits = self.classifier(fused_features)
        
        return logits
