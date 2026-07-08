"""
model.py
--------
Defines the parking-spot occupancy classifier.

Architecture: transfer learning on ResNet-18 (pretrained on ImageNet),
with the final fully-connected layer replaced by a small classification
head for 2 classes: {empty, occupied}.

Why transfer learning?
A parking-spot patch is a small, low-resolution crop, and labelled
parking data is comparatively scarce. Fine-tuning a network already
trained on ImageNet gives the model strong low-level visual features
(edges, textures, shapes) for free, so it converges faster and
generalizes better than training a CNN from scratch on a small dataset.
"""
import torch
import torch.nn as nn
from torchvision import models


class ParkingSpotClassifier(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            self.backbone = models.resnet18(weights=weights)
        except Exception as e:
            # No internet access to download ImageNet weights (e.g. offline
            # environment) -> fall back to random initialization. On a
            # machine with internet access, pretrained=True will work and
            # is recommended for faster convergence / higher accuracy.
            print(f"[model] Could not download pretrained weights ({e}); "
                  f"falling back to random initialization.")
            self.backbone = models.resnet18(weights=None)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


def build_model(device: str = "cpu", pretrained: bool = True, freeze_backbone: bool = False):
    model = ParkingSpotClassifier(num_classes=2, pretrained=pretrained, freeze_backbone=freeze_backbone)
    return model.to(device)


CLASS_NAMES = ["empty", "occupied"]


if __name__ == "__main__":
    m = build_model(pretrained=False)
    dummy = torch.randn(4, 3, 64, 64)
    out = m(dummy)
    print("Output shape:", out.shape)  # (4, 2)
