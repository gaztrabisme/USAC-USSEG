"""
Storm bounding box regression model.

ResNet18 backbone with 1-channel adapter + 4-output bbox head.
Supports phased unfreezing for fine-tuning on small datasets.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class StormBboxNet(nn.Module):
    """ResNet18 backbone → bbox regression (4 normalized coordinates).

    1-channel adapter: replaces conv1(3→64) with conv1(1→64),
    initialized by averaging the 3 pretrained channel weights.

    Head: AdaptiveAvgPool → FC(512→128) → ReLU → Dropout → FC(128→4) → Sigmoid
    Internally predicts (cx, cy, w, h) in [0,1], then converts to
    (minX, minY, maxX, maxY) to guarantee valid (non-degenerate) boxes.
    """

    def __init__(self, pretrained=True):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        # 1-channel adapter: average pretrained 3-channel weights
        old_conv1 = backbone.conv1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                self.conv1.weight.copy_(old_conv1.weight.mean(dim=1, keepdim=True))

        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Bbox regression head — outputs (cx, cy, w, h) via Sigmoid
        self.head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 4),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        cxywh = self.head(x)  # (cx, cy, w, h) all in [0, 1]

        cx = cxywh[:, 0]
        cy = cxywh[:, 1]
        w = cxywh[:, 2]
        h = cxywh[:, 3]

        minX = (cx - w / 2).clamp(0, 1)
        minY = (cy - h / 2).clamp(0, 1)
        maxX = (cx + w / 2).clamp(0, 1)
        maxY = (cy + h / 2).clamp(0, 1)

        return torch.stack([minX, minY, maxX, maxY], dim=1)

    def freeze_backbone(self):
        """Phase 1: freeze all backbone layers, only head trains."""
        for name, param in self.named_parameters():
            if not name.startswith('head.'):
                param.requires_grad = False

    def unfreeze_top_blocks(self):
        """Phase 2: unfreeze layer3 + layer4."""
        for name, param in self.named_parameters():
            if name.startswith(('layer3.', 'layer4.')):
                param.requires_grad = True

    def unfreeze_all(self):
        """Phase 3: unfreeze everything."""
        for param in self.parameters():
            param.requires_grad = True
