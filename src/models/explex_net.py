import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class VisionCNN(nn.Module):
    """Image branch: ResNet18 trunk → projection.
    Supports variable input channels by replacing the first conv when needed.
    """
    def __init__(self, out_dim: int = 128, train_backbone: bool = False, in_ch: int = 3):
        super().__init__()
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # If input channels differ from 3, replace the first conv and
        # initialize its weights by repeating the mean of the pretrained RGB weights.
        if in_ch != 3:
            old_w = m.conv1.weight.data  # [64,3,7,7]
            mean_w = old_w.mean(dim=1, keepdim=True)  # [64,1,7,7]
            new_w = mean_w.repeat(1, in_ch, 1, 1)  # [64,in_ch,7,7]
            new_conv = torch.nn.Conv2d(in_ch, m.conv1.out_channels,
                                       kernel_size=m.conv1.kernel_size,
                                       stride=m.conv1.stride,
                                       padding=m.conv1.padding,
                                       bias=(m.conv1.bias is not None))
            new_conv.weight.data.copy_(new_w)
            if m.conv1.bias is not None:
                new_conv.bias.data.zero_()
            m.conv1 = new_conv

        self.backbone = nn.Sequential(*list(m.children())[:-1])  # [B,512,1,1]
        if not train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.proj = nn.Linear(512, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x).flatten(1)      # [B,512]
        return F.relu(self.proj(h))          # [B,out_dim]

class AudioCNN(nn.Module):
    """Audio branch: small 2D CNN over log-mel spectrograms."""
    def __init__(self, out_dim: int = 128, in_ch: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).flatten(1)           # [B,128]
        return F.relu(self.proj(h))          # [B,out_dim]

class ExplexNet(nn.Module):
    """Explosion-or-Explanation classifier (late fusion)."""
    def __init__(self, vid_in_ch: int = 3, vid_dim: int = 128, aud_dim: int = 128):
        super().__init__()
        self.vision = VisionCNN(out_dim=vid_dim, in_ch=vid_in_ch)
        self.audio  = AudioCNN(out_dim=aud_dim)
        self.classifier = nn.Sequential(
            nn.Linear(vid_dim + aud_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)  # sigmoid
        )

    def forward(self, img: torch.Tensor, mel: torch.Tensor) -> torch.Tensor:
        zv = self.vision(img)
        za = self.audio(mel)
        z = torch.cat([zv, za], dim=1)
        return self.classifier(z)  # [B,1] (logits)
