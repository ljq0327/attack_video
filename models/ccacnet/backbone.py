import torch
import torch.nn as nn

from project_paths import PYTORCHVIDEO_ROOT


class X3DBackbone(nn.Module):
    """
    Input:  [B, T, C, H, W]
    Output: [B, out_dim, T', 1, 1]
    """

    def __init__(self, model_name="x3d_m", out_dim=512, pretrained=True):
        super().__init__()
        model = torch.hub.load(
            str(PYTORCHVIDEO_ROOT),
            model_name,
            pretrained=pretrained,
            source="local",
        )
        self.blocks = nn.Sequential(*list(model.blocks[:-1]))
        self.proj = nn.Conv3d(192, out_dim, kernel_size=1, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool3d((8, 1, 1))

    def forward(self, inputs):
        x = inputs.permute(0, 2, 1, 3, 4)
        x = self.blocks(x)
        x = self.proj(x)
        x = self.avg_pool(x)
        return x
