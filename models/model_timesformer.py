import os
from pathlib import Path
import torch
import torch.nn as nn

from project_paths import PROJECT_ROOT, TIMESFORMER_K400_PRETRAINED, add_third_party_paths

add_third_party_paths()

from timesformer.models.vit import TimeSformer

class TimeSformerClassifier(nn.Module):
    """
    输入:  [B, V, T, C, H, W]
    输出:  {"per_view_logits": [B, V, num_actions]}
    """
    def __init__(self, num_actions, num_frames=16, pretrained_model=None):
        super().__init__()
        self.num_actions = num_actions

        if pretrained_model is None:
            pretrained_model = TIMESFORMER_K400_PRETRAINED

        # 核心修改：自动将权重文件名转换为绝对路径，彻底根治 FileNotFoundError
        if pretrained_model:
            abs_weight_path = Path(pretrained_model)
            if not abs_weight_path.is_absolute():
                abs_weight_path = PROJECT_ROOT / abs_weight_path
            
            if os.path.exists(abs_weight_path):
                pretrained_model = str(abs_weight_path)
            else:
                print(f"警告: 在路径 {abs_weight_path} 下未找到权重文件，将转为随机初始化！")
                pretrained_model = ""

        # 初始化官方 TimeSformer 模型
        self.model = TimeSformer(
            img_size=224,
            num_classes=num_actions,
            num_frames=num_frames,
            attention_type='divided_space_time',
            pretrained_model=pretrained_model
        )

        # 记录特征维度
        if hasattr(self.model, 'head'):
            self.feature_dim = self.model.head.in_features
        else:
            self.feature_dim = 768

    def forward_single_view(self, x):
        """
        x: [B, T, C, H, W]
        """
        # TimeSformer 官方要求输入为 [B, C, T, H, W]
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        logits = self.model(x)
        return logits

    def forward(self, inputs):
        """
        inputs: [B, V, T, C, H, W]
        """
        B, V, T, C, H, W = inputs.shape
        per_view_logits_list = []

        # 逐视角提取时空特征
        for v_idx in range(V):
            v_input = inputs[:, v_idx]  # [B, T, C, H, W]
            v_logits = self.forward_single_view(v_input)
            per_view_logits_list.append(v_logits)

        per_view_logits = torch.stack(per_view_logits_list, dim=1)

        return {
            "per_view_logits": per_view_logits,
        }

if __name__ == "__main__":
    # 快速自测
    num_actions = 10
    model = TimeSformerClassifier(num_actions=num_actions, num_frames=16).eval()
    inp = torch.randn(2, 4, 16, 3, 224, 224)
    out = model(inp)
    print("TimeSformer Loop output shape:", out["per_view_logits"].shape)
