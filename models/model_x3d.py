import torch
import torch.nn as nn

from project_paths import PYTORCHVIDEO_ROOT


class X3DClassifier(nn.Module):
    """
    输入:  [B, V, T, C, H, W]
    输出:  {"per_view_logits": [B, V, num_actions]}
    """
    def __init__(self, num_actions, model_name="x3d_m", pretrained=True):
        super().__init__()

        self.num_actions = num_actions

        self.model = torch.hub.load(
            str(PYTORCHVIDEO_ROOT),
            model_name,
            pretrained=pretrained,
            source="local",
        )

        head = self.model.blocks[-1]
        self.feature_dim = head.proj.in_features
        head.proj = nn.Linear(self.feature_dim, num_actions)

    def forward_single_view(self, x):
        """
        x: [B, T, C, H, W]
        """
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        logits = self.model(x)
        return logits

    def forward(self, inputs):
        """
        inputs: [B, V, T, C, H, W]
        """
        B, V, T, C, H, W = inputs.shape

        per_view_logits_list = []

        # 逐视角提取共享时空特征 / logits
        for v_idx in range(V):
            v_input = inputs[:, v_idx]  # [B, T, C, H, W]
            v_logits = self.forward_single_view(v_input)
            per_view_logits_list.append(v_logits)

        per_view_logits = torch.stack(per_view_logits_list, dim=1)

        return {
            "per_view_logits": per_view_logits,
        }

if __name__ == "__main__":
    # quick shape check
    num_actions = 10  # 你的动作类别数
    model = X3DClassifier(num_actions=num_actions, pretrained=False).eval()

    # 打印分类头信息
    print("="*60)
    print(f"self.feature_dim = {model.feature_dim}")
    cls_head = model.model.blocks[-1].proj
    print(f"cls_head: {cls_head}")
    print(f"in_features: {cls_head.in_features}, out_features:{cls_head.out_features}")

    # B=2 batch, V=4个视角, T=16帧, C=3通道, H=224 W=224
    inp = torch.randn(2, 4, 16, 3, 224, 224)
    print("input shape:", inp.shape)

    with torch.no_grad():
        out_dict = model(inp)

    per_view_logits = out_dict["per_view_logits"]
    print("per_view_logits shape:", per_view_logits.shape)  # [B, V, num_actions]
    print("output dtype:", per_view_logits.dtype)
    print("output device:", per_view_logits.device)
    print("output stats: mean={:.6f}, std={:.6f}, min={:.6f}, max={:.6f}".format(
        per_view_logits.mean().item(),
        per_view_logits.std().item(),
        per_view_logits.min().item(),
        per_view_logits.max().item()
    ))

    # 顺带测试单视角forward_single_view，5维输入 [B, T, C, H, W]
    single_view_input = torch.randn(2, 16, 3, 224, 224)
    single_logits = model.forward_single_view(single_view_input)
    print("\nsingle view logits shape:", single_logits.shape) # [B, num_actions]
