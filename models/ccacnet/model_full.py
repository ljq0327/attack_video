import torch
import torch.nn as nn

from models.ccacnet.backbone import X3DBackbone


def build_gn(num_channels, max_groups=8):
    """自动选择合适的 GroupNorm 组数。"""
    for g in [max_groups, 4, 2, 1]:
        if num_channels % g == 0:
            return nn.GroupNorm(g, num_channels)
    return nn.GroupNorm(1, num_channels)

class IntraViewActionRefinementModule(nn.Module):
    """
    IARM: 视角内动作细化模块。
    输入:
        x: [B*V, D, Tb]
    输出:
        delta: [B*V, D]，时序细化残差
    """
    def __init__(
        self,
        dim,
        hidden_ratio=0.5,
        kernel_size=3,
        dropout=0.0,
    ):
        super().__init__()

        hidden_dim = int(dim * hidden_ratio)
        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv1d(dim, hidden_dim, kernel_size=kernel_size, padding=padding, bias=False),
            build_gn(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, dim, kernel_size=1, bias=False),
            build_gn(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        x: [B*V, D, Tb]
        """
        out = self.block(x)  # [B*V, D, Tb]
        out = out.mean(dim=-1)  # [B*V, D]
        return out

class ViewWeightEstimator(nn.Module):
    """
    VWE: 视角权重估计模块。
    输入:
        z: [B, V, D]
    输出:
        center:  [B, D]，动作一致性参考
        weights: [B, V, 1]，视角权重
    """
    def __init__(self, dim, hidden_ratio=0.5, dropout=0.0):
        super().__init__()

        hidden_dim = int(dim * hidden_ratio)

        self.score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z):
        """
        z: [B, V, D]
        """
        score = self.score(z)  # [B, V, 1]
        weights = torch.softmax(score, dim=1)  # [B, V, 1]
        center = (weights * z).sum(dim=1)  # [B, D]
        return center, weights


class CompensationDirectionPredictor(nn.Module):
    """
    CDP: 补偿方向预测模块。
    输入:
        z: [B, V, D]
    输出:
        pred_compensation_direction: [B, V, D]
    目标:
        预测每个视角特征相对于动作一致性参考的补偿方向
    """
    def __init__(
        self,
        dim,
        hidden_ratio=0.5,
        dropout=0.1,
        zero_init_direction=True,
    ):
        super().__init__()

        hidden_dim = int(dim * hidden_ratio)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

        if zero_init_direction:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, z):
        """
        z: [B, V, D]
        """
        pred_compensation_direction = self.net(z)  # [B, V, D]
        return pred_compensation_direction


class CrossViewModel(nn.Module):
    """
    CCAC-Net 主体：X3D + IARM + IACM。
    """
    def __init__(
        self,
        num_actions,
        feature_dim=512,

        expert_kernel_size=3,
        expert_hidden_ratio=0.5,
        expert_dropout=0.0,

        direction_hidden_ratio=0.5,
        direction_dropout=0.1,
        compensation_strength=0.3,
        zero_init_direction=True,
        use_auxiliary_modules=True,
    ):
        super().__init__()

        self.num_actions = num_actions

        self.feature_dim = feature_dim

        self.compensation_strength = compensation_strength
        self.use_auxiliary_modules = use_auxiliary_modules

        self.backbone = X3DBackbone()

        self.intra_view_refiner = IntraViewActionRefinementModule(
            dim=self.feature_dim,
            hidden_ratio=expert_hidden_ratio,
            kernel_size=expert_kernel_size,
            dropout=expert_dropout,
        )

        self.view_weight_estimator = ViewWeightEstimator(
            dim=self.feature_dim,
            hidden_ratio=0.5,
            dropout=0.0,
        )

        self.compensation_direction_predictor = CompensationDirectionPredictor(
            dim=self.feature_dim,
            hidden_ratio=direction_hidden_ratio,
            dropout=direction_dropout,
            zero_init_direction=zero_init_direction,
        )

        self.action_classifier = nn.Linear(self.feature_dim, num_actions)

    def set_auxiliary_modules_enabled(self, enabled):
        self.use_auxiliary_modules = enabled

    def forward(self, inputs,):
        """
        inputs: [B, V, T, C, H, W]
        """
        B, V, T, C, H, W = inputs.shape

        per_view_feats_list = []
        # 逐视角提取共享时空特征
        for v_idx in range(V):
            v_input = inputs[:, v_idx]  # [B, T, C, H, W]

            v_feat = self.backbone(v_input)  # [B, D, Tb, 1, 1]
            v_feat = v_feat.squeeze(-1).squeeze(-1)  # [B, D, Tb]
            v_feat = v_feat.permute(0, 2, 1)  # [B, Tb, D]

            per_view_feats_list.append(v_feat)

        per_view_feats_seq = torch.stack(per_view_feats_list, dim=1)  # [B, V, Tb, D]

        # 对每个视角的时序特征做平均池化
        per_view_feat = per_view_feats_seq.mean(dim=2)  # [B, V, D]

        if self.use_auxiliary_modules:
            # IARM: 时序细化与残差融合
            x_seq = per_view_feats_seq.permute(0, 1, 3, 2).contiguous()  # [B, V, D, Tb]
            x_seq = x_seq.view(B * V, self.feature_dim, -1)  # [B*V, D, Tb]
            temporal_refinement_delta = self.intra_view_refiner(x_seq)  # [B*V, D]
            temporal_refinement_delta = temporal_refinement_delta.view(B, V, self.feature_dim)  # [B, V, D]
            refined_view_repr = per_view_feat + temporal_refinement_delta  # [B, V, D]

            # IACM: 构造参考并预测补偿方向
            if V > 1:
                # 构造动作一致性参考
                action_consistent_reference, _ = self.view_weight_estimator(refined_view_repr)  # [B, D]
                reference_logits = self.action_classifier(action_consistent_reference)  # [B, num_actions]
                compensation_target = action_consistent_reference.detach().unsqueeze(1) - refined_view_repr  # [B, V, D]
            else:
                reference_logits = None
                action_consistent_reference = refined_view_repr.mean(dim=1)
                compensation_target = torch.zeros_like(refined_view_repr)
            # 预测每个视角的补偿方向
            pred_compensation_direction = self.compensation_direction_predictor(refined_view_repr)  # [B, V, D]
            # 残差补偿得到校准后的视角表示
            calibrated_view_repr = refined_view_repr + self.compensation_strength * pred_compensation_direction
        else:
            reference_logits = None
            temporal_refinement_delta = torch.zeros_like(per_view_feat)
            refined_view_repr = per_view_feat
            compensation_target = torch.zeros_like(refined_view_repr)
            pred_compensation_direction = torch.zeros_like(refined_view_repr)
            calibrated_view_repr = refined_view_repr

        # 视角级分类
        per_view_logits = self.action_classifier(calibrated_view_repr)  # [B, V, num_actions]

        outputs = {
            "per_view_logits": per_view_logits,
            "pred_compensation_direction": pred_compensation_direction,
            "compensation_target": compensation_target,
            "reference_logits": reference_logits,
            "per_view_feat": per_view_feat,
            "refined_view_repr": refined_view_repr,
            "calibrated_view_repr": calibrated_view_repr,
            "temporal_refinement_delta": temporal_refinement_delta,
        }

        return outputs

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, V, T, C, H, W = 8, 3, 16, 3, 224, 224
    num_actions = 10

    x = torch.randn(B, V, T, C, H, W).to(device)

    model = CrossViewModel(
        num_actions=num_actions,
        feature_dim=512,

        expert_kernel_size=3,
        expert_hidden_ratio=0.5,
        expert_dropout=0.0,

        compensation_strength=0.3,
        zero_init_direction=True,
    ).to(device)

    model.train()

    outputs = model(x,)

    print("per_view_logits:", outputs["per_view_logits"].shape)
    print("pred_compensation_direction:", outputs["pred_compensation_direction"].shape)
    print("compensation_target:", outputs["compensation_target"].shape)
    print("per_view_feat:", outputs["per_view_feat"].shape)
    print("refined_view_repr:", outputs["refined_view_repr"].shape)
    print("calibrated_view_repr:", outputs["calibrated_view_repr"].shape)

    model.eval()
