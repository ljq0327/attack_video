import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tbcnet.MVA import TBCNet_test


class TBCNetClassifier(nn.Module):
    def __init__(self, num_actions, num_views=3, device="cpu"):
        super().__init__()
        self.num_views = num_views
        self.model = TBCNet_test(
            view_num=num_views,
            low_feature_dim=256,
            high_feature_dim=128,
            num_actions=num_actions,
            device=device,
            name="tbcnet",
        )

    def forward(self, inputs):
        logits = self.forward_logits(inputs)
        return {"per_view_logits": logits.unsqueeze(1)}

    def _ensure_view_count(self, inputs):
        if inputs.dim() == 5:
            inputs = inputs.unsqueeze(1)
        if inputs.dim() != 6:
            raise ValueError(f"TBCNet expects [B,V,T,C,H,W] or [B,T,C,H,W], got {tuple(inputs.shape)}")

        batch_size, view_count, frames, channels, height, width = inputs.shape
        if view_count == self.num_views:
            return inputs
        if view_count > self.num_views:
            return inputs[:, :self.num_views]

        pad = inputs[:, -1:].expand(
            batch_size,
            self.num_views - view_count,
            frames,
            channels,
            height,
            width,
        )
        return torch.cat([inputs, pad], dim=1)

    def forward_logits(self, xs):
        xs = self._ensure_view_count(xs)
        low_fs = []
        for v in range(self.model.view):
            x = xs[:, v]
            low_f = self.model.encoder(x)
            low_fs.append(low_f)

        com_all = torch.cat(low_fs, -1)
        com_q, _ = self.model.fusion_net(com_all)

        com_fs = torch.stack(low_fs, dim=0)
        com_f = self.model.crossTrans(com_q, com_fs)
        com_f = self.model.pool2D(com_f).squeeze(-1).squeeze(-1)
        com_n = F.normalize(self.model.Linear_c(com_f), dim=1)
        com_n = self.model.bn_h(com_n)

        # 跳过原来的 Softmax，只取 Linear 输出
        logits = self.model.Classifier[1](com_n)
        return logits
